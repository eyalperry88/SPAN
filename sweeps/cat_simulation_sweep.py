#!/usr/bin/env python3
"""Sweep Poisson-disc bipartite graph parameters and reconstruct a 12x12 cat with STRND.

Simulates SPAN proximity graphs (Supplementary Fig. S2): barcodes are placed by Poisson-disc
sampling in a 500-um disk, split into two bipartite classes, and connected by edges whose
lengths follow a gamma distribution (mean degree 4). A 12x12 cat pattern is written into the
graph, and the result is reconstructed with STRND.

Default sweep:
    MIN_SPACING_UM             = 1, 2, 3, 4, 5
    TARGET_MEAN_EDGE_LENGTH_UM = 5, 10, 20, 40

A '1' is a WRITTEN bit: remove graph vertices inside the corresponding
20-um-diameter disc, creating a hole. A '0' leaves vertices untouched.
The cat is centered on a 30-um-pitch square lattice in the 500-um disk.

Connected components are found with SciPy, so this script does not need graph-tool.

Output per completed run (CSV):
    node_ID,bipartite_class,x,y
where x and y are *raw STRND embedding coordinates* (not micrometers).
Also writes sweep_summary.csv with run statistics and elapsed times.

Dependencies: numpy, pandas, scipy, strnd (all in environment.yml).

Examples:
    python -u cat_simulation_sweep.py
    python -u cat_simulation_sweep.py --output-dir cat_sweep
    python -u cat_simulation_sweep.py --min-spacings 5 --mean-edge-lengths 10
    python -u cat_simulation_sweep.py --overwrite --workers 8

Existing reconstruction CSVs are skipped by default (resume behavior).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


# Each row is left-to-right; first row is the TOP of the picture.
CAT_BITS = (
    "100000000001",
    "110000000011",
    "110000000011",
    "101011110101",
    "100000000001",
    "000000000000",
    "000100001000",
    "000000000000",
    "111100001111",
    "000000000000",
    "001001100100",
    "000110011000",
)
PATTERN_HASH = hashlib.sha1("".join(CAT_BITS).encode("ascii")).hexdigest()[:8]

DEFAULT_MIN_SPACINGS = (1, 2, 3, 4, 5)
DEFAULT_EDGE_LENGTHS = (5, 10, 20, 40)

# Simulation geometry and graph statistics.
DIAMETER_UM = 500.0
TARGET_AVG_DEGREE = 4.0
EDGE_GAMMA_SHAPE = 2.0
BIT_DIAMETER_UM = 20.0
BIT_PITCH_UM = 30.0
BASE_SEED = 123
STRND_SEED = 42

# STRND settings (the same for every sweep run).
NODE_EMBEDDING_COMPONENTS = 32
UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.5
P = 1.0
Q = 1.0
WORKERS = 4


def poisson_disk_sampling(radius: float, d: float, rng: np.random.Generator, k: int = 30) -> np.ndarray:
    """Bridson Poisson-disc sampling in a circle."""
    if d <= 0 or radius <= 0:
        raise ValueError("Poisson radius and spacing must be positive")
    cell_size = d / np.sqrt(2)
    grid_shape = int(np.ceil(2 * radius / cell_size)) + 1
    grid = np.full((grid_shape, grid_shape), -1, dtype=np.int32)
    samples: list[np.ndarray] = []
    active: list[int] = []

    def grid_coords(p: np.ndarray) -> tuple[int, int]:
        return int((p[0] + radius) / cell_size), int((p[1] + radius) / cell_size)

    def in_circle(p: np.ndarray) -> bool:
        return bool(p[0] ** 2 + p[1] ** 2 <= radius ** 2)

    def far_enough(p: np.ndarray) -> bool:
        gx, gy = grid_coords(p)
        for i in range(max(0, gx - 2), min(grid_shape, gx + 3)):
            for j in range(max(0, gy - 2), min(grid_shape, gy + 3)):
                idx = grid[i, j]
                if idx >= 0 and np.sum((p - samples[idx]) ** 2) < d * d:
                    return False
        return True

    while True:
        p0 = rng.uniform(-radius, radius, size=2)
        if in_circle(p0):
            samples.append(p0)
            active.append(0)
            grid[grid_coords(p0)] = 0
            break

    while active:
        slot = int(rng.integers(len(active)))
        p = samples[active[slot]]
        found = False
        for _ in range(k):
            theta = rng.uniform(0, 2 * np.pi)
            r = d * np.sqrt(1 + 3 * rng.random())
            q = p + r * np.array([np.cos(theta), np.sin(theta)])
            if in_circle(q) and far_enough(q):
                idx = len(samples)
                samples.append(q)
                active.append(idx)
                grid[grid_coords(q)] = idx
                found = True
                break
        if not found:
            active[slot] = active[-1]
            active.pop()
    return np.asarray(samples, dtype=np.float64)


def sample_candidate_edges(
    n_samples: int,
    positions: np.ndarray,
    node_type: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    left_tree: cKDTree,
    right_tree: cKDTree,
    circle_radius: float,
    gamma_shape: float,
    gamma_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample edges between opposite bipartite classes."""
    n = len(positions)
    source = rng.integers(0, n, size=n_samples, dtype=np.int64)
    source_pos = positions[source]
    desired_r = rng.gamma(gamma_shape, gamma_scale, size=n_samples)

    max_possible = circle_radius + np.linalg.norm(source_pos, axis=1)
    impossible = desired_r > max_possible
    while np.any(impossible):
        desired_r[impossible] = rng.gamma(
            gamma_shape, gamma_scale, size=int(np.count_nonzero(impossible))
        )
        impossible = desired_r > max_possible

    theta = rng.uniform(0, 2 * np.pi, size=n_samples)
    target_xy = source_pos + desired_r[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    outside = np.sum(target_xy ** 2, axis=1) > circle_radius ** 2
    while np.any(outside):
        theta[outside] = rng.uniform(0, 2 * np.pi, size=int(np.count_nonzero(outside)))
        target_xy[outside] = source_pos[outside] + desired_r[outside, None] * np.column_stack(
            (np.cos(theta[outside]), np.sin(theta[outside]))
        )
        outside = np.sum(target_xy ** 2, axis=1) > circle_radius ** 2

    target = np.empty(n_samples, dtype=np.int64)
    is_left = node_type[source] == 0
    if np.any(is_left):
        _, idx = right_tree.query(target_xy[is_left], k=1)
        target[is_left] = right_indices[idx]
    if np.any(~is_left):
        _, idx = left_tree.query(target_xy[~is_left], k=1)
        target[~is_left] = left_indices[idx]
    return np.minimum(source, target), np.maximum(source, target)


def generate_bipartite_edges(
    positions: np.ndarray,
    node_type: np.ndarray,
    target_mean_edge_um: float,
    target_avg_degree: float,
    circle_radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exactly ~target_avg_degree * N / 2 unique edges."""
    n = len(positions)
    if n < 2:
        raise RuntimeError("Not enough points to create graph")
    left_indices = np.flatnonzero(node_type == 0)
    right_indices = np.flatnonzero(node_type == 1)
    if not len(left_indices) or not len(right_indices):
        raise RuntimeError("Both bipartite classes must have nodes")
    left_tree = cKDTree(positions[left_indices])
    right_tree = cKDTree(positions[right_indices])
    target_e = int(round(target_avg_degree * n / 2))
    gamma_scale = target_mean_edge_um / EDGE_GAMMA_SHAPE

    def candidates(count: int) -> np.ndarray:
        u, v = sample_candidate_edges(
            count, positions, node_type, left_indices, right_indices,
            left_tree, right_tree, circle_radius, EDGE_GAMMA_SHAPE,
            gamma_scale, rng,
        )
        return np.unique(u * np.int64(n) + v)

    keys = candidates(int(np.ceil(1.10 * target_e)))
    while len(keys) < target_e:
        remaining = target_e - len(keys)
        keys = np.union1d(keys, candidates(max(10_000, int(np.ceil(1.2 * remaining)))))
    if len(keys) > target_e:
        keys = keys[rng.choice(len(keys), size=target_e, replace=False)]
    edges = np.column_stack((keys // n, keys % n)).astype(np.int64)
    if np.any(node_type[edges[:, 0]] == node_type[edges[:, 1]]):
        raise AssertionError("Bipartite edge constraint violated")
    return edges


def keep_lcc(
    positions: np.ndarray,
    node_type: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the largest undirected connected component; dense reindexing."""
    n = len(positions)
    if n == 0 or len(edges) == 0:
        raise RuntimeError("No nodes/edges remain for the largest connected component")
    adjacency = csr_matrix(
        (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    _, labels = connected_components(adjacency, directed=False, return_labels=True)
    largest_label = int(np.argmax(np.bincount(labels)))
    keep = labels == largest_label
    old_ids = np.flatnonzero(keep)
    old_to_new = np.full(n, -1, dtype=np.int64)
    old_to_new[old_ids] = np.arange(len(old_ids), dtype=np.int64)
    surviving = keep[edges[:, 0]] & keep[edges[:, 1]]
    return positions[old_ids], node_type[old_ids], old_to_new[edges[surviving]]


def cat_bit_centers(pitch_um: float, graph_radius_um: float, bit_diameter_um: float) -> np.ndarray:
    """Returns (x,y) centers of all written/1 bits; row 0 is at +y."""
    n_rows = len(CAT_BITS)
    n_cols = len(CAT_BITS[0])
    if any(len(row) != n_cols or set(row) - {"0", "1"} for row in CAT_BITS):
        raise ValueError("Invalid binary cat pattern")
    # Verify the entire 12x12 lattice, not only the 1 bits, fits in the disk.
    outer_radius = np.hypot((n_cols - 1) * pitch_um / 2, (n_rows - 1) * pitch_um / 2)
    if outer_radius + bit_diameter_um / 2 > graph_radius_um:
        raise ValueError("The cat bit lattice does not fit within the graph circle")
    centers = []
    for row_idx, row in enumerate(CAT_BITS):
        for col_idx, bit in enumerate(row):
            if bit == "1":
                x = (col_idx - (n_cols - 1) / 2) * pitch_um
                y = ((n_rows - 1) / 2 - row_idx) * pitch_um
                centers.append((x, y))
    return np.asarray(centers, dtype=np.float64)


def subtract_cat_bits(
    positions: np.ndarray,
    node_type: np.ndarray,
    edges: np.ndarray,
    written_centers: np.ndarray,
    bit_radius_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Remove nodes inside 1-bit discs; then keep the storage LCC."""
    remove = np.zeros(len(positions), dtype=bool)
    for node_ids in cKDTree(positions).query_ball_point(written_centers, r=bit_radius_um):
        remove[node_ids] = True
    removed_count = int(np.count_nonzero(remove))
    keep = ~remove
    old_ids = np.flatnonzero(keep)
    old_to_new = np.full(len(positions), -1, dtype=np.int64)
    old_to_new[old_ids] = np.arange(len(old_ids), dtype=np.int64)
    surviving = keep[edges[:, 0]] & keep[edges[:, 1]]
    masked_edges = old_to_new[edges[surviving]]
    storage_positions, storage_types, storage_edges = keep_lcc(
        positions[old_ids], node_type[old_ids], masked_edges
    )
    return storage_positions, storage_types, storage_edges, removed_count


def reconstruct_strnd(
    edges: np.ndarray,
    node_type: np.ndarray,
    seed: int,
    workers: int,
    verbose: bool,
) -> pd.DataFrame:
    """Reconstruct 2D positions with STRND using the fixed settings above."""
    try:
        from strnd import get_strnd_positions_df
    except ImportError as exc:
        raise ImportError(
            "Cannot import 'strnd'. Install it with `pip install strnd` "
            "(included in environment.yml)."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    edge_df = pd.DataFrame({"source": edges[:, 0], "target": edges[:, 1]})
    reconstruction_df = get_strnd_positions_df(
        edge_df,
        node_embedding_components=NODE_EMBEDDING_COMPONENTS,
        dim=2,
        umap_n_neighbors=UMAP_N_NEIGHBORS,
        umap_min_dist=UMAP_MIN_DIST,
        weighted=False,
        weight_to_distance=False,
        p=P,
        q=Q,
        workers=workers,
        directed=False,
        random_state=seed,
        verbose=verbose,
    ).sort_values("node_ID")
    expected_ids = np.arange(len(node_type), dtype=np.int64)
    actual_ids = reconstruction_df["node_ID"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_ids, expected_ids):
        raise RuntimeError("STRND returned node IDs that are not dense 0...N-1")
    return pd.DataFrame(
        {
            "node_ID": expected_ids,
            "bipartite_class": np.where(node_type == 0, "left", "right"),
            "x": reconstruction_df["x"].to_numpy(),
            "y": reconstruction_df["y"].to_numpy(),
        }
    )


def filename_for_run(
    min_spacing: int, mean_edge_length: int, graph_diameter: float,
    bit_diameter: float, bit_pitch: float, base_seed: int, strnd_seed: int,
) -> str:
    return (
        f"cat12x12_{PATTERN_HASH}_minspacing{min_spacing:g}um_"
        f"meanedge{mean_edge_length:g}um_graphdiam{graph_diameter:g}um_"
        f"bitdiam{bit_diameter:g}um_pitch{bit_pitch:g}um_"
        f"base{base_seed}_strnd{strnd_seed}_reconstruction.csv"
    )


def save_summary(output_dir: Path, records: dict[tuple[int, int], dict]) -> None:
    """Atomically replace the manifest after each run."""
    columns = [
        "min_spacing_um", "target_mean_edge_length_um", "graph_diameter_um",
        "bit_diameter_um", "bit_pitch_um", "base_seed", "strnd_seed",
        "base_nodes", "base_edges", "removed_nodes", "storage_nodes", "storage_edges",
        "actual_mean_storage_edge_length_um", "generation_seconds", "strnd_seconds",
        "status", "csv_file", "error",
    ]
    summary = pd.DataFrame(list(records.values())).reindex(columns=columns)
    if len(summary):
        summary = summary.sort_values(["min_spacing_um", "target_mean_edge_length_um"])
    temporary = output_dir / "sweep_summary.csv.tmp"
    summary.to_csv(temporary, index=False)
    temporary.replace(output_dir / "sweep_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("cat_strnd_sweep_output"))
    parser.add_argument("--min-spacings", type=int, nargs="+", default=list(DEFAULT_MIN_SPACINGS))
    parser.add_argument("--mean-edge-lengths", type=int, nargs="+", default=list(DEFAULT_EDGE_LENGTHS))
    parser.add_argument("--graph-diameter-um", type=float, default=DIAMETER_UM)
    parser.add_argument("--bit-diameter-um", type=float, default=BIT_DIAMETER_UM)
    parser.add_argument("--bit-pitch-um", type=float, default=BIT_PITCH_UM)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--strnd-seed", type=int, default=STRND_SEED)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--strnd-verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Rerun already existing reconstruction CSVs")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately if one run fails")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from strnd import get_strnd_positions_df  # noqa: F401 - early dependency check
    except ImportError as exc:
        raise ImportError(
            "STRND is unavailable in this environment. Install it with "
            "`pip install strnd` (included in environment.yml)."
        ) from exc
    if any(x <= 0 for x in (*args.min_spacings, *args.mean_edge_lengths)):
        raise ValueError("Sweep spacing and edge lengths must be positive")
    graph_radius = args.graph_diameter_um / 2
    centers = cat_bit_centers(args.bit_pitch_um, graph_radius, args.bit_diameter_um)
    n_total = len(args.min_spacings) * len(args.mean_edge_lengths)
    print(
        f"CAT STRND SWEEP: {n_total} runs | 12x12 pattern, {len(centers)} written/removed bits | "
        f"bit diameter {args.bit_diameter_um:g} um, pitch {args.bit_pitch_um:g} um | "
        f"graph diameter {args.graph_diameter_um:g} um | output: {args.output_dir}",
        flush=True,
    )
    print("Convention: '1' = written bit = 20-um disc removed; '0' = graph unchanged", flush=True)
    records: dict[tuple[int, int], dict] = {}
    manifest = args.output_dir / "sweep_summary.csv"
    if manifest.exists():
        for entry in pd.read_csv(manifest).to_dict(orient="records"):
            try:
                key = (int(entry["min_spacing_um"]), int(entry["target_mean_edge_length_um"]))
                records[key] = entry
            except (ValueError, KeyError, TypeError):
                continue
    failed = 0
    overall_start = time.perf_counter()
    job_index = 0

    # Identical physical point locations for all edge-length conditions at
    # each min-spacing; independent, seeded edge samples for each condition.
    for min_spacing in args.min_spacings:
        runs = []
        for mean_edge_length in args.mean_edge_lengths:
            job_index += 1
            outfile = args.output_dir / filename_for_run(
                min_spacing, mean_edge_length, args.graph_diameter_um,
                args.bit_diameter_um, args.bit_pitch_um, args.base_seed, args.strnd_seed
            )
            if outfile.exists() and not args.overwrite:
                print(f"[{job_index}/{n_total}] SKIP spacing={min_spacing} um, "
                      f"mean edge={mean_edge_length} um (CSV exists: {outfile.name})", flush=True)
                key = (min_spacing, mean_edge_length)
                if key not in records or records[key].get("csv_file") != outfile.name:
                    records[key] = {
                        "min_spacing_um": min_spacing,
                        "target_mean_edge_length_um": mean_edge_length,
                        "graph_diameter_um": args.graph_diameter_um,
                        "bit_diameter_um": args.bit_diameter_um,
                        "bit_pitch_um": args.bit_pitch_um,
                        "base_seed": args.base_seed,
                        "strnd_seed": args.strnd_seed,
                        "csv_file": outfile.name,
                        "status": "skipped_existing",
                    }
            else:
                runs.append((job_index, mean_edge_length, outfile))

        if not runs:
            save_summary(args.output_dir, records)
            continue
        t0 = time.perf_counter()
        print(f"Sampling Poisson-disc positions for min spacing={min_spacing} um...", flush=True)
        base_rng = np.random.default_rng(np.random.SeedSequence([args.base_seed, min_spacing, 0]))
        raw_positions = poisson_disk_sampling(graph_radius, min_spacing, base_rng)
        raw_types = np.zeros(len(raw_positions), dtype=np.uint8)
        raw_types[len(raw_types) // 2:] = 1
        base_rng.shuffle(raw_types)
        print(f"  {len(raw_positions):,} positions generated in "
              f"{time.perf_counter()-t0:.1f}s (will reuse for this spacing)", flush=True)

        for job_num, mean_edge_length, outfile in runs:
            run_start = time.perf_counter()
            record = {
                "min_spacing_um": min_spacing,
                "target_mean_edge_length_um": mean_edge_length,
                "graph_diameter_um": args.graph_diameter_um,
                "bit_diameter_um": args.bit_diameter_um,
                "bit_pitch_um": args.bit_pitch_um,
                "base_seed": args.base_seed,
                "strnd_seed": args.strnd_seed,
                "csv_file": outfile.name,
            }
            print(f"[{job_num}/{n_total}] START spacing={min_spacing} um, "
                  f"target mean edge={mean_edge_length} um", flush=True)
            try:
                edge_rng = np.random.default_rng(
                    np.random.SeedSequence([args.base_seed, min_spacing, mean_edge_length, 1])
                )
                edges = generate_bipartite_edges(
                    raw_positions, raw_types, mean_edge_length,
                    TARGET_AVG_DEGREE, graph_radius, edge_rng
                )
                base_positions, base_types, base_edges = keep_lcc(raw_positions, raw_types, edges)
                record["base_nodes"] = len(base_positions)
                record["base_edges"] = len(base_edges)
                storage_positions, storage_types, storage_edges, removed = subtract_cat_bits(
                    base_positions, base_types, base_edges, centers, args.bit_diameter_um / 2
                )
                record["removed_nodes"] = removed
                record["storage_nodes"] = len(storage_positions)
                record["storage_edges"] = len(storage_edges)
                diffs = storage_positions[storage_edges[:, 0]] - storage_positions[storage_edges[:, 1]]
                record["actual_mean_storage_edge_length_um"] = float(
                    np.linalg.norm(diffs, axis=1).mean()
                )
                del diffs, edges, base_positions, base_types, base_edges
                record["generation_seconds"] = round(time.perf_counter() - run_start, 2)
                strnd_start = time.perf_counter()
                print(f"[{job_num}/{n_total}] STRND starting: {len(storage_positions):,} nodes, "
                      f"{len(storage_edges):,} edges...", flush=True)
                output = reconstruct_strnd(
                    storage_edges, storage_types, args.strnd_seed, args.workers, args.strnd_verbose
                )
                record["strnd_seconds"] = round(time.perf_counter() - strnd_start, 2)
                # Temp file avoids considering an incomplete CSV valid on resume.
                tmp = outfile.with_name(outfile.name + ".tmp")
                output.to_csv(tmp, index=False)
                tmp.replace(outfile)
                record["status"] = "done"
                print(
                    f"[{job_num}/{n_total}] DONE spacing={min_spacing} um, "
                    f"mean edge={mean_edge_length} um | nodes={len(output):,} "
                    f"edges={len(storage_edges):,} | removed={removed:,} | "
                    f"STRND={record['strnd_seconds']:.1f}s | "
                    f"total={time.perf_counter()-run_start:.1f}s | {outfile.name}",
                    flush=True,
                )
                del output, storage_positions, storage_types, storage_edges
            except Exception as exc:
                failed += 1
                record["status"] = "failed"
                record["error"] = str(exc)
                print(f"[{job_num}/{n_total}] FAILED spacing={min_spacing} um, "
                      f"mean edge={mean_edge_length} um: {exc}", flush=True)
                traceback.print_exc()
            records[min_spacing, mean_edge_length] = record
            save_summary(args.output_dir, records)
            if record["status"] == "failed" and args.fail_fast:
                return 1
        del raw_positions, raw_types

    save_summary(args.output_dir, records)
    print(f"Sweep finished: {n_total} configurations; {failed} failed; "
          f"elapsed {time.perf_counter()-overall_start:.1f}s | "
          f"summary: {args.output_dir / 'sweep_summary.csv'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
