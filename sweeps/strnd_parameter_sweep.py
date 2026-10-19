#!/usr/bin/env python3
"""
STRND parameter sweep on an experimental graph (Supplementary Figs. S3-S4).

Same input and output naming as pipeline/07_strnd_reconstruction.py, but runs STRND
over a grid of parameters, one stage at a time (--sweep_stage):
  0: single run with the scalar arguments
  1: UMAP n_neighbors x min_dist
  2: node embedding dimension
  3: node2vec p x q
Parameters that are not swept are held at the scalar arguments (--umap_n_neighbors, ...).

Output, in $SPAN_DATA_ROOT/scratch/<project>/:
  <sample>_positions_<suffix>_STRND_d<dim>_nn<n>_md<min_dist>_p<p>_q<q>_rnd<seed>.csv  (one per run)
  <sample>_STRND_sweep_stage<stage>_<suffix>_summary.csv
Existing outputs are skipped unless --overwrite is given.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

from graph_tool import load_graph, Graph
from graph_tool.topology import label_components

from strnd import get_strnd_positions_df


# Root data directory (contains scratch/<project>/). Override with SPAN_DATA_ROOT.
DATA_ROOT = os.environ.get("SPAN_DATA_ROOT", "./orcd")


def _scratch_prefix_for(project_name: str) -> str:
    return os.path.join(DATA_ROOT, "scratch", project_name) + "/"


def _assert_fully_connected(g: Graph) -> None:
    """
    Exit with an error unless the graph has exactly 1 connected component
    (undirected connectivity).
    """
    if g.num_vertices() == 0:
        raise RuntimeError("Graph has 0 vertices; cannot be 'fully connected'.")

    comp, hist = label_components(g, directed=False)
    n_components = len(hist)

    if n_components != 1:
        # hist[k] = size of component k
        largest = int(hist.max()) if len(hist) else 0
        raise RuntimeError(
            f"Graph is NOT fully connected: components={n_components}, "
            f"largest_component={largest:,}/{g.num_vertices():,} vertices."
        )


def _format_h_m_s(seconds: float) -> str:
    total = int(round(seconds))
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins}m {sec}s"


def _float_tag(x: float) -> str:
    """Filename-safe compact float representation."""
    return f"{x:g}".replace("-", "m").replace(".", "p")


def _gt_to_edgelist_df(g: Graph, *, weighted: bool) -> pd.DataFrame:
    """
    Convert an undirected graph_tool graph to a dense edge list DataFrame.

    Uses ORIGINAL vertex IDs (graph_tool internal indices) with no pruning/remapping.

    Columns:
      - always: ["source", "target"]
      - if weighted: ["source", "target", "weight"]

    For undirected graphs we canonicalize each edge as source <= target.
    """
    if weighted:
        if "weight" not in g.ep:
            raise RuntimeError(
                "Requested --weighted but edge property 'weight' was not found. "
                f"Available edge properties: {list(g.ep.keys())}"
            )
        w_ep = g.ep["weight"]
    else:
        w_ep = None

    sources = []
    targets = []
    weights = [] if weighted else None

    for e in g.edges():
        s = int(e.source())
        t = int(e.target())
        if t < s:
            s, t = t, s
        sources.append(s)
        targets.append(t)
        if weighted:
            weights.append(float(w_ep[e]))

    data = {
        "source": np.asarray(sources, dtype=np.int64),
        "target": np.asarray(targets, dtype=np.int64),
    }
    if weighted:
        data["weight"] = np.asarray(weights, dtype=np.float64)

    return pd.DataFrame(data)


def _parameter_sets(args):
    """
    Yield dictionaries describing the STRND runs to perform.

    sweep_stage = 0:
        One ordinary run using the scalar arguments.

    sweep_stage = 1:
        Sweep only UMAP n_neighbors x min_dist.
        Hold embedding dimension, p, and q fixed.

    sweep_stage = 2:
        Sweep only node embedding dimension.
        Use the scalar UMAP settings selected from stage 1.

    sweep_stage = 3:
        Sweep p x q.
        Use the scalar UMAP settings and embedding dimension selected earlier.
    """
    base = dict(
        node_embedding_components=args.node_embedding_components,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        p=args.p,
        q=args.q,
    )

    if args.sweep_stage == 0:
        yield base
        return

    if args.sweep_stage == 1:
        for n_neighbors, min_dist in itertools.product(
            args.sweep_umap_n_neighbors,
            args.sweep_umap_min_dist,
        ):
            params = base.copy()
            params["umap_n_neighbors"] = n_neighbors
            params["umap_min_dist"] = min_dist
            yield params
        return

    if args.sweep_stage == 2:
        for components in args.sweep_node_embedding_components:
            params = base.copy()
            params["node_embedding_components"] = components
            yield params
        return

    if args.sweep_stage == 3:
        for p, q in itertools.product(args.sweep_p, args.sweep_q):
            params = base.copy()
            params["p"] = p
            params["q"] = q
            yield params
        return

    raise ValueError(f"Unsupported sweep stage: {args.sweep_stage}")


def _run_tag(params: dict, random_state: int) -> str:
    return (
        f"d{params['node_embedding_components']}"
        f"_nn{params['umap_n_neighbors']}"
        f"_md{_float_tag(params['umap_min_dist'])}"
        f"_p{_float_tag(params['p'])}"
        f"_q{_float_tag(params['q'])}"
        f"_rnd{random_state}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_name")
    ap.add_argument("sample_id")
    ap.add_argument("--suffix", type=str, default="LCC")

    # STRND scalar args. These retain their original meaning and are also the
    # fixed values for parameters that are NOT being swept in a given stage.
    ap.add_argument("--node_embedding_components", type=int, default=64)
    ap.add_argument("--umap_n_neighbors", type=int, default=15)
    ap.add_argument("--umap_min_dist", type=float, default=1.0)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--weight_to_distance", action="store_true", default=False)
    ap.add_argument("--weighted", action="store_true", default=False)

    # Staged sweep.
    #
    # 0 = no sweep (original behavior)
    # 1 = UMAP n_neighbors x min_dist
    # 2 = embedding dimension
    # 3 = Node2Vec p x q
    ap.add_argument(
        "--sweep_stage",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help=(
            "0: one normal STRND run; "
            "1: sweep UMAP n_neighbors x min_dist; "
            "2: sweep embedding dimension; "
            "3: sweep Node2Vec p x q"
        ),
    )

    # Recommended coarse grids. Override any of these directly on the command line.
    ap.add_argument(
        "--sweep_umap_n_neighbors",
        type=int,
        nargs="+",
        default=[10, 15, 30, 60, 120, 256],
    )
    ap.add_argument(
        "--sweep_umap_min_dist",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.75, 1.0],
    )
    ap.add_argument(
        "--sweep_node_embedding_components",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
    )
    ap.add_argument(
        "--sweep_p",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.5, 2.0],
    )
    ap.add_argument(
        "--sweep_q",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.5, 2.0],
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Recompute runs even if the expected output CSV already exists.",
    )
    ap.add_argument(
        "--continue_on_error",
        action="store_true",
        default=False,
        help="Continue the sweep if one parameter combination fails.",
    )

    args = ap.parse_args()

    scratch_prefix = _scratch_prefix_for(args.project_name)
    os.makedirs(scratch_prefix, exist_ok=True)

    in_filename = os.path.join(
        scratch_prefix,
        f"{args.sample_id}_barcodesGraph_{args.suffix}.gt.gz"
    )
    if not os.path.exists(in_filename):
        raise FileNotFoundError(f"Input graph not found: {in_filename}")

    print(f"Loading graph: {in_filename}")
    g = load_graph(in_filename)
    print(f"Graph: |V|={g.num_vertices():,}  |E|={g.num_edges():,}")

    print("Checking graph connectivity (must be 100% connected)...")
    try:
        _assert_fully_connected(g)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    print("Connectivity OK: graph is fully connected.")

    print(f"Building edge list DataFrame (weighted={args.weighted})...")
    edgelist_df = _gt_to_edgelist_df(g, weighted=args.weighted)
    print(f"Edgelist rows: {len(edgelist_df):,}")
    print(edgelist_df.head())

    parameter_sets = list(_parameter_sets(args))
    print(
        f"\nSweep stage {args.sweep_stage}: "
        f"{len(parameter_sets)} STRND run(s) will be performed."
    )

    if args.sweep_stage == 1:
        print(
            "Stage 1: sweeping UMAP n_neighbors x min_dist; "
            f"holding embedding_components={args.node_embedding_components}, "
            f"p={args.p:g}, q={args.q:g} fixed."
        )
    elif args.sweep_stage == 2:
        print(
            "Stage 2: sweeping node embedding dimension; "
            f"holding UMAP n_neighbors={args.umap_n_neighbors}, "
            f"min_dist={args.umap_min_dist:g}, p={args.p:g}, q={args.q:g} fixed."
        )
    elif args.sweep_stage == 3:
        print(
            "Stage 3: sweeping p x q; "
            f"holding embedding_components={args.node_embedding_components}, "
            f"UMAP n_neighbors={args.umap_n_neighbors}, "
            f"min_dist={args.umap_min_dist:g} fixed."
        )

    summary_rows = []
    sweep_t0 = time.perf_counter()

    for run_idx, params in enumerate(parameter_sets, start=1):
        tag = _run_tag(params, args.random_state)
        out_filename = os.path.join(
            scratch_prefix,
            f"{args.sample_id}_positions_{args.suffix}_STRND_{tag}.csv"
        )

        print("\n" + "=" * 80)
        print(f"Run {run_idx}/{len(parameter_sets)}")
        print(
            "Parameters: "
            f"embedding_components={params['node_embedding_components']}, "
            f"umap_n_neighbors={params['umap_n_neighbors']}, "
            f"umap_min_dist={params['umap_min_dist']}, "
            f"p={params['p']}, q={params['q']}, "
            f"random_state={args.random_state}"
        )
        print(f"Output: {out_filename}")

        # Resume-friendly behavior: do not recompute a finished reconstruction.
        if os.path.exists(out_filename) and not args.overwrite:
            print("Output already exists; skipping. Use --overwrite to recompute.")
            summary_rows.append(
                {
                    **params,
                    "random_state": args.random_state,
                    "runtime_seconds": np.nan,
                    "status": "skipped_exists",
                    "output_file": out_filename,
                }
            )
            continue

        t0 = time.perf_counter()
        status = "ok"
        error_message = ""

        try:
            positions_df = get_strnd_positions_df(
                edgelist_df,
                node_embedding_components=params["node_embedding_components"],
                dim=2,
                umap_n_neighbors=params["umap_n_neighbors"],
                umap_min_dist=params["umap_min_dist"],
                weighted=args.weighted,
                weight_to_distance=args.weight_to_distance,
                p=params["p"],
                q=params["q"],
                workers=args.workers,
                directed=False,
                random_state=args.random_state,
                verbose=True,
            )

            positions_df.to_csv(out_filename, index=False)

        except Exception as e:
            status = "error"
            error_message = repr(e)
            print(f"ERROR in run {run_idx}: {error_message}", file=sys.stderr)

            if not args.continue_on_error:
                raise

        runtime_seconds = time.perf_counter() - t0
        print(f"Run runtime: {_format_h_m_s(runtime_seconds)}")

        summary_rows.append(
            {
                **params,
                "random_state": args.random_state,
                "runtime_seconds": runtime_seconds,
                "status": status,
                "error": error_message,
                "output_file": out_filename,
            }
        )

        # Update the manifest after every run so progress survives interruption.
        summary_df = pd.DataFrame(summary_rows)
        summary_filename = os.path.join(
            scratch_prefix,
            f"{args.sample_id}_STRND_sweep_stage{args.sweep_stage}_{args.suffix}_summary.csv"
        )
        summary_df.to_csv(summary_filename, index=False)
        print(f"Updated sweep summary: {summary_filename}")

    sweep_runtime = time.perf_counter() - sweep_t0
    print("\n" + "=" * 80)
    print(f"All requested runs finished in {_format_h_m_s(sweep_runtime)}")

    summary_filename = os.path.join(
        scratch_prefix,
        f"{args.sample_id}_STRND_sweep_stage{args.sweep_stage}_{args.suffix}_summary.csv"
    )
    pd.DataFrame(summary_rows).to_csv(summary_filename, index=False)
    print(f"Final sweep summary: {summary_filename}")
    print("Done.")


if __name__ == "__main__":
    main()
