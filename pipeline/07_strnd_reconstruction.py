#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    # If you truly only want hours+minutes, return f"{hrs}h {mins}m"
    return f"{hrs}h {mins}m {sec}s"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_name")
    ap.add_argument("sample_id")
    ap.add_argument("--suffix", type=str, default="LCC")

    # STRND args (defaults match get_strnd_positions_df signature)
    ap.add_argument("--node_embedding_components", type=int, default=32)
    ap.add_argument("--umap_n_neighbors", type=int, default=120)
    ap.add_argument("--umap_min_dist", type=float, default=0.5)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--weight_to_distance", action="store_true", default=False)
    ap.add_argument("--weighted", action="store_true", default=False)

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

    # NEW: enforce full connectivity; do NOT take/prune LCC
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

    print("Running STRND get_strnd_positions_df(...)")
    t0 = time.perf_counter()

    positions_df = get_strnd_positions_df(
        edgelist_df,
        node_embedding_components=args.node_embedding_components,
        dim=2,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        weighted=args.weighted,
        weight_to_distance=args.weight_to_distance,
        p=args.p,
        q=args.q,
        workers=args.workers,
        directed=False,
        random_state=args.random_state,
        verbose=True,
    )

    t1 = time.perf_counter()
    print(f"STRND runtime: {_format_h_m_s(t1 - t0)}")

    out_filename = os.path.join(
        scratch_prefix,
        f"{args.sample_id}_positions_{args.suffix}_STRND_d{args.node_embedding_components}_rnd{args.random_state}.csv"
    )
    print(f"Saving positions to: {out_filename}")
    positions_df.to_csv(out_filename, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
