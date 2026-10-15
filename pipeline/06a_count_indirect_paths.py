"""
indirect_paths_pipeline.py

For every edge e=(u,v) in an undirected graph-tool graph, count the number of
SIMPLE indirect paths from u to v that do NOT use the edge (u,v), with:

  - exact length 3  -> saved as ep["indirectP3"]
  - exact length 5  -> saved as ep["indirectP5"]

A "simple" path here means no repeated vertices.

Usage (example):
  python indirect_paths_pipeline.py \
    --project myproj --sample-id S01 --suffix LCC \
    --workers 96 --job-size 10000

Input/output naming (defaults):
  input  = $SPAN_DATA_ROOT/scratch/{project}/{sample_id}_barcodesGraph_{suffix}.gt.gz
  output = $SPAN_DATA_ROOT/scratch/{project}/{sample_id}_barcodesGraph_{suffix}_indirectPaths.gt.gz
"""

import argparse
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from time import perf_counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

import graph_tool.all as gt


# --------------------------- CSR build ---------------------------

def build_csr_adj_from_graphtool(g: gt.Graph, verbose: bool = True):
    """
    Build undirected CSR adjacency (indptr, indices) and edge list edges_uv in the SAME
    iteration order as g.edges(), skipping self-loops.
    """
    t0 = perf_counter()
    n = int(g.num_vertices())
    m = int(g.num_edges())

    if verbose:
        print("[1/7] Building CSR adjacency (undirected=True)")
        print(f"      Graph: |V|={n:,}, |E|={m:,}")

    deg = np.zeros(n, dtype=np.int64)
    edges_uv = []

    it = g.edges()
    if verbose:
        it = tqdm(it, total=m, desc="Scan edges (deg)", mininterval=0.5)

    for e in it:
        u = int(e.source())
        v = int(e.target())
        if u == v:
            continue
        edges_uv.append((u, v))
        deg[u] += 1
        deg[v] += 1

    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(deg, out=indptr[1:])

    indices = np.empty(indptr[-1], dtype=np.int32)
    cursor = indptr[:-1].copy()

    it2 = edges_uv
    if verbose:
        it2 = tqdm(it2, total=len(edges_uv), desc="Fill CSR", mininterval=0.5)

    for (u, v) in it2:
        indices[cursor[u]] = v
        cursor[u] += 1
        indices[cursor[v]] = u
        cursor[v] += 1

    if verbose:
        dt = perf_counter() - t0
        print(f"      CSR built: indptr={indptr.shape}, indices={indices.shape} (nnz={indices.size:,}) in {dt:.2f}s")

    return indptr, indices, edges_uv


# --------------------------- Worker globals ---------------------------

_G = {}


def _init_worker(shm_name_indptr: str, shm_name_indices: str, n_indptr: int, n_indices: int):
    """Initializer runs once per worker process."""
    shm_i = shared_memory.SharedMemory(name=shm_name_indptr)
    shm_x = shared_memory.SharedMemory(name=shm_name_indices)

    indptr = np.ndarray((n_indptr,), dtype=np.int64, buffer=shm_i.buf)
    indices = np.ndarray((n_indices,), dtype=np.int32, buffer=shm_x.buf)

    _G["shm_i"] = shm_i
    _G["shm_x"] = shm_x
    _G["indptr"] = indptr
    _G["indices"] = indices


def _count_indirect_p3_p5_for_edge(u: int, v: int):
    """
    Count simple paths from u to v of exact length 3 and exact length 5,
    excluding the direct edge (u,v) in either direction.
    """
    indptr = _G["indptr"]
    indices = _G["indices"]

    count3 = 0
    count5 = 0

    # path[depth] = vertex at that depth (depth = #edges from u)
    # max depth needed is 5 => up to 6 vertices in path
    path = [-1] * 6
    path[0] = u

    def dfs(x: int, depth: int):
        nonlocal count3, count5

        if depth == 5:
            return

        for i in range(indptr[x], indptr[x + 1]):
            y = int(indices[i])

            # Ban the direct edge (u,v) in either direction
            if (x == u and y == v) or (x == v and y == u):
                continue

            next_depth = depth + 1

            # Enforce simple path: no repeated vertices already in current path
            repeated = False
            for k in range(depth + 1):
                if path[k] == y:
                    repeated = True
                    break
            if repeated:
                continue

            # If we reach v, only count exact target lengths; do not continue through v
            if y == v:
                if next_depth == 3:
                    count3 += 1
                elif next_depth == 5:
                    count5 += 1
                continue

            # Otherwise continue exploring (only if we still have depth budget)
            if next_depth < 5:
                path[next_depth] = y
                dfs(y, next_depth)

    dfs(u, 0)
    return count3, count5


def _worker_chunk(edge_chunk):
    """Compute indirect path counts for one chunk of edges."""
    out3 = np.zeros(len(edge_chunk), dtype=np.int32)
    out5 = np.zeros(len(edge_chunk), dtype=np.int32)

    for j, (u, v) in enumerate(edge_chunk):
        c3, c5 = _count_indirect_p3_p5_for_edge(u, v)
        out3[j] = c3
        out5[j] = c5

    return out3, out5


# --------------------------- Main compute ---------------------------

def compute_indirect_paths_parallel(
    g: gt.Graph,
    workers: int,
    job_size: int,
    verbose: bool = True,
    mp_start_method: str | None = None,
):
    """
    Returns:
      p3_prop (EdgePropertyMap "int")
      p5_prop (EdgePropertyMap "int")
      p3_arr (np.ndarray int32) in g.edges() order skipping self-loops
      p5_arr (np.ndarray int32) in g.edges() order skipping self-loops
      edges_uv (list of (u,v)) in same order
    """
    t_all = perf_counter()

    # Stage 1: CSR
    indptr, indices, edges_uv = build_csr_adj_from_graphtool(g, verbose=verbose)

    # Stage 2: Shared memory
    t0 = perf_counter()
    if verbose:
        print("[2/7] Creating shared memory for CSR arrays...")
        print(f"      indptr bytes:  {indptr.nbytes / 1e6:.1f} MB")
        print(f"      indices bytes: {indices.nbytes / 1e6:.1f} MB")

    shm_i = shared_memory.SharedMemory(create=True, size=indptr.nbytes)
    shm_x = shared_memory.SharedMemory(create=True, size=indices.nbytes)

    indptr_sh = np.ndarray(indptr.shape, dtype=indptr.dtype, buffer=shm_i.buf)
    indices_sh = np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_x.buf)
    indptr_sh[:] = indptr
    indices_sh[:] = indices

    if verbose:
        print(f"      Shared memory ready in {perf_counter() - t0:.2f}s")

    # Stage 3: Chunk edges
    t0 = perf_counter()
    if verbose:
        print("[3/7] Chunking edges...")
        print(f"      Work edges (no self-loops): {len(edges_uv):,}")
        print(f"      job_size={job_size:,}")

    chunks = [edges_uv[i:i + job_size] for i in range(0, len(edges_uv), job_size)]

    if verbose:
        print(f"      Total chunks: {len(chunks):,} (in {perf_counter() - t0:.2f}s)")

    # Stage 4: Parallel compute
    if verbose:
        print("[4/7] Counting indirect paths in parallel...")
        print(f"      workers={workers}")
        if mp_start_method:
            print(f"      mp_start_method={mp_start_method}")

    p3_arr = np.empty(len(edges_uv), dtype=np.int32)
    p5_arr = np.empty(len(edges_uv), dtype=np.int32)

    mp_context = None
    if mp_start_method:
        mp_context = mp.get_context(mp_start_method)

    try:
        t0 = perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
            initializer=_init_worker,
            initargs=(shm_i.name, shm_x.name, indptr.shape[0], indices.shape[0]),
        ) as ex:
            offset = 0
            for out3, out5 in tqdm(
                ex.map(_worker_chunk, chunks),
                total=len(chunks),
                desc="Chunks done",
                mininterval=0.5,
                disable=not verbose,
            ):
                n = len(out3)
                p3_arr[offset:offset + n] = out3
                p5_arr[offset:offset + n] = out5
                offset += n

        if verbose:
            dt = perf_counter() - t0
            rate = len(edges_uv) / dt if dt > 0 else float("inf")
            print(f"      Parallel counting complete in {dt:.2f}s  (~{rate:,.0f} edges/sec)")
    finally:
        # Stage 5: cleanup shared memory
        if verbose:
            print("[5/7] Cleaning up shared memory...")
        shm_i.close(); shm_i.unlink()
        shm_x.close(); shm_x.unlink()

    # Stage 6: write results back to graph-tool edge properties
    t0 = perf_counter()
    if verbose:
        print("[6/7] Writing results into graph-tool edge properties...")

    p3_prop = g.new_edge_property("int")
    p5_prop = g.new_edge_property("int")

    idx = 0
    it3 = g.edges()
    if verbose:
        it3 = tqdm(it3, total=g.num_edges(), desc="Assign counts", mininterval=0.5)

    for e in it3:
        u = int(e.source())
        v = int(e.target())
        if u == v:
            p3_prop[e] = 0
            p5_prop[e] = 0
        else:
            p3_prop[e] = int(p3_arr[idx])
            p5_prop[e] = int(p5_arr[idx])
            idx += 1

    if verbose:
        print(f"      Assignment done in {perf_counter() - t0:.2f}s")
        print(f"      Total compute time (incl. CSR/build/assign): {perf_counter() - t_all:.2f}s")

    return p3_prop, p5_prop, p3_arr, p5_arr, edges_uv


def count_stats_from_array(arr: np.ndarray):
    """Basic summary stats for one count array."""
    n_total = int(arr.size)
    n_nonzero = int(np.count_nonzero(arr))
    total_paths = int(arr.sum(dtype=np.int64))
    max_count = int(arr.max()) if n_total else 0
    return n_total, n_nonzero, total_paths, max_count


# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Count simple indirect paths of exact lengths 3 and 5 (excluding the edge itself) in parallel for a graph-tool .gt.gz file."
    )
    p.add_argument("--project", required=True, help="Project name (used for input/output file naming)")
    p.add_argument("--sample-id", required=True, help="Sample ID (used for input/output file naming)")
    p.add_argument("--suffix", default="", help="Suffix appended after sample-id (e.g. LCC)")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="Number of worker processes")
    p.add_argument("--job-size", type=int, default=10_000, help="Edges per job/chunk")
    p.add_argument("--in-dir", default=os.path.join(os.environ.get("SPAN_DATA_ROOT", "./orcd"), "scratch"), help="Input directory base")
    p.add_argument("--out-dir", default=os.path.join(os.environ.get("SPAN_DATA_ROOT", "./orcd"), "scratch"), help="Output directory base")
    p.add_argument("--input", default=None, help="Optional explicit input path to .gt.gz")
    p.add_argument("--output", default=None, help="Optional explicit output path")
    p.add_argument("--mp-start-method", default="fork", choices=["fork", "spawn", "forkserver"], help="Multiprocessing start method")
    p.add_argument("--quiet", action="store_true", help="Reduce prints/progress bars")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if args.input is None:
        in_path = in_dir / f"{args.project}/{args.sample_id}_barcodesGraph_{args.suffix}.gt.gz"
    else:
        in_path = Path(args.input)

    if not in_path.exists():
        raise FileNotFoundError(f"Input graph not found: {in_path}")

    if args.output is None:
        out_path = out_dir / f"{args.project}/{args.sample_id}_barcodesGraph_{args.suffix}_indirectPaths.gt.gz"
    else:
        out_path = Path(args.output)

    prop_name3 = "indirectP3"
    prop_name5 = "indirectP5"

    if verbose:
        print("============================================================")
        print("Indirect Paths Pipeline")
        print("------------------------------------------------------------")
        print(f"Input:        {in_path}")
        print(f"Output:       {out_path}")
        print(f"Properties:   ep['{prop_name3}'], ep['{prop_name5}']")
        print(f"Workers:      {args.workers}")
        print(f"Job size:     {args.job_size}")
        print("Graph mode:   undirected (fixed)")
        print("Path lengths: exact 3 and exact 5")
        print("Path type:    simple paths, excluding the direct edge")
        print("============================================================")

    # Load graph
    t0 = perf_counter()
    if verbose:
        print("[0/7] Loading graph...")
    g = gt.load_graph(str(in_path))
    if verbose:
        print(f"      Loaded in {perf_counter() - t0:.2f}s")

    # Compute
    p3_prop, p5_prop, p3_arr, p5_arr, _edges_uv = compute_indirect_paths_parallel(
        g=g,
        workers=int(args.workers),
        job_size=int(args.job_size),
        verbose=verbose,
        mp_start_method=args.mp_start_method,
    )

    # Stats
    n_total3, n_nonzero3, total_paths3, max_count3 = count_stats_from_array(p3_arr)
    n_total5, n_nonzero5, total_paths5, max_count5 = count_stats_from_array(p5_arr)

    if verbose:
        print("------------------------------------------------------------")
        print("Indirect path statistics (computed over non-self-loop edges):")
        print(f"  edges total:        {n_total3:,}")
        print(f"  indirectP3 > 0:     {n_nonzero3:,} ({(n_nonzero3 / n_total3 if n_total3 else 0):.2%})")
        print(f"  indirectP3 total:   {total_paths3:,}")
        print(f"  indirectP3 max:     {max_count3:,}")
        print(f"  indirectP5 > 0:     {n_nonzero5:,} ({(n_nonzero5 / n_total5 if n_total5 else 0):.2%})")
        print(f"  indirectP5 total:   {total_paths5:,}")
        print(f"  indirectP5 max:     {max_count5:,}")
        print("------------------------------------------------------------")

    # Attach and save
    if verbose:
        print("[7/7] Saving graph with indirect path properties...")
    g.ep[prop_name3] = p3_prop
    g.ep[prop_name5] = p5_prop
    g.save(str(out_path))

    if verbose:
        print("Done.")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()