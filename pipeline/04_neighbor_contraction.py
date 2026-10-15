"""
Pipeline D (MP): k-hop neighborhood barcode contraction on the LCC graph (graph_tool),
using multiprocessing workers that emit candidate union pairs, and a parent process that
applies DSU + rebuild.

Algorithm (iterative rounds):
  For each vertex v:
    - BFS out to distance <= K
    - Consider only nodes u at EVEN distance (2,4,...) from v
    - Require barcode first char matches (L/R)
    - If Levenshtein(bc[v], bc[u]) <= dist_thresh: union(v,u)
  After pass, rebuild contracted graph; repeat until 0 unions.

Notes:
  - We build a CSR adjacency (neighbors + offsets) once per round in the parent.
  - On Linux with fork (typical HPC), workers share these arrays copy-on-write (fast).
  - Workers return candidate (vid, uid) pairs; parent unions them (deterministic rep).
"""

import os
import sys
import time
import argparse
import numpy as np
import graph_tool as gt
import multiprocessing as mp
from collections import deque
from tqdm import tqdm
from Levenshtein import distance as lev_distance


# Root data directory (contains scratch/<project>/). Override with SPAN_DATA_ROOT.
DATA_ROOT = os.environ.get("SPAN_DATA_ROOT", "./orcd")


def _scratch_prefix_for(project_name: str) -> str:
    return os.path.join(DATA_ROOT, "scratch", project_name) + "/"


class DSU:
    """Union-find with deterministic representative (smallest vertex id in component)."""
    __slots__ = ("parent", "rep")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rep = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False

        if ra < rb:
            self.parent[rb] = ra
            if self.rep[rb] < self.rep[ra]:
                self.rep[ra] = self.rep[rb]
        else:
            self.parent[ra] = rb
            if self.rep[ra] < self.rep[rb]:
                self.rep[rb] = self.rep[ra]
        return True


def rebuild_contracted_graph(g: gt.Graph, dsu: DSU, has_bip: bool) -> gt.Graph:
    """Rebuild graph by DSU components; aggregate edge weights; preserve representative properties."""
    n = g.num_vertices()

    roots = np.fromiter((dsu.find(i) for i in range(n)), dtype=np.int64, count=n)
    unique_roots, inv = np.unique(roots, return_inverse=True)
    ncomp = unique_roots.shape[0]

    ng = gt.Graph(directed=False)
    ng.add_vertex(ncomp)

    bc_old = g.vp.barcode
    ng_bc = ng.new_vertex_property("string")
    ng.vertex_properties["barcode"] = ng_bc

    if has_bip:
        bip_old = g.vp.bipartite
        ng_bip = ng.new_vertex_property("int")
        ng.vertex_properties["bipartite"] = ng_bip
    else:
        bip_old = None
        ng_bip = None

    w_old = g.ep.weight if "weight" in g.ep else None
    ng_w = ng.new_edge_property("double")
    ng.edge_properties["weight"] = ng_w

    # component representative props
    for new_id, old_root in enumerate(unique_roots):
        rep_vid = int(dsu.rep[old_root])
        v_old = g.vertex(rep_vid)
        ng_bc[ng.vertex(new_id)] = bc_old[v_old]
        if has_bip:
            ng_bip[ng.vertex(new_id)] = int(bip_old[v_old])

    # edges (aggregate weights between components)
    if w_old is None:
        edges = g.get_edges()  # [src, dst]
        wt = np.ones(edges.shape[0], dtype=np.float64)
        src = edges[:, 0].astype(np.int64, copy=False)
        dst = edges[:, 1].astype(np.int64, copy=False)
    else:
        edges_w = g.get_edges([w_old])  # [src, dst, w]
        src = edges_w[:, 0].astype(np.int64, copy=False)
        dst = edges_w[:, 1].astype(np.int64, copy=False)
        wt = edges_w[:, 2].astype(np.float64, copy=False)

    cs = inv[src]
    ct = inv[dst]

    edge_acc = {}
    for a, b, ww in zip(cs, ct, wt):
        if a == b:
            continue
        if a < b:
            key = (int(a), int(b))
        else:
            key = (int(b), int(a))
        edge_acc[key] = edge_acc.get(key, 0.0) + float(ww)

    for (a, b), ww in edge_acc.items():
        e = ng.add_edge(ng.vertex(a), ng.vertex(b))
        ng_w[e] = ww

    return ng


# ---- CSR adjacency build (parent) ----

def build_undirected_csr(g: gt.Graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Build CSR adjacency for an undirected graph_tool graph.
    Returns:
      offsets: shape (n+1,), int64
      nbrs:    shape (2m,), int64
    """
    n = g.num_vertices()
    edges = g.get_edges()  # [src, dst]
    src = edges[:, 0].astype(np.int64, copy=False)
    dst = edges[:, 1].astype(np.int64, copy=False)

    # undirected => add both directions
    all_u = np.concatenate([src, dst])
    all_v = np.concatenate([dst, src])

    deg = np.bincount(all_u, minlength=n).astype(np.int64, copy=False)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(deg, out=offsets[1:])

    nbrs = np.empty(all_v.shape[0], dtype=np.int64)

    # fill using a cursor array
    cursor = offsets[:-1].copy()
    for u, v in zip(all_u, all_v):
        i = cursor[u]
        nbrs[i] = v
        cursor[u] = i + 1

    return offsets, nbrs


# ---- multiprocessing worker globals ----
_G_OFFSETS = None
_G_NBRS = None
_G_BARCODES = None
_G_LR = None
_G_K = None
_G_DIST = None
_G_ONLY_EVEN = None


def _init_worker(offsets, nbrs, barcodes, lr, k_hops, dist_thresh, only_even):
    global _G_OFFSETS, _G_NBRS, _G_BARCODES, _G_LR, _G_K, _G_DIST, _G_ONLY_EVEN
    _G_OFFSETS = offsets
    _G_NBRS = nbrs
    _G_BARCODES = barcodes
    _G_LR = lr
    _G_K = k_hops
    _G_DIST = dist_thresh
    _G_ONLY_EVEN = only_even


def _bfs_emit_pairs_for_vid(start: int) -> list[tuple[int, int]]:
    """
    BFS up to K from start; emit (start, uid) where:
      - uid != start
      - (dist is even) if only_even
      - lr matches
      - Levenshtein <= dist_thresh
      - and we only emit if start < uid (to reduce duplicates)
    """
    offsets = _G_OFFSETS
    nbrs = _G_NBRS
    barcodes = _G_BARCODES
    lr = _G_LR
    K = _G_K
    DIST = _G_DIST
    only_even = _G_ONLY_EVEN

    v_bc = barcodes[start]
    v_lr = lr[start]

    visited = {start}
    q = deque([(start, 0)])
    out_pairs: list[tuple[int, int]] = []

    while q:
        vid, dist = q.popleft()
        if dist == K:
            continue

        nb_start = offsets[vid]
        nb_end = offsets[vid + 1]
        for uid in nbrs[nb_start:nb_end]:
            uid = int(uid)
            if uid in visited:
                continue
            visited.add(uid)
            nd = dist + 1

            # Evaluate candidate at discovery time
            if uid != start:
                if (not only_even) or (nd % 2 == 0):
                    if lr[uid] == v_lr:
                        # Only emit one direction to cut duplicates
                        if start < uid:
                            u_bc = barcodes[uid]
                            if lev_distance(v_bc, u_bc, score_cutoff=DIST) <= DIST:
                                out_pairs.append((start, uid))

            if nd < K:
                q.append((uid, nd))

    return out_pairs


def _work_chunk(args) -> tuple[int, list[tuple[int, int]]]:
    """
    Process a chunk [lo, hi) of vertex ids.
    Returns (num_pairs, pairs_list).
    """
    lo, hi = args
    pairs: list[tuple[int, int]] = []
    for vid in range(lo, hi):
        pairs.extend(_bfs_emit_pairs_for_vid(vid))
    return (len(pairs), pairs)


def chunk_ranges(n: int, chunk_size: int):
    for lo in range(0, n, chunk_size):
        yield (lo, min(n, lo + chunk_size))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_name")
    ap.add_argument("sample_id")
    ap.add_argument("--suffix", type=str, default="LCC")
    ap.add_argument("--k", type=int, default=4, help="BFS hop limit")
    ap.add_argument("--dist", type=int, default=4, help="Levenshtein threshold")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000, help="vertex ids per task")
    ap.add_argument("--all-distances", action="store_true",
                    help="if set, consider all distances (not just even distances)")
    args = ap.parse_args()

    scratch_prefix = _scratch_prefix_for(args.project_name)
    os.makedirs(scratch_prefix, exist_ok=True)

    in_filename = os.path.join(scratch_prefix, f"{args.sample_id}_barcodesGraph_{args.suffix}.gt.gz")
    out_filename = os.path.join(scratch_prefix, f"{args.sample_id}_barcodesGraph_{args.suffix}_Lev{args.dist}_contractedK{args.k}.gt.gz")

    print("graph_tool Version:", gt.__version__, flush=True)
    print("OpenMP Enabled:", gt.openmp_enabled(), flush=True)
    if gt.openmp_enabled():
        print("Number of threads: ", gt.openmp_get_num_threads(), flush=True)
    print("Project Name:", args.project_name, flush=True)
    print("Sample ID:", args.sample_id, flush=True)
    print("Workers:", args.workers, "Chunk:", args.chunk, flush=True)

    print("Loading:", in_filename, flush=True)
    t0 = time.perf_counter()
    g = gt.load_graph(in_filename)
    print(f"Loaded in {(time.perf_counter() - t0)/60.0:.2f} minutes.", flush=True)

    if "barcode" not in g.vp:
        raise RuntimeError("Graph is missing vertex property 'barcode'")
    has_bip = "bipartite" in g.vp

    print("Initial graph:", flush=True)
    print(f"  vertices: {g.num_vertices():,}", flush=True)
    print(f"  edges:    {g.num_edges():,}", flush=True)
    if "weight" in g.ep:
        print(f"  total edge weight: {int(np.sum(g.ep.weight.a)):,}", flush=True)

    # Prefer fork if available (critical for not duplicating giant arrays)
    if "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
    else:
        ctx = mp.get_context()
        print("WARNING: 'fork' start method not available; memory use may be high.", flush=True)

    only_even = (not args.all_distances)

    rnd = 1
    total_unions = 0
    t_total0 = time.perf_counter()

    while True:
        n = g.num_vertices()

        # Materialize barcodes into a plain Python list once per round
        # (With fork, workers see this list without copying.)
        bc_pm = g.vp.barcode
        barcodes = [bc_pm[g.vertex(i)] for i in range(n)]
        # first-char filter as bytes for speed
        lr = np.fromiter((ord(barcodes[i][0]) if barcodes[i] else 0 for i in range(n)),
                         dtype=np.uint8, count=n)

        print(f"Building CSR adjacency (round {rnd})...", flush=True)
        t_csr0 = time.perf_counter()
        offsets, nbrs = build_undirected_csr(g)
        print(f"CSR built in {(time.perf_counter() - t_csr0)/60.0:.2f} min", flush=True)

        dsu = DSU(n)
        unions = 0
        pairs_seen = 0

        print(f"{args.k}-hop merge round {rnd} (Levenshtein<= {args.dist}) "
              f"{'[even distances only]' if only_even else '[all distances]'}", flush=True)

        t_round0 = time.perf_counter()

        # schedule chunks
        ranges = list(chunk_ranges(n, args.chunk))

        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(offsets, nbrs, barcodes, lr, args.k, args.dist, only_even),
        ) as pool:
            # Use imap_unordered so parent can union incrementally
            for num_pairs, pairs in tqdm(
                pool.imap_unordered(_work_chunk, ranges, chunksize=1),
                total=len(ranges),
                mininterval=0.5,
                smoothing=0.05,
            ):
                pairs_seen += num_pairs
                # Apply unions in parent
                for a, b in pairs:
                    if dsu.union(a, b):
                        unions += 1

        round_sec = time.perf_counter() - t_round0
        total_unions += unions

        print(f"Round {rnd}: candidate pairs={pairs_seen:,} unions={unions:,} "
              f"(time {round_sec/60.0:.2f} min)", flush=True)

        if unions <= 5:
            print("Unions <= 5; stopping.", flush=True)
            break

        print("Rebuilding graph from DSU components...", flush=True)
        t_rebuild0 = time.perf_counter()
        g = rebuild_contracted_graph(g, dsu, has_bip=has_bip)
        print(f"Rebuilt in {(time.perf_counter() - t_rebuild0)/60.0:.2f} min", flush=True)

        print("Graph after round:", flush=True)
        print(f"  vertices: {g.num_vertices():,}", flush=True)
        print(f"  edges:    {g.num_edges():,}", flush=True)
        if "weight" in g.ep:
            print(f"  total edge weight: {int(round(float(np.sum(g.ep.weight.a)))):,}", flush=True)

        rnd += 1

    total_sec = time.perf_counter() - t_total0
    print(f"Total merge time: {total_sec/60.0:.2f} minutes", flush=True)
    print(f"Total unions across rounds: {total_unions:,}", flush=True)

    print("Final graph:", flush=True)
    print(f"  vertices: {g.num_vertices():,}", flush=True)
    print(f"  edges:    {g.num_edges():,}", flush=True)
    if "weight" in g.ep:
        print(f"  total edge weight: {int(round(float(np.sum(g.ep.weight.a)))):,}", flush=True)

    print("Saving:", out_filename, flush=True)
    t_save0 = time.perf_counter()
    g.save(out_filename)
    print(f"Saved in {(time.perf_counter() - t_save0)/60.0:.2f} minutes.", flush=True)
    print("----DONE----", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
