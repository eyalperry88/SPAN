#!/usr/bin/env python3
"""
Exact Hamming distance <= 2 contraction for barcode graphs.

- Input graph path is constructed from: project_name, sample_id, suffix
- Vertex property: "barcode" (or --barcode_vp)
- Two disjoint subsets:
    L: barcodes start with 'L', total length 28 => DNA length 27 (after prefix)
    R: barcodes start with 'R', total length 25 => DNA length 24 (after prefix)
  Only compare within each subset.

- Distance: exact Hamming <= 2 on the DNA suffix (A/C/G/T).
- Indexing strategy: 2-deletion keys
    For each barcode and each pair of positions (i<j), compute delete2(code,i,j).
    Any pair with Hamming<=2 collides in at least one such bucket.
  We shard records to P partitions on key hash, then each partition is processed
  independently (parallel) by sorting records by (i,j,key). Within each bucket we
  VERIFY true Hamming <= 2 before union.

- Merge/representative rule when contracting:
    Choose within each DSU component the vertex with maximum "strength"
    (weighted degree: sum of edge weights to neighbors). Ties -> smaller vertex id.

- Output:
    Save contracted graph to:
      {scratch}/{sample_id}_barcodesGraph_{suffix}_Hamming2.gt.gz
    Print match counts and |V|, |E| before/after.

Progress:
    Uses tqdm for scanning vertices, writing keys, and per-partition processing.
"""

import argparse
import os
import sys
import struct
import math
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm

from graph_tool.all import load_graph
from graph_tool.generation import condensation_graph


# Root data directory (contains scratch/<project>/). Override with SPAN_DATA_ROOT.
DATA_ROOT = os.environ.get("SPAN_DATA_ROOT", "./orcd")


def _scratch_prefix_for(project_name: str) -> str:
    return os.path.join(DATA_ROOT, "scratch", project_name) + "/"


# ----------------------------
# DSU
# ----------------------------
class DSU:
    __slots__ = ("parent", "size")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return False
        sa = self.size[pa]
        sb = self.size[pb]
        if sa < sb:
            pa, pb = pb, pa
        self.parent[pb] = pa
        self.size[pa] = sa + sb
        return True


# ----------------------------
# Barcode normalization + 2-bit encoding
# ----------------------------
DNA2 = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def _norm_barcode(x) -> str:
    if isinstance(x, (bytes, bytearray, memoryview)):
        s = bytes(x).decode("utf-8", errors="replace")
    else:
        s = str(x)
    s = s.strip()
    if s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    return s

def encode_dna_2bit(seq: str) -> np.uint64:
    """Encode DNA (A/C/G/T) into uint64, left-packed 2 bits/base."""
    s = seq.strip().upper()
    v = 0
    for ch in s:
        try:
            b = DNA2[ch]
        except KeyError:
            raise ValueError(f"Non-ACGT base {ch!r} in DNA {seq[:80]!r}")
        v = (v << 2) | b
    return np.uint64(v)

def popcount_u64(x: np.uint64) -> int:
    # python int popcount is very fast
    return int(int(x).bit_count())

def hamming_le2(code_a: np.uint64, code_b: np.uint64, L: int) -> bool:
    """
    Hamming distance on 2-bit packed DNA:
      Each base uses 2 bits. Distance is number of positions where 2-bit chunks differ.
    Trick:
      d = a ^ b. For each base, the 2 bits are 00 iff equal, else nonzero.
      Collapse each 2-bit chunk to 1 bit:
        m = d | (d >> 1), then mask the L least-significant bits of the chunk lanes.
      Count bits in m_lane. If <=2 => distance<=2.
    """
    d = np.uint64(code_a ^ code_b)
    m = np.uint64(d | (d >> np.uint64(1)))
    # lane bits are at positions 0,2,4,...,2*(L-1) if we right-pack;
    # but our encoding is left-packed, still contiguous 2L bits.
    # We can just build a mask selecting the L "LSB of each 2-bit pair" within the 2L bits.
    # Example for L=3: mask = 0b010101 = 0x15
    # In general:
    lane_mask = (np.uint64(1) << np.uint64(2*L)) - np.uint64(1)
    m &= lane_mask
    # Extract lane bits: take every other bit (LSB of each pair).
    # We can mask with pattern 0b0101... over 2L bits.
    alt_mask = np.uint64(0)
    # build pattern once per L is better; we’ll precompute per group in worker
    # This function is only used with precomputed mask in worker below.
    raise RuntimeError("Internal: use worker-local hamming function with precomputed mask.")


def build_alt_mask_2L(L: int) -> np.uint64:
    # mask with bits set at positions 0,2,4,...,2L-2 within the 2L-bit window
    m = 0
    for i in range(L):
        m |= (1 << (2*i))
    return np.uint64(m)

def hamming_le2_pre(code_a: np.uint64, code_b: np.uint64, L: int, alt_mask: np.uint64, lane_mask: np.uint64) -> bool:
    d = np.uint64(code_a ^ code_b)
    m = np.uint64(d | (d >> np.uint64(1)))
    m &= lane_mask
    m &= alt_mask
    # number of differing positions = popcount(m)
    return (int(m).bit_count() <= 2)


# ----------------------------
# 2-deletion key
# ----------------------------
def delete2_keys(codes: np.ndarray, L: int, i: int, j: int) -> np.ndarray:
    """
    Compute 2-deletion key by removing positions i and j (0-based, i<j) from left-packed code.
    Implemented by first deleting j then deleting i (or vice versa) using bit slicing.

    For left-packed representation, easiest is do on the 2L-bit window:
    - Consider the bits as 2-bit symbols.
    - Remove symbol at j, then at i (with j>i, indices adjust after first deletion).

    We'll implement by extracting each symbol via shifts in vectorized fashion is too slow.
    Instead, do bit slicing:
      Delete one position:
        bitpos = 2*(L-1-pos)
        lower = code & ((1<<bitpos)-1)
        upper = code >> (bitpos+2)
        del1 = lower | (upper << bitpos)
    Then delete the second from the del1 code with length L-1 and adjusted pos.
    """
    if i >= j:
        raise ValueError("Require i<j")
    # delete position j from length L
    bitpos_j = 2 * (L - 1 - j)
    if bitpos_j == 0:
        lower_j = np.zeros_like(codes)
    else:
        lowmask_j = (np.uint64(1) << np.uint64(bitpos_j)) - np.uint64(1)
        lower_j = codes & lowmask_j
    upper_j = codes >> np.uint64(bitpos_j + 2)
    del1 = lower_j | (upper_j << np.uint64(bitpos_j))  # now length L-1

    # delete position i from length L-1 (positions < j unchanged)
    L1 = L - 1
    bitpos_i = 2 * (L1 - 1 - i)
    if bitpos_i == 0:
        lower_i = np.zeros_like(del1)
    else:
        lowmask_i = (np.uint64(1) << np.uint64(bitpos_i)) - np.uint64(1)
        lower_i = del1 & lowmask_i
    upper_i = del1 >> np.uint64(bitpos_i + 2)
    del2 = lower_i | (upper_i << np.uint64(bitpos_i))  # now length L-2
    return del2


# ----------------------------
# Partitioned record IO
# Record: key(uint64), idx(uint32), i(uint8), j(uint8) => 14 bytes
# ----------------------------
REC_STRUCT = struct.Struct("<QI2B")
REC_DTYPE = np.dtype([("key", "<u8"), ("idx", "<u4"), ("i", "u1"), ("j", "u1")], align=False)


@dataclass
class GroupData:
    name: str           # 'L' or 'R'
    L: int              # DNA length (27 or 24)
    vids: np.ndarray    # original graph vertex ids, len n
    codes: np.ndarray   # uint64 codes, len n
    barcodes: List[str] # full barcode strings with prefix, len n


def build_group(g, group_char: str, total_len: int, barcode_vp: str) -> GroupData:
    vp = g.vp[barcode_vp]
    vids: List[int] = []
    codes: List[np.uint64] = []
    barcodes: List[str] = []

    bad_len = 0
    bad_dna = 0

    for v in tqdm(g.vertices(), total=g.num_vertices(), desc=f"Collecting {group_char}"):
        s = _norm_barcode(vp[v])
        if not s or s[0] != group_char:
            continue
        if len(s) != total_len:
            bad_len += 1
            continue
        dna = s[1:]
        try:
            code = encode_dna_2bit(dna)
        except Exception:
            bad_dna += 1
            continue
        vids.append(int(v))
        codes.append(code)
        barcodes.append(s)

    L = total_len - 1
    if 2 * L > 64:
        raise RuntimeError(f"{group_char}: DNA length {L} too long for uint64.")

    print(f"[{group_char}] kept {len(vids):,} | total_len={total_len} | dna_len={L} | bad_len={bad_len:,} | bad_dna={bad_dna:,}")
    return GroupData(
        name=group_char,
        L=L,
        vids=np.asarray(vids, dtype=np.int64),
        codes=np.asarray(codes, dtype=np.uint64),
        barcodes=barcodes,
    )


def write_partitions_2del(
    grp: GroupData,
    out_dir: str,
    partitions: int,
    chunk_size: int,
) -> List[str]:
    """
    Write partition files containing (key, idx, i, j) for all i<j.
    idx is index in grp arrays.
    Partition chosen by key & (P-1). P must be power of 2.
    """
    os.makedirs(out_dir, exist_ok=True)
    P = partitions
    if P & (P - 1) != 0:
        raise ValueError("--partitions must be a power of two.")

    paths = [os.path.join(out_dir, f"{grp.name}_part_{p:04d}.bin") for p in range(P)]
    for pth in paths:
        if os.path.exists(pth):
            os.remove(pth)

    fps = [open(pth, "ab", buffering=1024*1024) for pth in paths]
    buffers = [bytearray() for _ in range(P)]
    flush_threshold = 8 * 1024 * 1024  # 8MB per partition buffer

    n = len(grp.codes)
    L = grp.L

    outer = tqdm(range(0, n, chunk_size), desc=f"[{grp.name}] Writing 2-del keys", unit="chunk")
    for start in outer:
        end = min(n, start + chunk_size)
        idx = np.arange(start, end, dtype=np.uint32)
        codes = grp.codes[start:end]

        # For each (i,j), compute keys for this chunk vectorized, then shard by partition
        for i in range(L - 1):
            for j in range(i + 1, L):
                key = delete2_keys(codes, L, i, j)
                part = (key & np.uint64(P - 1)).astype(np.uint16)

                # sort by partition to batch
                order = np.argsort(part, kind="stable")
                part_s = part[order]
                key_s = key[order]
                idx_s = idx[order]

                run_start = 0
                while run_start < len(part_s):
                    p = int(part_s[run_start])
                    run_end = run_start + 1
                    while run_end < len(part_s) and part_s[run_end] == part_s[run_start]:
                        run_end += 1

                    buf = buffers[p]
                    for t in range(run_start, run_end):
                        buf += REC_STRUCT.pack(int(key_s[t]), int(idx_s[t]), int(i), int(j))

                    if len(buf) >= flush_threshold:
                        fps[p].write(buf)
                        buffers[p].clear()

                    run_start = run_end

    for p in range(P):
        if buffers[p]:
            fps[p].write(buffers[p])
            buffers[p].clear()
        fps[p].close()

    return paths


def process_partition_file_2del(args) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Worker: read one partition file, sort by (i,j,key), and emit VERIFIED union pairs.
    Returns: (matches, pairs)
    """
    path, codes_path, n_codes, L = args

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0, []

    # memmap codes for verification
    codes = np.memmap(codes_path, dtype=np.uint64, mode="r", shape=(n_codes,))

    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % REC_STRUCT.size != 0:
        raise RuntimeError(f"Corrupt partition file {path}: size not divisible by {REC_STRUCT.size}.")
    recs = raw.view(REC_DTYPE)

    # sort by i, j, key
    order = np.lexsort((recs["key"], recs["j"], recs["i"]))
    recs = recs[order]

    lane_mask = (np.uint64(1) << np.uint64(2*L)) - np.uint64(1)
    alt_mask = build_alt_mask_2L(L)

    matches = 0
    pairs: List[Tuple[int, int]] = []

    i_arr = recs["i"]
    j_arr = recs["j"]
    k_arr = recs["key"]
    idx_arr = recs["idx"]

    m = len(recs)
    s = 0
    while s < m:
        e = s + 1
        while e < m and i_arr[e] == i_arr[s] and j_arr[e] == j_arr[s] and k_arr[e] == k_arr[s]:
            e += 1

        # bucket is [s,e)
        if e - s >= 2:
            # verify Hamming<=2 within the bucket
            # Use first as rep candidate; verify each against rep first, but also in case
            # rep isn't within <=2 to others (shouldn't happen for true candidates, but collisions can exist),
            # we do a light spanning approach: connect to first element that matches.
            bucket = idx_arr[s:e].astype(np.int64, copy=False)
            # maintain a list of "seeds" within bucket to connect
            seeds: List[int] = [int(bucket[0])]

            for t in range(1, len(bucket)):
                u = int(bucket[t])
                cu = codes[u]
                linked = False
                for seed in seeds:
                    if hamming_le2_pre(cu, codes[seed], L, alt_mask, lane_mask):
                        pairs.append((seed, u))
                        matches += 1
                        linked = True
                        break
                if not linked:
                    # New seed (rare). This prevents wrong unions if collisions happen.
                    seeds.append(u)

        s = e

    return matches, pairs


def build_dsu_from_partitions_2del(
    partition_paths: List[str],
    codes_mmap_path: str,
    n_codes: int,
    L: int,
    workers: int,
    desc: str,
) -> Tuple[DSU, int]:
    import multiprocessing as mp
    dsu = DSU(n_codes)
    total_matches = 0

    ctx = mp.get_context("fork")
    work = [(p, codes_mmap_path, n_codes, L) for p in partition_paths]

    with ctx.Pool(processes=workers) as pool:
        it = pool.imap_unordered(process_partition_file_2del, work, chunksize=1)
        pbar = tqdm(total=len(work), desc=desc, unit="part")
        for matches, pairs in it:
            total_matches += matches
            for a, b in pairs:
                dsu.union(a, b)
            pbar.update(1)
        pbar.close()

    return dsu, total_matches


def densify_components(dsu: DSU) -> Tuple[np.ndarray, Dict[int, int]]:
    n = len(dsu.parent)
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        roots[i] = dsu.find(i)
    root_to_dense: Dict[int, int] = {}
    dense = 0
    for r in roots:
        rr = int(r)
        if rr not in root_to_dense:
            root_to_dense[rr] = dense
            dense += 1
    return roots, root_to_dense


def pick_representatives_by_strength(
    g,
    grp: GroupData,
    roots: np.ndarray,
    root_to_dense: Dict[int, int],
    strength_vp,
) -> Tuple[np.ndarray, List[int]]:
    n = len(grp.vids)
    B = len(root_to_dense)
    dense_id = np.empty(n, dtype=np.int64)

    best_strength = np.full(B, -1.0, dtype=np.float64)
    best_vid = np.full(B, -1, dtype=np.int64)

    for i in range(n):
        d = root_to_dense[int(roots[i])]
        dense_id[i] = d
        v = g.vertex(int(grp.vids[i]))
        s = float(strength_vp[v])
        if (s > best_strength[d]) or (s == best_strength[d] and int(grp.vids[i]) < int(best_vid[d])):
            best_strength[d] = s
            best_vid[d] = int(grp.vids[i])

    return dense_id, best_vid.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_name")
    ap.add_argument("sample_id")
    ap.add_argument("--suffix", type=str, default="LCC")
    ap.add_argument("--barcode_vp", type=str, default="barcode")
    ap.add_argument("--weight_ep", type=str, default="weight")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--partitions", type=int, default=512, help="Power of two. More partitions -> smaller files.")
    ap.add_argument("--chunk_size", type=int, default=250_000, help="Chunk size for 2-del key generation (tune for RAM/IO).")
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()

    scratch_prefix = _scratch_prefix_for(args.project_name)
    os.makedirs(scratch_prefix, exist_ok=True)

    in_filename = os.path.join(scratch_prefix, f"{args.sample_id}_barcodesGraph_{args.suffix}.gt.gz")
    out_filename = os.path.join(scratch_prefix, f"{args.sample_id}_barcodesGraph_{args.suffix}_Hamming2.gt.gz")

    print("Input :", in_filename)
    print("Output:", out_filename)
    if not os.path.exists(in_filename):
        raise FileNotFoundError(in_filename)

    g = load_graph(in_filename)

    if args.barcode_vp not in g.vp:
        raise RuntimeError(f"Missing vertex property {args.barcode_vp!r}; available: {list(g.vp.keys())}")

    # Ensure canonical name "barcode" exists as you requested
    if args.barcode_vp != "barcode":
        g.vp["barcode"] = g.vp[args.barcode_vp]

    # Strength = weighted degree if weight ep exists, else plain degree
    if args.weight_ep in g.ep:
        eweight = g.ep[args.weight_ep]
        strength = g.degree_property_map("total", weight=eweight)
        print(f"Using edge weights ep[{args.weight_ep!r}] for strength + condensed edge weights.")
    else:
        eweight = None
        strength = g.degree_property_map("total")
        print(f"WARNING: ep[{args.weight_ep!r}] not found. Using unweighted degree, and condensation counts edges.")

    print(f"Graph BEFORE: |V|={g.num_vertices():,} |E|={g.num_edges():,}")

    # Your corrected lengths:
    #   L: total 26 (L + 25 DNA)
    #   R: total 25 (R + 25 DNA)
    grpL = build_group(g, "L", total_len=26, barcode_vp="barcode")
    grpR = build_group(g, "R", total_len=26, barcode_vp="barcode")

    tmp_root = os.path.join(scratch_prefix, f"tmp_hamming2_{args.sample_id}_{args.suffix}")
    os.makedirs(tmp_root, exist_ok=True)

    total_matches = 0
    group_infos = {}

    for grp in (grpL, grpR):
        if len(grp.codes) == 0:
            group_infos[grp.name] = dict(
                dense_ids=np.array([], dtype=np.int64),
                rep_vids=[],
                vids=np.array([], dtype=np.int64),
                n_dense=0,
                matches=0,
            )
            continue

        part_dir = os.path.join(tmp_root, f"{grp.name}_parts")
        if os.path.exists(part_dir):
            shutil.rmtree(part_dir)
        os.makedirs(part_dir, exist_ok=True)

        # Save codes to a memmap file for worker verification
        codes_path = os.path.join(tmp_root, f"{grp.name}_codes.mmap")
        mm = np.memmap(codes_path, dtype=np.uint64, mode="w+", shape=(len(grp.codes),))
        mm[:] = grp.codes
        mm.flush()
        del mm

        part_paths = write_partitions_2del(
            grp=grp,
            out_dir=part_dir,
            partitions=args.partitions,
            chunk_size=args.chunk_size,
        )

        dsu, matches_grp = build_dsu_from_partitions_2del(
            partition_paths=part_paths,
            codes_mmap_path=codes_path,
            n_codes=len(grp.codes),
            L=grp.L,
            workers=min(args.workers, args.partitions),
            desc=f"[{grp.name}] Processing partitions",
        )

        total_matches += matches_grp

        roots, root_to_dense = densify_components(dsu)
        dense_id_per_item, rep_vids = pick_representatives_by_strength(
            g=g,
            grp=grp,
            roots=roots,
            root_to_dense=root_to_dense,
            strength_vp=strength,
        )

        group_infos[grp.name] = dict(
            dense_ids=dense_id_per_item,
            rep_vids=rep_vids,
            vids=grp.vids,
            n_dense=len(rep_vids),
            matches=matches_grp,
        )

        print(f"[{grp.name}] VERIFIED matches (Hamming<=2 unions emitted): {matches_grp:,}")
        print(f"[{grp.name}] components: {len(rep_vids):,} from {len(grp.vids):,} vertices")

    print(f"TOTAL VERIFIED matches (L+R): {total_matches:,}")

    # Build dense block id for all vertices: merged blocks for L and R, identity for the rest
    N = g.num_vertices()
    block = np.full(N, -1, dtype=np.int64)

    next_id = 0
    rep_vid_per_block: List[int] = []

    # L blocks
    if group_infos["L"]["n_dense"] > 0:
        dense_ids = group_infos["L"]["dense_ids"]
        vids = group_infos["L"]["vids"]
        rep_vids = group_infos["L"]["rep_vids"]
        for i in range(len(vids)):
            block[int(vids[i])] = next_id + int(dense_ids[i])
        rep_vid_per_block.extend(rep_vids)
        next_id += len(rep_vids)

    # R blocks
    if group_infos["R"]["n_dense"] > 0:
        dense_ids = group_infos["R"]["dense_ids"]
        vids = group_infos["R"]["vids"]
        rep_vids = group_infos["R"]["rep_vids"]
        for i in range(len(vids)):
            block[int(vids[i])] = next_id + int(dense_ids[i])
        rep_vid_per_block.extend(rep_vids)
        next_id += len(rep_vids)

    # Remaining vertices get unique identity blocks
    mask = (block == -1)
    rem = int(mask.sum())
    if rem > 0:
        block[mask] = np.arange(next_id, next_id + rem, dtype=np.int64)
        remaining_vids = np.flatnonzero(mask).astype(np.int64)
        rep_vid_per_block.extend(remaining_vids.tolist())
        next_id += rem

    B = next_id
    if len(rep_vid_per_block) != B:
        raise RuntimeError("Internal error: rep_vid_per_block length mismatch.")

    # Create graph-tool block property
    prop_block = g.new_vertex_property("int64_t")
    prop_block.a = block

    # Condense graph; sum edge weights if provided
    if eweight is None:
        cg, vcount, ecount = condensation_graph(g, prop_block, parallel_edges=False)
    else:
        cg, vcount, ecount = condensation_graph(g, prop_block, eweight=eweight, parallel_edges=False)

    cg.vp["vcount"] = vcount
    cg.ep["weight"] = ecount

    # Assign representative barcode to each condensed vertex
    bc_in = g.vp["barcode"]
    bc_out = cg.new_vertex_property("string")
    for v in tqdm(cg.vertices(), total=cg.num_vertices(), desc="Assigning rep barcodes"):
        bid = int(v)
        rep_vid = rep_vid_per_block[bid]
        bc_out[v] = _norm_barcode(bc_in[g.vertex(int(rep_vid))])
    cg.vp["barcode"] = bc_out

    print(f"Graph AFTER : |V|={cg.num_vertices():,} |E|={cg.num_edges():,}")
    print("Saving:", out_filename)
    cg.save(out_filename)

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
