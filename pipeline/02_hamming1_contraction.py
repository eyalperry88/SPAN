"""
Hamming-1 contraction for barcode graphs.

- Input:  graph-tool .gt.gz with vertex property "barcode"
- L barcodes: start with 'L', length 27, compare within L only
- R barcodes: start with 'R', length 24, compare within R only
- Distance: Hamming = 1 (single base substitution)
- Merge rule: DSU over vertices that have at least one Hamming-1 match
- Representative barcode per merged component: vertex with largest
  weighted neighbor sum (weighted degree). Uses edge property "weight" by default.

Output:
  {sample_id}_barcodesGraph_{suffix}_Hamming1.gt.gz in the scratch dir.

This script uses a partitioned masked-key approach:
  For each barcode and each position i, compute a "deletion key" where base i is removed
  (in a 2-bit encoding). Any two barcodes with the same deletion key at position i differ
  only at i => Hamming=1 or 0 (but exact duplicates are fine to union as well).
  We write records (pos, key, idx) into P partition files by key hash. Then each partition
  is processed independently (in parallel), sorting by (pos,key) to emit union pairs.
"""

import argparse
import os
import sys
import struct
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
        # iterative path compression
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
# Barcode normalization + encoding
# ----------------------------
DNA2 = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def _norm_barcode(x) -> str:
    """Normalize graph-tool vp 'barcode' entries into plain str."""
    if isinstance(x, (bytes, bytearray)):
        s = x.decode("utf-8", errors="replace")
    else:
        s = str(x)
    s = s.strip()
    # handle stringified bytes like "b'ACGT...'"
    if s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    return s

def encode_2bit(barcode: str) -> int:
    """
    Encode DNA string into uint64 using 2 bits per base.
    We place the FIRST base in the HIGHEST bits (left-packed).
    For length L, base i (0-based) lives at bitpos = 2*(L-1-i).
    """
    s = barcode.upper()
    v = 0
    for ch in s:
        try:
            b = DNA2[ch]
        except KeyError:
            raise ValueError(f"Non-ACGT base {ch!r} in barcode {barcode!r}")
        v = (v << 2) | b
    return v

def deletion_key(code_u64: np.ndarray, L: int, pos: int) -> np.ndarray:
    """
    Remove the 2-bit base at position pos (0-based, pos=0 is first char)
    from a left-packed 2-bit code.

    bitpos = 2*(L-1-pos)
    key = lower_bits | (upper_bits shifted down by 2)
    """
    bitpos = 2 * (L - 1 - pos)
    if bitpos == 0:
        lower = np.zeros_like(code_u64)
    else:
        lowmask = (np.uint64(1) << np.uint64(bitpos)) - np.uint64(1)
        lower = code_u64 & lowmask
    upper = code_u64 >> np.uint64(bitpos + 2)
    return lower | (upper << np.uint64(bitpos))


# ----------------------------
# Partition file IO
# ----------------------------
REC_STRUCT = struct.Struct("<QIB")  # key:uint64, idx:uint32, pos:uint8  => 13 bytes
REC_DTYPE = np.dtype([("key", "<u8"), ("idx", "<u4"), ("pos", "u1")], align=False)

@dataclass
class GroupData:
    name: str               # 'L' or 'R'
    L: int                  # barcode length
    vids: np.ndarray        # vertex indices in original graph (int64)
    codes: np.ndarray       # 2-bit codes for those vertices (uint64)
    barcodes: List[str]     # original barcode strings, parallel to vids/codes


def build_group(g, group_char: str, expected_len: int) -> GroupData:
    bc = g.vp["barcode"]

    vids: List[int] = []
    codes: List[int] = []
    barcodes: List[str] = []

    bad_prefix = 0
    bad_len = 0

    # Iterate vertices once with a progress bar.
    for v in tqdm(g.vertices(), total=g.num_vertices(), desc=f"Scanning {group_char} barcodes"):
        s = _norm_barcode(bc[v])
        if not s:
            bad_prefix += 1
            continue
        if s[0] != group_char:
            continue
        if len(s) != expected_len:
            bad_len += 1
            continue
        vids.append(int(v))
        barcodes.append(s)
        codes.append(encode_2bit(s[1:]))  # includes leading 'L'/'R'! (see note below)

    L_dna = expected_len - 1
    if L_dna * 2 > 64:
        raise RuntimeError(f"{group_char} DNA length {L_dna} is too long for uint64 encoding.")

    print(f"[{group_char}] kept: {len(vids):,} vertices | bad_len={bad_len:,} | skipped_prefix/empty≈{bad_prefix:,}")

    return GroupData(
        name=group_char,
        L=L_dna,
        vids=np.asarray(vids, dtype=np.int64),
        codes=np.asarray(codes, dtype=np.uint64),
        barcodes=barcodes,
    )


def write_partition_files(
    grp: GroupData,
    out_dir: str,
    partitions: int,
    chunk_size: int = 1_000_000,
) -> List[str]:
    """
    Write P partition files, each containing records (key, idx, pos) for all positions.
    idx is the index within grp (0..len(grp)-1), NOT the original graph vertex id.
    """
    os.makedirs(out_dir, exist_ok=True)
    P = partitions
    if P & (P - 1) != 0:
        raise ValueError("--partitions must be a power of two (for fast masking).")

    paths = [os.path.join(out_dir, f"{grp.name}_part_{p:04d}.bin") for p in range(P)]
    # Ensure empty
    for pth in paths:
        if os.path.exists(pth):
            os.remove(pth)

    # Open all files once
    fps = [open(pth, "ab", buffering=1024 * 1024) for pth in paths]
    buffers = [bytearray() for _ in range(P)]
    flush_threshold = 8 * 1024 * 1024  # 8MB per partition buffer

    n = len(grp.codes)
    L = grp.L

    # We'll stream the group indices (0..n-1) in chunks and for each chunk
    # emit records for each position.
    outer = tqdm(range(0, n, chunk_size), desc=f"[{grp.name}] Writing masked keys", unit="chunk")
    for start in outer:
        end = min(n, start + chunk_size)
        idx = np.arange(start, end, dtype=np.uint32)
        codes = grp.codes[start:end]

        # For each position, compute deletion keys vectorized, then append to partition buffers
        for pos in range(L):
            key = deletion_key(codes, L, pos)
            part = (key & np.uint64(P - 1)).astype(np.uint16)

            # Sort this chunk by partition id (cheap-ish) to batch writes
            order = np.argsort(part, kind="stable")
            part_s = part[order]
            key_s = key[order]
            idx_s = idx[order]

            # Scan contiguous runs of same partition
            run_start = 0
            while run_start < len(part_s):
                p = int(part_s[run_start])
                run_end = run_start + 1
                while run_end < len(part_s) and part_s[run_end] == part_s[run_start]:
                    run_end += 1

                buf = buffers[p]
                # Pack records for this run
                # (loop is in Python, but per-run, so amortized)
                for j in range(run_start, run_end):
                    buf += REC_STRUCT.pack(int(key_s[j]), int(idx_s[j]), int(pos))

                # Flush if large
                if len(buf) >= flush_threshold:
                    fps[p].write(buf)
                    buffers[p].clear()

                run_start = run_end

    # Flush remaining buffers
    for p in range(P):
        if buffers[p]:
            fps[p].write(buffers[p])
            buffers[p].clear()
        fps[p].close()

    return paths


def _process_partition_file(path: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Worker: read one partition file, sort by (pos,key), and emit union pairs (rep, other).
    Returns:
      (num_matches, pairs)
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0, []

    data = np.fromfile(path, dtype=np.uint8)
    # Each record is 13 bytes
    if len(data) % REC_STRUCT.size != 0:
        raise RuntimeError(f"Corrupt partition file {path}: byte length not divisible by 13.")

    recs = data.view(REC_DTYPE)
    # Sort by pos then key
    order = np.lexsort((recs["key"], recs["pos"]))
    recs = recs[order]

    pairs: List[Tuple[int, int]] = []
    matches = 0

    pos = recs["pos"]
    key = recs["key"]
    idx = recs["idx"]

    i = 0
    n = len(recs)
    while i < n:
        j = i + 1
        while j < n and pos[j] == pos[i] and key[j] == key[i]:
            j += 1
        # run [i, j)
        if j - i >= 2:
            rep = int(idx[i])
            # Union everyone to rep; this is sufficient to contract the whole bucket
            for k in range(i + 1, j):
                pairs.append((rep, int(idx[k])))
            matches += (j - i - 1)
        i = j

    return matches, pairs


def build_dsu_from_partitions(
    partition_paths: List[str],
    n_items: int,
    workers: int,
    desc: str,
) -> Tuple[DSU, int]:
    """
    Process all partition files in parallel, unioning emitted pairs into a DSU
    in the main process.
    """
    import multiprocessing as mp

    dsu = DSU(n_items)
    total_matches = 0

    ctx = mp.get_context("fork")  # good default on linux; change to "spawn" if needed
    with ctx.Pool(processes=workers) as pool:
        it = pool.imap_unordered(_process_partition_file, partition_paths, chunksize=1)
        pbar = tqdm(total=len(partition_paths), desc=desc, unit="part")
        for matches, pairs in it:
            total_matches += matches
            for a, b in pairs:
                dsu.union(a, b)
            pbar.update(1)
        pbar.close()

    return dsu, total_matches


def densify_components(dsu: DSU) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Return:
      root_of[i] for i in [0..n-1]
      root_to_dense: mapping root -> dense_id (0..B-1)
    """
    n = len(dsu.parent)
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        roots[i] = dsu.find(i)

    uniq = {}
    dense = 0
    for r in roots:
        rr = int(r)
        if rr not in uniq:
            uniq[rr] = dense
            dense += 1
    return roots, uniq


def pick_representatives_by_strength(
    g,
    grp: GroupData,
    roots: np.ndarray,
    root_to_dense: Dict[int, int],
    strength_vp,  # vertex property map with weighted degree
) -> Tuple[np.ndarray, List[int]]:
    """
    For each dense component id in this group, pick the original-graph vertex index
    with maximum strength.
    Returns:
      dense_id_per_item: array len n_items
      rep_vid_per_dense: list len B where entry is original-graph vertex id
    """
    n = len(grp.vids)
    B = len(root_to_dense)
    dense_id_per_item = np.empty(n, dtype=np.int64)

    # Track best (strength, -tie_break) per dense component
    best_strength = np.full(B, -1.0, dtype=np.float64)
    best_vid = np.full(B, -1, dtype=np.int64)

    for i in range(n):
        r = int(roots[i])
        d = root_to_dense[r]
        dense_id_per_item[i] = d

        v = g.vertex(int(grp.vids[i]))
        s = float(strength_vp[v])
        if (s > best_strength[d]) or (s == best_strength[d] and int(grp.vids[i]) < int(best_vid[d])):
            best_strength[d] = s
            best_vid[d] = int(grp.vids[i])

    return dense_id_per_item, best_vid.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_name")
    ap.add_argument("sample_id")
    ap.add_argument("--suffix", type=str, default="LCC")
    ap.add_argument("--barcode_vp", type=str, default="barcode", help="Vertex property name for barcodes")
    ap.add_argument("--weight_ep", type=str, default="weight", help="Edge property name for weights (weighted degree + condensed edge weights)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--partitions", type=int, default=256, help="Number of hash partitions (power of two). More partitions -> smaller files.")
    ap.add_argument("--chunk_size", type=int, default=1_000_000, help="Chunk size for masked-key generation")
    ap.add_argument("--keep_tmp", action="store_true", help="Keep temporary partition files")
    args = ap.parse_args()

    scratch_prefix = _scratch_prefix_for(args.project_name)
    os.makedirs(scratch_prefix, exist_ok=True)

    in_filename = os.path.join(
        scratch_prefix,
        f"{args.sample_id}_barcodesGraph_{args.suffix}.gt.gz"
    )
    out_filename = os.path.join(
        scratch_prefix,
        f"{args.sample_id}_barcodesGraph_{args.suffix}_Hamming1.gt.gz"
    )

    print("Input :", in_filename)
    print("Output:", out_filename)

    if not os.path.exists(in_filename):
        raise FileNotFoundError(in_filename)

    g = load_graph(in_filename)

    if args.barcode_vp not in g.vp:
        raise RuntimeError(f"Missing vertex property {args.barcode_vp!r}")
    # Make sure it's accessible by the exact key "barcode" too, since you asked for that convention
    if args.barcode_vp != "barcode":
        g.vp["barcode"] = g.vp[args.barcode_vp]

    # Edge weight property (optional)
    if args.weight_ep in g.ep:
        eweight = g.ep[args.weight_ep]
        strength = g.degree_property_map("total", weight=eweight)
        print(f"Using edge weights from ep[{args.weight_ep!r}] for strength + condensed edge weights.")
    else:
        eweight = None
        strength = g.degree_property_map("total")
        print(f"WARNING: ep[{args.weight_ep!r}] not found; using unweighted degree for strength and unit weights for condensation.")

    print(f"Graph BEFORE: |V|={g.num_vertices():,} |E|={g.num_edges():,}")

    # Build L and R groups according to your constraints:
    grpL = build_group(g, "L", expected_len=26)  # 1 + 25 DNA
    grpR = build_group(g, "R", expected_len=26)  # 1 + 25 DNA

    # Temporary dir for partitions
    tmp_root = os.path.join(scratch_prefix, f"tmp_hamming1_{args.sample_id}_{args.suffix}")
    os.makedirs(tmp_root, exist_ok=True)

    total_matches_all = 0

    # For each group, write partitions, process partitions, build DSU + reps
    group_results = {}  # name -> dict of info

    for grp in (grpL, grpR):
        if len(grp.codes) == 0:
            group_results[grp.name] = {
                "dsu": DSU(0),
                "dense_ids": np.array([], dtype=np.int64),
                "rep_vids": [],
                "n_dense": 0,
                "matches": 0,
            }
            continue

        part_dir = os.path.join(tmp_root, f"{grp.name}_parts")
        if os.path.exists(part_dir):
            shutil.rmtree(part_dir)
        os.makedirs(part_dir, exist_ok=True)

        part_paths = write_partition_files(
            grp=grp,
            out_dir=part_dir,
            partitions=args.partitions,
            chunk_size=args.chunk_size,
        )

        dsu, matches = build_dsu_from_partitions(
            partition_paths=part_paths,
            n_items=len(grp.codes),
            workers=min(args.workers, args.partitions),
            desc=f"[{grp.name}] Processing partitions",
        )
        total_matches_all += matches

        roots, root_to_dense = densify_components(dsu)
        dense_id_per_item, rep_vids = pick_representatives_by_strength(
            g=g,
            grp=grp,
            roots=roots,
            root_to_dense=root_to_dense,
            strength_vp=strength,
        )

        group_results[grp.name] = {
            "dsu": dsu,
            "dense_ids": dense_id_per_item,   # per grp item
            "rep_vids": rep_vids,             # per dense component
            "n_dense": len(rep_vids),
            "matches": matches,
            "vids": grp.vids,                 # original-graph vertex ids per grp item
        }

        print(f"[{grp.name}] matches found (union ops emitted): {matches:,}")
        print(f"[{grp.name}] components after DSU: {len(rep_vids):,} (from {len(grp.vids):,} vertices)")

    print(f"TOTAL matches found (L+R): {total_matches_all:,}")

    # Build a dense block id for EVERY vertex in the original graph
    N = g.num_vertices()
    block = np.full(N, -1, dtype=np.int64)

    next_id = 0
    rep_vid_per_block: List[int] = []

    # Assign L blocks
    if len(group_results["L"].get("rep_vids", [])) > 0:
        nL = group_results["L"]["n_dense"]
        rep_vids_L = group_results["L"]["rep_vids"]
        dense_ids_L = group_results["L"]["dense_ids"]
        vids_L = group_results["L"]["vids"]

        # Set block IDs for L vertices
        for i in range(len(vids_L)):
            block[int(vids_L[i])] = next_id + int(dense_ids_L[i])

        # Record reps
        rep_vid_per_block.extend(rep_vids_L)
        next_id += nL

    # Assign R blocks (offset by next_id)
    if len(group_results["R"].get("rep_vids", [])) > 0:
        nR = group_results["R"]["n_dense"]
        rep_vids_R = group_results["R"]["rep_vids"]
        dense_ids_R = group_results["R"]["dense_ids"]
        vids_R = group_results["R"]["vids"]

        for i in range(len(vids_R)):
            block[int(vids_R[i])] = next_id + int(dense_ids_R[i])

        rep_vid_per_block.extend(rep_vids_R)
        next_id += nR

    # Remaining vertices: identity (no contraction)
    mask = (block == -1)
    rem = int(mask.sum())
    if rem > 0:
        block[mask] = np.arange(next_id, next_id + rem, dtype=np.int64)
        # reps are themselves for these identity blocks
        # We need rep_vid_per_block aligned with dense ids 0..B-1, so append in index order
        # of the newly assigned block ids. Since we assigned blocks in increasing order, we can:
        remaining_vids = np.flatnonzero(mask).astype(np.int64)
        rep_vid_per_block.extend(remaining_vids.tolist())
        next_id += rem

    B = next_id
    if len(rep_vid_per_block) != B:
        raise RuntimeError("Internal error: representative list length mismatch.")

    # Create graph-tool vertex property map for block ids
    # (must be integer-valued)
    prop_block = g.new_vertex_property("int64_t")
    # Fill via numpy for speed
    # NOTE: graph-tool uses vertex indices 0..N-1 for g.vertex_index
    prop_block.a = block

    # Condense graph; also sum weights between blocks if eweight is provided (else unit weights).
    if eweight is None:
        cg, vcount, ecount = condensation_graph(g, prop_block, parallel_edges=False)
    else:
        cg, vcount, ecount = condensation_graph(g, prop_block, eweight=eweight, parallel_edges=False)

    # Attach properties to condensed graph
    # Edge weight: ecount is SUM of eweight across inter-block edges (or counts if eweight absent)
    cg.ep["weight"] = ecount
    cg.vp["vcount"] = vcount

    # Barcode: set to representative barcode
    bc_in = g.vp["barcode"]
    bc_out = cg.new_vertex_property("string")

    # Because we ensured blocks are dense 0..B-1, cg vertex indices correspond to block ids.
    for v in tqdm(cg.vertices(), total=cg.num_vertices(), desc="Assigning representative barcodes"):
        bid = int(v)
        rep_vid = rep_vid_per_block[bid]
        bc_out[v] = _norm_barcode(bc_in[g.vertex(rep_vid)])

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
        # allow piping output to head, etc.
        pass
