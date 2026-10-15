#!/usr/bin/env python3

# Standard libraries
import os
import sys

# BioPython
from Bio import SeqIO

# Visualizations / progress
from tqdm import tqdm

# String distance
from Levenshtein import distance

# Graph-tool
from graph_tool.all import Graph

print("Loaded Libraries", flush=True)

# Parameters
if len(sys.argv) != 5:
    print("Usage:", sys.argv[0], "PROJECT_NAME SAMPLE_ID FLOWCELL_LANES SAMPLE_IDX")
    print(sys.argv)
    print("Example FLOWCELL_LANES: 7393S:2;3,7389S:3;4")
    sys.exit(1)

PROJECT_NAME = sys.argv[1]
SAMPLE_ID = sys.argv[2]
FLOWCELL_LANES_ARG = sys.argv[3]
SAMPLE_IDX = int(sys.argv[4])

# Suffix-overlap filter
OVERLAP_K = 18
OVERLAP_LEV_THRESHOLD = 3


def suffixes_overlap(record1, record2) -> bool:
    """
    Check whether the K-base suffix of record1 is close to the reverse
    complement of the K-base suffix of record2.

    This follows the request literally and skips when Levenshtein distance
    is less than 2, meaning distance 0 or 1.

    Reads shorter than K are not filtered.
    """
    if len(record1.seq) < OVERLAP_K or len(record2.seq) < OVERLAP_K:
        return False

    suffix1 = str(record1.seq[-OVERLAP_K:])
    suffix2_rc = str(record2.seq[-OVERLAP_K:].reverse_complement())

    overlap_distance = distance(
        suffix1,
        suffix2_rc,
        score_cutoff=OVERLAP_LEV_THRESHOLD,
    )

    return overlap_distance <= OVERLAP_LEV_THRESHOLD


def parse_flowcell_lanes(spec: str):
    """
    Parse strings like:
      7393S:2;3,7389S:3;4
    into:
      {
        "7393S": ["2", "3"],
        "7389S": ["3", "4"]
      }

    Also supports:
      7393S:0
    for the no-lane case in your original code.
    """
    out = {}
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue

        if ':' not in item:
            raise ValueError(f"Invalid flowcell spec '{item}'. Expected FLOWCELL:lane;lane")

        flowcell, lane_str = item.split(':', 1)
        flowcell = flowcell.strip()
        lane_str = lane_str.strip()

        if not flowcell:
            raise ValueError(f"Empty flowcell in spec '{item}'")

        lanes = [x.strip() for x in lane_str.split(';') if x.strip()]
        if not lanes:
            raise ValueError(f"No lanes provided for flowcell '{flowcell}'")

        out[flowcell] = lanes

    return out


FLOWCELL_LANES = parse_flowcell_lanes(FLOWCELL_LANES_ARG)
FLOWCELLS = list(FLOWCELL_LANES.keys())

print("Project Name: ", PROJECT_NAME)
print("Sample ID: ", SAMPLE_ID)
print("Flowcell/Lanes: ", FLOWCELL_LANES)
print("Sample IDX: ", SAMPLE_IDX)
print("Overlap suffix K: ", OVERLAP_K)
print("Overlap skip condition: Levenshtein distance < ", OVERLAP_LEV_THRESHOLD, flush=True)

# Root data directory (contains pool/<project>/ and scratch/<project>/). Override with SPAN_DATA_ROOT.
DATA_ROOT = os.environ.get("SPAN_DATA_ROOT", "./orcd")
pool_prefix = os.path.join(DATA_ROOT, "pool", PROJECT_NAME) + "/"
scratch_prefix = os.path.join(DATA_ROOT, "scratch", PROJECT_NAME) + "/"

if not os.path.exists(pool_prefix):
    print("Project path ", pool_prefix, "does not exist!")
    sys.exit(1)

if not os.path.exists(scratch_prefix):
    os.makedirs(scratch_prefix)

fastqs = {}
fastqs[SAMPLE_ID] = {}

for flowcell in FLOWCELLS:
    fastqs[SAMPLE_ID][flowcell] = {}
    for lane in FLOWCELL_LANES[flowcell]:
        if len(FLOWCELL_LANES[flowcell]) == 1 and int(lane) == 0:
            fastqs[SAMPLE_ID][flowcell][lane] = {
                "fastq1_filename": f'{pool_prefix}D26-{SAMPLE_IDX}-{flowcell}/{PROJECT_NAME}_D26-{SAMPLE_IDX}_1_sequence.fastq',
                "fastq2_filename": f'{pool_prefix}D26-{SAMPLE_IDX}-{flowcell}/{PROJECT_NAME}_D26-{SAMPLE_IDX}_2_sequence.fastq'
            }
        else:
            fastqs[SAMPLE_ID][flowcell][lane] = {
                "fastq1_filename": f'{pool_prefix}D26-{SAMPLE_IDX}-{lane}-{flowcell}/{PROJECT_NAME}_D26-{SAMPLE_IDX}-{lane}_1_sequence.fastq',
                "fastq2_filename": f'{pool_prefix}D26-{SAMPLE_IDX}-{lane}-{flowcell}/{PROJECT_NAME}_D26-{SAMPLE_IDX}-{lane}_2_sequence.fastq'
            }

        if not os.path.exists(fastqs[SAMPLE_ID][flowcell][lane]["fastq1_filename"]):
            print("File ", fastqs[SAMPLE_ID][flowcell][lane]["fastq1_filename"], " does not exist!")
            sys.exit(1)
        if not os.path.exists(fastqs[SAMPLE_ID][flowcell][lane]["fastq2_filename"]):
            print("File ", fastqs[SAMPLE_ID][flowcell][lane]["fastq2_filename"], " does not exist!")
            sys.exit(1)


def count_fastq_records(file_path: str) -> int:
    with open(file_path, 'r') as f:
        num_lines = sum(1 for _ in f)
    return num_lines // 4


print("Fastq files: ")
for flowcell in FLOWCELLS:
    for lane in FLOWCELL_LANES[flowcell]:
        print("Flowcell: ", flowcell)
        print("Lane: ", lane)
        print(fastqs[SAMPLE_ID][flowcell][lane]["fastq1_filename"])
        print(fastqs[SAMPLE_ID][flowcell][lane]["fastq2_filename"])

        fastq_file = fastqs[SAMPLE_ID][flowcell][lane]["fastq1_filename"]
        num_records = count_fastq_records(fastq_file)
        print(f"Number of records: {num_records}", flush=True)
        fastqs[SAMPLE_ID][flowcell][lane]["num_records"] = num_records


def parse_fastq_files(file1: str, file2: str):
    """
    Parse two FASTQ files simultaneously and yield pairs of sequences.
    Yields:
      tuple: (rec1, rec2) SeqRecord objects
    """
    with open(file1) as f1, open(file2) as f2:
        for rec1, rec2 in zip(SeqIO.parse(f1, "fastq"), SeqIO.parse(f2, "fastq")):
            yield rec1, rec2


# 0924+1130 design
# primer_left = "AGGTTACAGAGTGCGGC"
# primer_right = "CCGACCAACTGCTCACC"
# barcode_left_start = 17
# barcode_left_length = 27
# barcode_right_start = 17
# barcode_right_length = 24
# lev_threshold = 3


# 0924+1130 design, immobilized
# primer_left = "AGGTTACAGAGTGCGGC"
# primer_right = "CCGACTTCCTGACCAAC"
# barcode_left_start = 17
# barcode_left_length = 27
# barcode_right_start = 25
# barcode_right_length = 24
# lev_threshold = 3


# 0319 design
# primer_left = "AGGTTACAGAGTGCGGC"
# primer_right = "CCGACCAACTGCTCACC"
# barcode_left_start = 17
# barcode_left_length = 25
# barcode_right_start = 17
# barcode_right_length = 25
# lev_threshold = 3


# 0319 design, immobilized
primer_left = "AGGTTACAGAGTGCGGC"
primer_right = "CCGACTTCCTGACCAAC"
barcode_left_start = 17
barcode_left_length = 25
barcode_right_start = 25
barcode_right_length = 25
lev_threshold = 3


# ----------------------------
# Graph-tool graph setup
# ----------------------------
g = Graph(directed=False)

# Vertex properties
vp_barcode = g.new_vertex_property("string")   # barcode string like "L...." or "R...."

# Edge properties
ep_weight = g.new_edge_property("int")         # weight / count

# Index: barcode string -> vertex
barcode_to_v = {}


def get_or_create_vertex(barcode: str):
    """
    Returns existing vertex for barcode, or creates a new vertex and sets properties.
    """
    v = barcode_to_v.get(barcode)
    if v is None:
        v = g.add_vertex()
        vp_barcode[v] = barcode
        barcode_to_v[barcode] = v
    return v


# Edge lookup: (min_vid, max_vid) -> edge
# This avoids O(E) lookups and mirrors "if has_edge then w+1".
edge_map = {}

prefix = scratch_prefix + SAMPLE_ID + "_"

all_reads = 0
all_overlap_skipped = 0

for flowcell in FLOWCELLS:
    for lane in FLOWCELL_LANES[flowcell]:
        num_records = fastqs[SAMPLE_ID][flowcell][lane]["num_records"]
        fastq1_filename = fastqs[SAMPLE_ID][flowcell][lane]["fastq1_filename"]
        fastq2_filename = fastqs[SAMPLE_ID][flowcell][lane]["fastq2_filename"]

        print("Parsing sample ", SAMPLE_ID, ", flowcell ", flowcell, ", lane ", lane, flush=True)
        total = 0
        useful = 0
        cl1 = 0
        cl2 = 0
        cr1 = 0
        cr2 = 0
        wrong_index = 0  # kept to mirror your script, though unused
        overlap_skipped = 0

        for record1, record2 in tqdm(
            parse_fastq_files(fastq1_filename, fastq2_filename),
            total=num_records,
        ):
            assert record1.id == record2.id

            total += 1
            all_reads += 1

            # New step: check paired-read suffix overlap before primer matching.
            if suffixes_overlap(record1, record2):
                overlap_skipped += 1
                all_overlap_skipped += 1
                continue

            dl1 = distance(primer_left, str(record1.seq[0:len(primer_left)]), score_cutoff=lev_threshold)
            dr2 = distance(primer_right, str(record2.seq[0:len(primer_right)]), score_cutoff=lev_threshold)
            dl2 = distance(primer_left, str(record2.seq[0:len(primer_left)]), score_cutoff=lev_threshold)
            dr1 = distance(primer_right, str(record1.seq[0:len(primer_right)]), score_cutoff=lev_threshold)

            # Determine orientation by primer matching (same logic as your script)
            if dl1 <= lev_threshold and dr2 <= lev_threshold:
                record_left = record1
                record_right = record2
            elif dl2 <= lev_threshold and dr1 <= lev_threshold:
                record_left = record2
                record_right = record1
            else:
                if dl1 <= lev_threshold:
                    cl1 += 1
                if dl2 <= lev_threshold:
                    cl2 += 1
                if dr1 <= lev_threshold:
                    cr1 += 1
                if dr2 <= lev_threshold:
                    cr2 += 1
                continue

            useful += 1

            barcode_left = "L" + str(
                record_left.seq[
                    barcode_left_start:
                    barcode_left_start + barcode_left_length
                ]
            )
            barcode_right = "R" + str(
                record_right.seq[
                    barcode_right_start:
                    barcode_right_start + barcode_right_length
                ]
            )

            vL = get_or_create_vertex(barcode_left)
            vR = get_or_create_vertex(barcode_right)

            # Canonical undirected key
            a = int(vL)
            b = int(vR)
            if a <= b:
                key = (a, b)
            else:
                key = (b, a)

            e = edge_map.get(key)
            if e is None:
                e = g.add_edge(vL, vR)
                edge_map[key] = e
                ep_weight[e] = 1
            else:
                ep_weight[e] += 1

        # Summary (mirrors your printouts and adds suffix-overlap count)
        if total > 0:
            print("Percentage of useful records: ", round(100 * useful / total, 1), "%")
            print("Percentage of L1: ", round(100 * cl1 / total, 2), "%")
            print("Percentage of L2: ", round(100 * cl2 / total, 2), "%")
            print("Percentage of R1: ", round(100 * cr1 / total, 2), "%")
            print("Percentage of R2: ", round(100 * cr2 / total, 2), "%")
            print(
                "Records skipped because of overlapping suffixes: "
                f"{overlap_skipped:,} "
                f"({100.0 * overlap_skipped / total:.2f}%)"
            )
        else:
            print("Percentage of useful records: 0.0 %")
            print("Records skipped because of overlapping suffixes: 0 (0.00%)")

        num_nodes = g.num_vertices()
        num_edges = g.num_edges()
        print(f"Number of nodes: {num_nodes:,}")
        print(f"Number of edges: {num_edges:,}")
        print()


# Attach properties to graph
g.vertex_properties["barcode"] = vp_barcode
g.edge_properties["weight"] = ep_weight

out_filename = prefix + "barcodesGraph_all.gt.gz"
print("Saving to file:", out_filename)

# Save as compressed graph-tool format (.gt.gz is inferred from suffix)
g.save(out_filename)

if all_reads > 0:
    all_overlap_percentage = 100.0 * all_overlap_skipped / all_reads
else:
    all_overlap_percentage = 0.0

print()
print("Overall suffix-overlap filtering summary:")
print(f"Total records processed: {all_reads:,}")
print(
    "Records skipped because of overlapping suffixes: "
    f"{all_overlap_skipped:,} "
    f"({all_overlap_percentage:.2f}%)"
)

print("DONE")