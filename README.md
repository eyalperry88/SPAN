# SPAN

Code accompanying *Archival data storage using DNA proximity graphs* (SPAN: Synthesis-free
Proximity-graph Archiving in Nucleic acids).

## Repository structure

| Folder | Contents |
|---|---|
| [`pipeline/`](pipeline) | From paired-end FASTQ files to a barcode proximity graph, barcode error correction, graph filtering, and 2D reconstruction with [STRND](https://github.com/DavidFernandezBonet/STRND) |
| [`evaluate/`](evaluate) | Compare independent reconstructions of the same graph (e.g. different STRND random seeds) to check their consistency |
| [`decoding/`](decoding) | Turn a reconstruction into a grayscale density image and decode the bit pattern with manually placed fiducials |

## Installation

```bash
conda env create -f environment.yml
conda activate span
```

`graph-tool` is only available through conda-forge (or system packages), not pip. STRND is installed
from PyPI (`pip install strnd`) by the environment file.

## Data layout and file naming

All scripts and notebooks read and write under a single data root, set with the `SPAN_DATA_ROOT` environment
variable (default `./orcd`, relative to the directory you run from, so setting an absolute path is safest):

```
$SPAN_DATA_ROOT/
├── pool/<PROJECT>/       # raw FASTQ files (input to step 01)
└── scratch/<PROJECT>/    # every graph and output produced by the analysis
```

```bash
export SPAN_DATA_ROOT=/path/to/data
```

Each step reads `scratch/<PROJECT>/<SAMPLE>_barcodesGraph_<SUFFIX>.gt.gz` and writes a new file
whose suffix extends the input suffix. You pass the input suffix to the next step with `--suffix`,
so the suffix records every processing step applied to a graph.

Graphs are undirected [graph-tool](https://graph-tool.skewed.de/) files with the vertex property
`barcode` (string; prefix `L` or `R` marks the two oligonucleotide classes) and the edge property
`weight` (number of supporting reads). Steps 02 and 03 also add `vcount` (number of merged vertices).

## `pipeline/` — FASTQ to reconstruction

The pipeline goes from paired-end FASTQ files to a barcode proximity graph, cleans up barcode
errors, filters the graph, and reconstructs 2D barcode coordinates with STRND.

### Overview

| Step | File | What it does | Paper (Methods) |
|---|---|---|---|
| 01 | [`01_build_barcode_graph.py`](pipeline/01_build_barcode_graph.py) | Extract the two barcodes from each read pair and build the bipartite barcode graph (edge weight = number of reads) | Sequencing; Proximity graph reconstruction |
| 02 | [`02_hamming1_contraction.py`](pipeline/02_hamming1_contraction.py) | Merge barcodes at Hamming distance 1 | Sequencing (Hamming ≤ 2) |
| 03 | [`03_hamming2_contraction.py`](pipeline/03_hamming2_contraction.py) | Merge barcodes at Hamming distance ≤ 2 | Sequencing (Hamming ≤ 2) |
| 04 | [`04_neighbor_contraction.py`](pipeline/04_neighbor_contraction.py) | Merge barcodes that are a graph path of length 2 apart and within a Levenshtein distance | Proximity graph reconstruction (≤ 3 edits) |
| 05 | [`05_filter_graph.ipynb`](pipeline/05_filter_graph.ipynb) | Inspect the graph, take the largest connected component (LCC), filter by edge weight and degree, strip degree-1 vertices, save | Proximity graph reconstruction (weight ≥ 3, remove degree 1) |
| 06a | [`06a_count_indirect_paths.py`](pipeline/06a_count_indirect_paths.py) | *Optional.* Count, for every edge, the simple paths of length 3 and 5 between its endpoints that avoid the edge | — |
| 06b | [`06b_filter_indirect_paths.ipynb`](pipeline/06b_filter_indirect_paths.ipynb) | *Optional.* Keep only edges with enough indirect-path support, save the LCC | — |
| 07 | [`07_strnd_reconstruction.py`](pipeline/07_strnd_reconstruction.py) | Reconstruct 2D coordinates with STRND, save as CSV | Proximity graph reconstruction |

Steps 01–04 and 06a–07 are command-line scripts (designed for a Linux/HPC node; 02–04 and 06a use
`multiprocessing` with `fork`). Steps 05 and 06b are Jupyter notebooks.

### Running the steps

Run the commands from inside `pipeline/`. The examples use `PROJECT=260921Jac` and `SAMPLE=M1`.

#### 01 — Build the barcode graph from FASTQ

```bash
python 01_build_barcode_graph.py PROJECT SAMPLE FLOWCELL_LANES SAMPLE_IDX
python 01_build_barcode_graph.py 260921Jac M1 "7393S:2;3" <SAMPLE_IDX>
```

- `FLOWCELL_LANES`: comma-separated `FLOWCELL:lane;lane` entries, e.g. `7393S:2;3,7389S:3;4`.
  Use `FLOWCELL:0` when the run has no per-lane split. Quote the argument, since `;` is special in the shell.
- `SAMPLE_IDX`: the sample index in the sequencing-core file names.
- Expected FASTQ paths (MIT BioMicro Center naming):
  `pool/<PROJECT>/D26-<IDX>-<LANE>-<FLOWCELL>/<PROJECT>_D26-<IDX>-<LANE>_{1,2}_sequence.fastq`
  (or `pool/<PROJECT>/D26-<IDX>-<FLOWCELL>/<PROJECT>_D26-<IDX>_{1,2}_sequence.fastq` for lane `0`).
  Edit the `fastqs[...]` block in the script if your files are named differently.
- Library design (primer sequences, barcode offsets and lengths) is set in the script; the active block
  is the design used in the paper (25-nt barcodes). Earlier designs are kept as comments.
- Read pairs whose last 18 nt are reverse complements of each other (≤ 3 edits) are discarded; the
  remaining pairs are oriented by matching the left/right primers (≤ 3 edits), and the barcodes are
  cut at fixed offsets.

Output: `<SAMPLE>_barcodesGraph_all.gt.gz`

#### 02 — Hamming-1 contraction

```bash
python 02_hamming1_contraction.py 260921Jac M1 --suffix all
```

Output: `<SAMPLE>_barcodesGraph_all_Hamming1.gt.gz`. Each merged group keeps the barcode of its
vertex with the largest weighted degree; parallel edges are summed.
Options: `--workers`, `--partitions`, `--chunk_size`, `--keep_tmp`.

#### 03 — Hamming-2 contraction

```bash
python 03_hamming2_contraction.py 260921Jac M1 --suffix all_Hamming1
```

Output: `<SAMPLE>_barcodesGraph_all_Hamming1_Hamming2.gt.gz`. Same merge rule as step 02, with
Hamming distance ≤ 2.

#### 04 — Neighborhood (Levenshtein) contraction

```bash
python 04_neighbor_contraction.py 260921Jac M1 --suffix all_Hamming1_Hamming2 --k 2 --dist 3 --workers 32
```

For each vertex, looks at vertices an even number of hops away (up to `--k`) with the same `L`/`R`
class and merges those within Levenshtein distance `--dist`; repeats until nothing merges.
The paper uses `--k 2 --dist 3` (the script defaults are `--k 4 --dist 4`).

Output: `<SAMPLE>_barcodesGraph_all_Hamming1_Hamming2_Lev3_contractedK2.gt.gz`

#### 05 — Inspect and filter (notebook)

Open `05_filter_graph.ipynb`, set `project_name`, `SAMPLE_ID` and `suffix` (the step 04 output) in
the parameter cells, and run all cells. The notebook:

1. loads the graph and prints basic statistics;
2. keeps the largest connected component;
3. shows optional diagnostics (degree and edge-weight distributions, LCC size vs. thresholds) to help
   choose the filter;
4. in the last cell, keeps edges with weight ≥ `min_w`, takes the LCC, removes vertices with degree
   \> `max_deg`, repeatedly removes degree-1 vertices, takes the LCC again, and saves.

The paper uses `min_w = 3` (and `max_deg = 30`). Output:
`<SAMPLE>_barcodesGraph_postPipeline0302_Lev3_min_e3_min_d2_max_d30_LCC.gt.gz`

#### 06a / 06b — Indirect-path filter (optional)

An alternative filter that keeps edges confirmed by other short paths in the graph.

```bash
python 06a_count_indirect_paths.py --project 260921Jac --sample-id M1 \
    --suffix postPipeline0302_Lev3_min_e3_min_d2_max_d30_LCC --workers 96
```

Adds edge properties `indirectP3` and `indirectP5` and writes `<input>_indirectPaths.gt.gz`
(`--input` / `--output` accept explicit paths instead).

Then open `06b_filter_indirect_paths.ipynb`, set `suffix` to the 06a output, and run it. It plots the
counts and saves the LCC after either filter:

- `indirectP3 >= p3_filter` OR `indirectP5 >= p5_filter` → `<input>_filter_<p3>_<p5>.gt.gz`
- `indirectP3 + indirectP5 >= sum_filter` → `<input>_sumFilter_<n>.gt.gz`

#### 07 — STRND reconstruction

```bash
python 07_strnd_reconstruction.py 260921Jac M1 --suffix postPipeline0302_Lev3_min_e3_min_d2_max_d30_LCC
```

The input graph must be connected (the output of step 05 or 06b is). Main options:
`--weighted` (use edge weights), `--weight_to_distance`, `--node_embedding_components` (default 32),
`--umap_n_neighbors` (120), `--umap_min_dist` (0.5), `--p`, `--q`, `--workers`, `--random_state` (42).

Output: `<SAMPLE>_positions_<SUFFIX>_STRND_d32_rnd42.csv` with one row per vertex (`node_ID`, `x`, `y`);
`node_ID` is the vertex index in the input graph, so barcodes can be looked up with
`g.vp["barcode"][node_ID]`.

## `evaluate/` — Compare reconstructions

Reconstruction algorithms such as STRND are stochastic. These notebooks evaluate a reconstruction by
comparing several random instantiations of it (e.g. step 07 run with different `--random_state`
values) to each other. One reconstruction is the reference: it is mapped onto a grid or a color
scheme, and each of the others is painted with the same per-barcode scheme. Consistent
reconstructions show the same grid or color field, up to rotation, reflection and scale.

| File | Scheme |
|---|---|
| [`compare_reconstructions_grid.ipynb`](evaluate/compare_reconstructions_grid.ipynb) | Barcodes on a grid of lines in the reference are drawn black in every reconstruction (other barcodes blue/red by L/R class) |
| [`compare_reconstructions_color.ipynb`](evaluate/compare_reconstructions_color.ipynb) | Every barcode is colored by its position in the reference (angle → hue, radius → saturation); the same colors are used for every reconstruction, and a color key is saved |

To run: produce several reconstructions of the same graph, for example

```bash
cd pipeline
for seed in 42 123 999; do
  python 07_strnd_reconstruction.py 260921Jac M1 --suffix <SUFFIX> --random_state $seed
done
```

then set `project_name`, `sample_name`, `suffix` and the list of reconstruction tags (`d32_rnd42`,
`d32_rnd123`, …, from the step 07 file names) in the notebook and run all cells. The images are
saved to `scratch/<PROJECT>/eval_grid/` and `scratch/<PROJECT>/eval_color/`.

## `decoding/` — Decode the bit pattern

| Step | File | What it does |
|---|---|---|
| 01 | [`01_export_density_image.ipynb`](decoding/01_export_density_image.ipynb) | Make a grayscale density image from one reconstruction |
| 02 | [`02_decode_bits.html`](decoding/02_decode_bits.html) | Fit a lattice with manual fiducials and decode the bits against a ground-truth pattern |

### 01 — Density image (notebook)

Set `INPUT_CSV` to a step 07 positions CSV and run all cells. The notebook normalizes the coordinates
(centered, central 95% scaled to [-1, 1]), optionally rotates/flips them, counts points in a square
sliding window (`WINDOW_WIDTH_NORM`), and maps the counts to 8-bit grayscale: at or below the 10th
percentile → white, at or above the 90th percentile → black, linear in between. It saves a borderless
PNG plus a JSON file with the settings to `<csv folder>/density_exports/`. Rotation and flips only
change how the image looks; the decoder's fit handles any orientation.

### 02 — Bit decoder (HTML)

Open `02_decode_bits.html` in a web browser; it runs locally and needs no server or internet
connection.

1. Load the PNG from step 01.
2. Add fiducials: click a bit in the ground-truth panel, then click the matching location in the image.
   With 3 or more pairs an affine lattice is fitted; with 6 or more, a regularized quadratic warp can be
   used to correct distortion.
3. Each lattice site is sampled with a circle (diameter set as a fraction of the local lattice pitch),
   and its mean pixel value is thresholded: dark (DNA-rich) sites → 0, light (DNA-depleted, written)
   sites → 1. The threshold defaults to the midpoint of the 25th and 75th percentiles of the site
   means and can be adjusted using the histogram.
4. The decoded panel compares the result with the ground truth: black/white = correct,
   red fill = false 1, red outline = false 0, plus the number of correct bits.

The ground-truth pattern (17 × 8, as in the paper) is set in `TRUTH_ROWS` at the top of the script;
edit `NX`, `NY` and `TRUTH_ROWS` to decode other patterns.

## License

MIT — see [LICENSE](LICENSE).
