# SPLIT

This tool is designed to analyze genetic data, determining the number of structural variants in a reading frame using statistical inference.

## Dependencies

### Python

- Tested with python version 3.11.3 and 3.12.10.
  - If running on Fiji, use `module load python/3.11.3`

### Python packages

- Create a python environment and install the packages listed in the minimum requirements file.

```
python -m venv venv
source venv/bin/activate
pip install -r min_requirements.txt
pip install -e .
```

- If you have any problems, feel free to refer to the `requirements.txt` file to see the full list of packages used in the paper.

### GIGGLE index & STIX

To create a GIGGLE index (a database for raw alignments), you will need to follow instructions from [GIGGLE GitHub](https://github.com/ryanlayer/giggle/tree/master). In addition, you will need to follow STIX download instructions from [STIX GitHub](https://github.com/ryanlayer/stix/tree/master).

#### A. List of SVs and the genotype of patients

- A file containing the SV information from samples for all the deletions of interest. Can be provided as a **vcf (or vcf.gz) or a comma-delimited file** (details below). If a VCF is provided, the software will create the comma-delimited file for faster next usage instead of the VCF.

  - CSV with the columns id, chr, start, stop, svlen, ref, alt, qual, filter, af, info, sample1, ..., sampleN, num_samples
  - info is a python dictionary
  - Example comma-delimited version is deletions.csv in assets/ (part shown below)

  EXAMPLE comma-delimited file version of sv_lookup

  ```
  id,chr,start,stop,svlen,ref,alt,qual,filter,af,info,HG00118,HG00119,num_samples
  HGSV_3,1,39999,107150,67151,N,<DEL>,120.0,['PASS'],0.11301916932907348,"{'SVTYPE': 'DEL', 'CHR2': 'chr1', 'SVLEN': -67150, 'ALGORITHMS': ('depth',), 'EVIDENCE': ('BAF', 'RD'), 'AC': (566,), 'AN': 4016, 'SOURCE': 'gatksv', 'ORIGIN_SVID': ('1KGP_2504_and_698_with_GIAB_DEL_chr1_1',), 'AF': (0.14177699387073517,), 'N_BI_GENOS': 2578, 'N_HOMREF': 2002, 'N_HET': 421, 'N_HOMALT': 155, 'FREQ_HOMREF': 0.7765709757804871, 'FREQ_HET': 0.16330499947071075, 'FREQ_HOMALT': 0.060124099254608154, 'MALE_AN': 2568, 'MALE_AC': (360,), 'MALE_AF': (0.14018699526786804,), 'MALE_N_BI_GENOS': 1284, 'MALE_N_HOMREF': 1003, 'MALE_N_HET': 202, 'MALE_N_HOMALT': 79, 'MALE_FREQ_HOMREF': 0.7811530232429504, 'MALE_FREQ_HET': 0.1573210060596466, 'MALE_FREQ_HOMALT': 0.061526499688625336, 'FEMALE_AN': 2588, 'FEMALE_AC': (371,), 'FEMALE_AF': (0.14335399866104126,), ...'SAN_FEMALE_FREQ_HOMALT': 0.04035869985818863, 'POPMAX_AF': 0.3734019994735718}","(0,0)","(0,1)",2
  ```

#### B. Read-based data

SPLIT uses information of sample read data within a given SV coordinate space to cluster and identify SVs. You can provide this in 3 ways:

1. STIX database/index of CRAMS/BAMS. Please refer to the [STIX github](https://github.com/ryanlayer/stix/tree/master) on how to create this.

- If using this approach, you will simply provide the paths to the STIX index, STIX database, where you store the active software, and the number of shards used to build the database.
- In this case, SPLIT will query the database for read evidence of an SV for you.

2. Read information already queried from a database in a tab-delimited-file with the naming convention being the same as that of the SV you are testing (e.g. if I'm considering SV chr3:1000-2000 then the file must be named 3:1000_3:2000.txt).

- Format is tab-delimited file with columns File ID, File Name, Chr, Left Start, Left End, Chr, Right Start, Right End, whether based on Paired or Split reads

```
3	bed_0/HG00122.bed.gz	3	173522849	173522939	3	173524168	173524318	paired
3	bed_0/HG00122.bed.gz	3	173522873	173522939	3	173524105	173524255	paired
```

3. PROCESSED evidence in a comma-delimited-file with the same naming convention (e.g. if I'm considering SV chr3:1000-2000 then the file must be named 3:1000-1000_3:2000-2000.csv)

- Format is comma-seperated file with columns Sample, XXXX

```
HG00102,173522744,173524280,173522761,173524248,173522840,173524340,173522842,173524369,173522861,173524496,173522862,173524445,173522866,173524347,173522888,173524262
HG00103,173522759,173524359,173522762,173524364,173522879,173524244,173522885,173524384,173522898,173524299
HG00110,173522816,173524440,173522826,173524457,173522837,173524327,173522878,173524414,173522884,173524303
HG00122,173522849,173524318,173522873,173524255,173522881,173524392
HG00149,173522763,173524247,173522838,173524265
```

#### C. Mean insert-sizes of patients (or default=450)

- `insert_sizes`: Provides the mean insert size for the samples to be used in the algorithm. If the file is not provided, a default of 450 is used.

  Example file:

  ```
  sample_id,mean_insert_size
  NA20509,440
  HG02941,436
  ```

### Config File

This file defines all file paths, input file names, executables (STIX and bedtools), query parameters, model parameters, and calibration parameters. An example is provided in data/assets/default_config.toml.

This file is loaded into all scripts in the scripts/ directory. You can either provide a path to the config file or place it in your home directory with the name "config.toml".

#### Input and output directories

- `input_dir`: Directory containing your input data (VCF/CSV callset, sample IDs, insert sizes, etc.).
- `output_dir`: Directory where results will be written.
- `stix_output_dir`: Sub-directory where raw STIX query output files are cached between runs.
- `intermediate_output_dir`: Intermediate directory used during processing. On HPC clusters, point this to a fast local scratch filesystem. On a local machine, a subdirectory of output_dir is fine.
- `local_intermediate_output_dir`: Local directory to store intermediate output files. Files are first written to intermediate_output_dir, then moved to local_intermediate_output_dir if they are not the same.

#### Input files

- `sv_lookup_file`: VCF or CSV file listing structural variants.
- `sample_id_file`: TXT file with one row for each sample ID.
- `insert_size_file`: Per-sample mean insert size file (two columns: sample_id, mean_insert_size). If this file is absent a uniform default of default_insert_size bp is written automatically.
- `ancestry_file`: Ancestry file with required columns "Sample name" and "Superpopulation code"

#### STIX

- `bin`: Absolute path to the compiled STIX binary.
- `index`: Absolute path to the STIX index file
- `database`: Absolute path to the STIX database.
- `num_shards`: Number of shards the index and database are split into. Set to 1 if your index is a single file.

#### Query

- `read_overlap`: Fraction of the SV length that a flanking read is allowed to overlap the SV region and still be included. 1.0 = full overlap allowed; 0.5 = half the SV length.

#### Calibrate (Required for scripts.calibrate_model)

- `truth_set`: Truth set to calibrate against (in input_dir). Required columns: chr, start, stop, n_svs_actual
- `search_func`: Search function to use for calibration. Supported algorithms: bo (bayesian optimization) or grid (grid search)
- Hyperparameter boundaries: Specify the min and max values the parameter can take. If using grid search specify the step size as well.
  - `d_min`, `d_max`, `d_step`: Minimum distance between SV cluster centroids in L, length space at which to start penalizing the model.
  - `r_min`, `r_max`, `r_step`: Minimum reciprocal overlap between SV clusters at which to start penalizing the model.
  - `q_min`, `q_max`, `q_step`: Fraction of the SV length that a flanking read is allowed to overlap the SV region and still be included.
  - `p_min`, `p_max`, `p_step`: Maximum penalty term for spurious cluster suppression.

## Arguments

To split a single SV: `python -m scripts.split_one [--config]`

```
Required arguments:
    --l                           Leftmost coordinate of SV (format is chr:Num) (Required if svid not used)
    --r                           Rightmost coordinate of SV (format is chr:Num) (Required if svid not used)
    OR
    --id                          Structural variant id from the input callset (Required if l and r not provided)

Processing Flags:
    -p                           If include then will plot the Length and L-coordinate of each sample
    -d                           If include then will continue to rerun algorithm until acheives ≥ 80% confidence

```

To split an entire callset: `python -m scripts.split_all [--config]`

## Output

### Intermediate Files saved to input directory (if not already provided):

- Sv-lookup as csv (if VCF provided) (refer to Required Input Files)
- Raw alignments queried from STIX for provided genomic regions/SV coordinates

### Standard Output (printed)

The algorithm will iterate through trials where the outcome refers to the mode number with the greatest probability (probabilities = [probability of 1 mode, probability of 2 modes, probability of 3 modes])

```

Trial 1: outcome=2, probabilities=[0.25 0.5 0.25]
Trial 2: outcome=2, probabilities=[0.2 0.6 0.2]
Trial 3: outcome=2, probabilities=[0.16666667 0.66666667 0.16666667]
Trial 4: outcome=2, probabilities=[0.14285714 0.71428571 0.14285714]
Trial 5: outcome=2, probabilities=[0.125 0.75 0.125]
Trial 6: outcome=2, probabilities=[0.11111111 0.77777778 0.11111111]
Trial 7: outcome=2, probabilities=[0.1 0.8 0.1]
Trial 8: outcome=2, probabilities=[0.09090909 0.81818182 0.09090909]
3:173522965-173524108 - stopping after 8 iterations, 2 modes, ci=[array([0.63104355]), array([0.82350191])]

```

### Plots

If using the flag `-p`, a plot showing the clusters of SVs identified based on length and L-coordinate are shown. Saved in output_directory/plots. Plot you should get from running the test example:
![alt text](README_images/3:173522965-3:173524108_LengthLCoord_plot.png)

## Test

Test if you can successfully run with `test_run.sh`. This uses the provided files and config in the assets directory.

You should get the outputs and results shown in "Output" above.

## More details on approach

### How does SPLIT query STIX database for evidence of an SV (e.g. what is in the unprocessed evidence (.txt file))?

- It queries STIX for all evidence within the region, with some tolerance from each end of the provided SV. Short-read and long-read data can be used, depending on the STIX indices that are available. The STIX output is saved in an intermediate output directory to prevent having to re-query STIX (can be costly in time).

### How does SPLIT process the queried data?

SPLIT removes samples that are homozygous for the reference genotype (0, 0) and takes the median start/end coordinates for each sample. The median L-coordinate and SV length (calculated as: R - L - mean_insert_size) are used to cluster the samples.

### How does SPLIT identify clusters/merged SVs?

Each sample is represented as a 2D point (L, length). SPLIT then runs a Gaussian Mixture Model over all samples within the region and performs a model selection to select the best fitting distribution (1, 2, or 3 clusters). Finally, it assigns samples to the clusters if more than 1 SV was found in the region.

Any region with 10 or fewer samples will be marked as "inconclusive".

<!-- #### Best Distribution Over All Data Points

<img width="500" alt="GMM Modes" src="https://github.com/user-attachments/assets/e0925a5d-7a4e-4b28-8bc8-ad157b835d8e"> -->

## Helpful Results from Paper

#### Left-Right Coordinates of Points for Each Distribution & Ancestry Splits

<img width="500" alt="L-R Points & Ancestry" src="https://github.com/user-attachments/assets/419ee98d-0fe6-4962-9fcf-329faa09abd8">

#### Length of SVs For Each Mode

<img width="500" alt="SV Lengths" src="https://github.com/user-attachments/assets/d50976f6-5989-41f7-85af-8706c194c25a">

---

## Synthetic Data Workflow

Synthetic data is generated to test the accuracy of SPLIT and GATK-ClusterBatch with increasing reciprocal overlap, _r_ for three different test cases. Noise is added to generated data to replicate short-read alignments. Running `python -m scripts.synthetic_tests` with varying sample counts and SV lengths will replicate the process described in the paper.

## 1kG Data Processing

The data used for this project originates from the 1000 Genomes Project. A pre-built [STIX](https://github.com/ryanlayer/stix) index is used to query genomic regions for short and long reads (by sample) that serve as evidence for the SV of interest. Low-coverage short read, high-coverage short read, and long-read data are all available. The deletions that were present in the initial 1kG callset are found in `assets/1kg.subset.vcf.gz`.

### Post-Processing & Figures

After writing the `all_split_trials.csv` file, run the `write_post_processed_files` function in `helper.py` to generate several other files useful in downstream analysis.

- `svs_n_modes.csv`: number of modes and confidence predicted for each SV with evidence in STIX
- `consensus_svs.csv`: consensus SVs by averaging the start/stop/length of each mode across all runs of the SV
- `sv_stats_collapsed.csv`: SV stats with 1 row for each SV based on the most common SV clustering
- `outliers.csv`: outliers identified based on a threshold
- `ancestry_dissimilarity.csv`: bray curtis dissimilarity calculated between clusters for SVs split into 2+ clusters
- `new_gene_intersections.bed` (needs a working build of bedtools): new gene intersections resulting from the split SVs
