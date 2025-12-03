# SV_GMM

This tool is designed to analyze genetic data, determining the number of structural variants in a reading frame using statistical inference.

## Requirements
* Tested with python version 3.11.3 and 3.12.10. 
  * If running on Fiji, use `module load python/3.11.3`
* Create a python environment and install the packages listed in requirements
```
python -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```



## 1. Get the Sample SV evidence coordinates

#### 1A. Edit query_stix.sh, query_stix_multifile.sh, query_stix_lr.sh to use the correct file paths for the STIX database (indices) and STIX build.

#### 1B. Run query_sv.py to query the samples and reads found within the coordinates of a predefined structural variant

* For short-read data (default), queries STIX for all evidence within the region, +- 50 bp from each end of the provided SV
* For long-read data (requires option `-lr`), Parses the cigar string for all available 1kg samples with long-read data
  * From the sample's entire cram file, download the bam file corresponding to an SV region (start - tolerance, stop + tolerance). Look for instances of "D" in the cigar string, corresponding with a deletion in the selected region. Use the size of the original SV +- an additional tolerance to find deletions that match the original SV. Using the reference, calculate the start/stop/length of the deletion.

Input: chr:left_paired_end_start-left_paired_end_end, chr:right_paired_end_start-right_paired_end_end\
Example: python query_sv.py -l 1:113799624-113799624 -r 1:113800089-113800089

**NOTES FROM HOPE:** 
* I'd like to see the full arguments for this since there are several missing (e.g. don't I need)
* What are ALL the outputs I get? And what files do I need available?
* What are the software requirements?
* I also don't understand the input. From my understanding, this is where I'd get the samples and coordinates that would support an SV of interest (e.g. deletion between positions 1000 and 2000 of chr2). I believe these would be some helpful examples:
* Example 1: Deletion at chr2 between positions 1000 and 2000 in gr38: `python query_sv.py -l 2:1000-1000 -r 2:2000-2000 -ref grch38`
* Example 2: SV Id from 1000 genomes project of SV1000 in grch37: `python query_sv.py -id SV1000 -ref grch37`
* Example 3: Deletion at chromsome 10 with end points estimated to be between 2000 and 2050, and 4000 and 4100, using long read data: `python query_sv.py -l 10:2000-2050 -r 10:4000-4100 -lr` -- THIS MAY BE INCORRECT BASED ON THE CODE AS THE CODE WOULD DO 2050-4100 (reverse_giggle_format).

#### Example Output for Short-read data
| File ID | File Name               | Chr | Left Start | Left End  | Chr | Right Start | Right End | Paired/Split |
| ------- | ----------------------- | --- | ---------- | --------- | --- | ----------- | --------- | ------------ |
| 0       | alt_sort/HG00096.bed.gz | 1   | 113799540  | 113799639 | 1   | 113800087   | 113800187 | paired       |
| 1       | alt_sort/HG00097.bed.gz | 1   | 113799542  | 113799642 | 1   | 113800287   | 113800388 | paired       |
| 2       | alt_sort/HG00099.bed.gz | 1   | 113799516  | 113799616 | 1   | 113800220   | 113800321 | paired       |
| 3       | alt_sort/HG00100.bed.gz | 1   | 113799234  | 113799333 | 1   | 113800090   | 113800190 | paired       |
| 3       | alt_sort/HG00100.bed.gz | 1   | 113799235  | 113799334 | 1   | 113800139   | 113800238 | paired       |

#### Example Output for Long-read data
| File ID | File Name               | Chr | Left Start | Left End  | Chr | Right Start | Right End | Paired/Split |
| ------- | ----------------------- | --- | ---------- | --------- | --- | ----------- | --------- | ------------ |
| 0       | alt_sort/HG00096.bed.gz | 1   | X  | X | 1   | X   | X | Split       |

### 2. Filter out reference samples & process the output so that the each line contains pairs of evidence that correspond with one sample

HG00096,113799540,113800187\
HG00097,113799542,113800388\
HG00099,113799516,113800321\
HG00100,113799234,113800190,113799235,113800238,113799318,113800230,113799328,113800353,113799349,113800342,113799356,113800259,113799379,113800269,113799403,113800296,113799467,113800440\
HG00101,113799430,113800209,113799529,113800307,113799553,113800389\

### 3. Reduce each sample to one point

Take the mean L and mean length of each paired-end read so that each sample is represented as a 2D point (length, L).

### 4. Run GMM

Decide if the points are most likely to fit a 1, 2, or 3 mode distribution. Runs the EM algorithm for 30 iterations for each of the distributions and compares the AIC scores for each model. Any SV with 10 or fewer samples will be marked as "inconclusive".

<!-- #### Best Distribution Over All Data Points

<img width="500" alt="GMM Modes" src="https://github.com/user-attachments/assets/e0925a5d-7a4e-4b28-8bc8-ad157b835d8e"> -->

### 5. Assign points to modes if more than 1 SV was found in the region

#### Left-Right Coordinates of Points for Each Distribution & Ancestry Splits

<img width="500" alt="L-R Points & Ancestry" src="https://github.com/user-attachments/assets/419ee98d-0fe6-4962-9fcf-329faa09abd8">

#### Length of SVs For Each Mode

<img width="500" alt="SV Lengths" src="https://github.com/user-attachments/assets/d50976f6-5989-41f7-85af-8706c194c25a">

---

## Synthetic Data Workflow

Synthetic data is generated to test the accuracy of SVeperator and GATK-ClusterBatch with increasing reciprocal overlap, _r_ for five different test cases. The `run_synthetic_data.sh` sbatch script on Fiji runs the `r_accuracy_test` function in `synthetic_tests.py`, varying the sample size and sv length.

```
n_samples_values=(10 21 66 206 313)
svlen_values=(51 167 802 3377 17352)
for n_samples in "${n_samples_values[@]}"; do
  for svlen in "${svlen_values[@]}"; do
    echo "Running with n_samples=$n_samples and svlen=$svlen"
    start_time=$(date +%s)
    python3 "$HOME/sv/sv_gmm/synthetic_tests.py" "$n_samples" "$svlen"
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "Completed n_samples=$n_samples and svlen=$svlen in $elapsed_time seconds"
  done
done
```

Noise is added to generated data to replicate short-read data, and results are saved in the synthetic_data directory by sample size. The results for a given sample size and sv length can be plotted with the `plot_reciprocal_overlap_all` function in `viz.py`.

## 1kG Data Processing

The data used for this project originates from the 1000 Genomes Project's original 2504 samples. All data is found in the `/scratch/Shares/layer/stix/indices/` directory on fiji. A pre-built [STIX](https://github.com/ryanlayer/stix) index is used to query genomic regions for short and long reads (by sample) that serve as evidence for the SV of interest. Low-coverage short read, high-coverage short read, and long-read data are all available. The deletions that were present in the initial 1kG analysis are found in `1kgp/1kg_hg38_deletions.vcf` or `1kgp/deletions_df.csv`.

### Low-coverage short-reads

Low-coverage data is supported by not used in the main analysis. Depending on the reference genome, the appropriate STIX index is in the `1kg_hg37_low_cov` or `1kg_hg38_low_cov` directory.

### High-coverage short-reads

High-coverage data is found in the `1kg_high_coverage_vivian` directory.
To run the analysis for a single SV, run `python query_sv.py` with the -id or -l and -r arguments.
To run the pipeline on the entire dataset, run `python run_svs_until_converge.py`. The dirichlet process is run for each SV until "convergence" (we have high confidence in the outcome or 100 trials). A new file is written into the `processed_svs_converge` directory, and the files are concatenated into one large file (`sv_stats_converge.csv`) at the end of the function. The process is parallelized and runs on fiji.

### Long-reads

Long-read data is found in the `LR_vivian` directory. Once the data has been pre-processed, run `python run_lr_until_converge.py` to run the dirichlet process.

### Post-Processing & Figures

After writing the `sv_stats_converge.csv` file, run the `write_post_processed_files` function in `helper.py` to generate several other files useful in downstream analysis.

- `svs_n_modes.csv`: number of modes and confidence predicted for each SV with evidence in STIX
- `sr_lr_merged.csv` (for long reads only): compares the number of modes and the confidence for each SV outcome between short and long read results
- `consensus_svs.csv`: consensus SVs by averaging the start/stop/length of each mode across all runs of the SV
- `sv_stats_collapsed.csv`: SV stats with 1 row for each SV based on the most common SV clustering
- `outliers.csv`: outliers identified based on a threshold
- `ancestry_dissimilarity.csv`: bray curtis dissimilarity calculated between clusters for SVs split into 2+ clusters
- `new_gene_intersections.bed` (needs a working build of bedtools): new gene intersections resulting from the split SVs

### Flowchart Figure

To create the flowchart figure, we need:

- the number of total deletions (from `deletions_df.csv`)
- the number of SVs with no or too little evidence in STIX (<= 10 samples)
  - no evidence: `deletions_df.shape[0] - svs_n_modes.shape[0]`
  - little evidence: `svs_n_modes[svs_n_modes["confidence"] == "inconclusive"].shape[0]`
  - total number: `deletions_df.shape[0] - svs_n_modes[svs_n_modes["confidence"] != "inconclusive"].shape[0]`
- the number of SVs with an outcome
  - `svs_n_modes[svs_n_modes["confidence"] != "inconclusive"].shape[0]`
- the number of high confidence, medium confidence, and low confidence SVs
  - ex: `svs_n_modes[svs_n_modes["confidence"] == "low"].shape[0]`
- the predicted number of clusters depending on confidence
  - ex: `Counter(svs_n_modes[svs_n_modes["confidence"] == "high"]["num_modes"])`
- the second most likely number of clusters (in low confidence situation)
  - `Counter(svs_n_modes[svs_n_modes["confidence"] == "low"]["num_modes_2"])`
