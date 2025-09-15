# SV_GMM

This tool is designed to analyze genetic data, determining the number of structural variants in a reading frame using statistical inference.

Input: chr:left_paired_end_start-left_paired_end_end, chr:right_paired_end_start-right_paired_end_end\
Example: python query_sv.py -l 1:113799624-113799624 -r 1:113800089-113800089

### 1a. Short-read data: Query STIX for all evidence within the region, +- 50 bp from each end

| File ID | File Name               | Chr | Left Start | Left End  | Chr | Right Start | Right End | Paired/Split |
| ------- | ----------------------- | --- | ---------- | --------- | --- | ----------- | --------- | ------------ |
| 0       | alt_sort/HG00096.bed.gz | 1   | 113799540  | 113799639 | 1   | 113800087   | 113800187 | paired       |
| 1       | alt_sort/HG00097.bed.gz | 1   | 113799542  | 113799642 | 1   | 113800287   | 113800388 | paired       |
| 2       | alt_sort/HG00099.bed.gz | 1   | 113799516  | 113799616 | 1   | 113800220   | 113800321 | paired       |
| 3       | alt_sort/HG00100.bed.gz | 1   | 113799234  | 113799333 | 1   | 113800090   | 113800190 | paired       |
| 3       | alt_sort/HG00100.bed.gz | 1   | 113799235  | 113799334 | 1   | 113800139   | 113800238 | paired       |

### 1b: Long-read data: Parse the cigar string for all available 1kg samples with long-read data

From the sample's entire cram file, download the bam file corresponding to an SV region (start - tolerance, stop + tolerance). Look for instances of "D" in the cigar string, corresponding with a deletion in the selected region. Use the size of the original SV +- an additional tolerance to find deletions that match the original SV. Using the reference, calculate the start/stop/length of the deletion.

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
