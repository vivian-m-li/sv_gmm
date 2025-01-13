# SV_GMM

This tool is designed to analyze genetic data, determining the number of structural variants in a reading frame using statistical inference.

Input: chr:left_paired_end_start-left_paired_end_end, chr:right_paired_end_start-right_paired_end_end\
_query_stix("1:113799624-113799624", "1:113800089-113800089")_

### 1. Query STIX for all evidence within the region, +- 50 bp from each end
   
|File ID |File Name               |Chr     |Left Start      |Left End        |Chr     |Right Start     |Right End       |Paired/Split|
|--------|------------------------|--------|----------------|----------------|--------|----------------|----------------|------------|
|0       |alt_sort/HG00096.bed.gz |1       |113799540       |113799639       |1       |113800087       |113800187       |paired      |
|1       |alt_sort/HG00097.bed.gz |1       |113799542       |113799642       |1       |113800287       |113800388       |paired      |
|2       |alt_sort/HG00099.bed.gz |1       |113799516       |113799616       |1       |113800220       |113800321       |paired      |
|3       |alt_sort/HG00100.bed.gz |1       |113799234       |113799333       |1       |113800090       |113800190       |paired      |
|3       |alt_sort/HG00100.bed.gz |1       |113799235       |113799334       |1       |113800139       |113800238       |paired      |

### 2. Filter out reference samples & process the output so that the each line contains pairs of evidence that correspond with one sample

HG00096,113799540,113800187\
HG00097,113799542,113800388\
HG00099,113799516,113800321\
HG00100,113799234,113800190,113799235,113800238,113799318,113800230,113799328,113800353,113799349,113800342,113799356,113800259,113799379,113800269,113799403,113800296,113799467,113800440\
HG00101,113799430,113800209,113799529,113800307,113799553,113800389\

#### Line Segments and Points
<img width="500" alt="Line Segments and Points" src="https://github.com/user-attachments/assets/36d465b8-f474-447c-9b43-e5d659b77554">

#### Fitted Lines
<img width="500" alt="Fitted Lines" src="https://github.com/user-attachments/assets/28b290f1-b4c5-4948-98f3-0132ad815846">

### 3. Calculate Intercepts
#### Intercepts
<img width="500" alt="Intercepts" src="https://github.com/user-attachments/assets/e6681b59-ca80-4598-a71f-88dfb9aa598f">

### 4. Run GMM
Decide if the points are most likely to fit a 1, 2, or 3 mode distribution. Runs the EM algorithm for 30 iterations for each of the distributions and compares the AIC scores for each model.
#### Best Distribution Over All Data Points
<img width="500" alt="GMM Modes" src="https://github.com/user-attachments/assets/e0925a5d-7a4e-4b28-8bc8-ad157b835d8e">

### 5. Assign points to modes
Remove approximately 10% of the data from each mode to reduce the overlap between modes.
#### Left-Right Coordinates of Points for Each Distribution & Ancestry Splits
<img width="500" alt="L-R Points & Ancestry" src="https://github.com/user-attachments/assets/419ee98d-0fe6-4962-9fcf-329faa09abd8">

#### Length of SVs For Each Mode
<img width="500" alt="SV Lengths" src="https://github.com/user-attachments/assets/d50976f6-5989-41f7-85af-8706c194c25a">



TODO:
- assign a data collection method to each sample, filter out 1000 Genomes on GRCh38,1000 Genomes 30x on GRCh38,1000 Genomes phase 3 release