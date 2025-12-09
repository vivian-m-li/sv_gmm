#!/bin/bash

# Ensure your python version is loaded (in this case using module load)
module purge
module load python/3.11.3

# Activate your python environment (assuming in the same folder as this script)
source venv/bin/activate
which python
python --version
#pysam 0.23.1



# IF DOING FROM a vcf and already have a STIX output
#python query_sv.py 
# -l left position of SV
# -r right position of SV
# -ref reference genome being used
# --input_dir list of files required can be found in README
# --output_dir 

#python query_sv.py \
#-l "3:173522965" \
#-r "3:173524108" \
#-ref "grch38" \
#--input_dir "./assets/" \
#--output_dir "./test_out/" \
#--sv_lookup "1kg.subset.vcf.gz" \
#--insert_size_file "insert_sizes.csv"


# IF DOING FROM a previously filtered SV set
python query_sv.py \
-l "3:173522965" \
-r "3:173524108" \
-ref "grch38" \
--input_dir "./assets/" \
--output_dir "./test_out/" \
--sv_lookup "test_deletions.csv" \
--insert_size_file "insert_sizes.csv" \
-p \
-d



#-p \ # true to plot the length and L-coordinate of each sample
#-d \ # keep rerunning algorithm until >= 80% confident in outcome
