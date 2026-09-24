#!/bin/bash
input_file=$1
output_file=$2
ploidy_table=$3
reference_file=$4
gatk_alg=$5

GATK_BUILD="/Users/vili4418/sv/gatk-4.6.2.0/gatk"

# write temporary ploidy table and reference file to avoid i/o issues
run_id=$RANDOM
ploidy_table_temp="/scratch/Users/vili4418/synthetic_data/ploidy_table_${run_id}.tsv"
reference_file_temp="/scratch/Users/vili4418/synthetic_data/reference_${run_id}.fasta"

cp $ploidy_table $ploidy_table_temp
cp $reference_file $reference_file_temp

# copy reference index files as well
cp $reference_file.fai $reference_file_temp.fai
cp ${reference_file%.fasta}.dict ${reference_file_temp%.fasta}.dict
cp ${reference_file%.fasta}.contig_list ${reference_file_temp%.fasta}.contig_list

# use gatk to cluster the SVs 
$GATK_BUILD SVCluster \
  -V $input_file \
  -O $output_file \
  --ploidy-table $ploidy_table_temp \
  -R $reference_file_temp \
  --algorithm $gatk_alg

# remove all temporary files
rm $ploidy_table_temp
rm $reference_file_temp
rm $reference_file_temp.fai
rm ${reference_file_temp%.fasta}.dict
rm ${reference_file_temp%.fasta}.contig_list