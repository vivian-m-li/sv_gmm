#!/bin/bash
sample_id=$1
input_file=$2

output_file="1000genomes/mapped_files/"$sample_id".tsv"
input_file="ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/"$input_file

wget -O $output_file $input_file