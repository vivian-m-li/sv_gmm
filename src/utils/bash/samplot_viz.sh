#!/bin/bash
dout=$(pwd)
sv_id=$1
# sv region
chr=$2
left=$3
right=$4

in_dir=$5
out_dir=$6
bams="${dout}/${in_dir}/${sv_id}"

zoom_mult=1
cd $bams
len=$(python -c "print(${right}-${left})")
zoom=$(python -c "print(int(${len}*${zoom_mult}))")

samplot plot \
    -b $(ls *.bam) \
    -c ${chr} -s ${left} -e ${right} \
    -t DEL --zoom ${zoom} \
    -o ${dout}/${out_dir}/${sv_id}.png
