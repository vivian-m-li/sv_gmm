#!/bin/bash
dout=$(pwd)
sv_id=$1
bams="${dout}/long_reads/bam_files/${sv_id}"
# sv region
chr=$2
left=$3
right=$4
zoom_mult=1
cd $bams
len=$(python -c "print(${right}-${left})")
zoom=$(python -c "print(int(${len}*${zoom_mult}))")

samplot plot \
    -b $(ls *.bam) \
    -c ${chr} -s ${left} -e ${right} \
    -t DEL --zoom ${zoom} \
    -o ${dout}/long_reads/samplot_viz/${sv_id}.png
