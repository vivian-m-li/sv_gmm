#!/bin/bash
dout=$(pwd)
sv_id=$1
bams="${dout}/long_reads/bam_files_subset/${sv_id}"
# sv region
chr=$2
left=$3
right=$4
zoom_mult=1
cd $bams
len=$(python -c "print(${right}-${left})")
zoom=$(python -c "print(int(${len}*${zoom_mult}))")

echo "${dout}/plots/samplot_${sv_id}.pdf"
samplot plot \
    -b $(ls *.bam) \
    -c ${chr} -s ${left} -e ${right} \
    -t DEL --zoom ${zoom} \
    -o ${dout}/plots/samplot_${sv_id}.pdf
