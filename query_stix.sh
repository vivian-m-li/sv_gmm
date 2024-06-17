#!/bin/bash

l=$1
r=$2
SV_DB="/scratch/Shares/layer/workspace/vivian_stix/files"

/Users/vili4418/sv/stix/bin/stix -i "$SV_DB/alt_sort_b" -d "$SV_DB/1kg.ped.db" -s 500 -t DEL -l "$l" -r "$r" >> stix_output.txt
