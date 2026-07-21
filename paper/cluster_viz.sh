# this script has not been working so I'm running it manually on the command line

sv_ids=("HGSV_15262" "HGSV_143868" "HGSV_39753" "HGSV_204881" "HGSV_226693" "HGSV_5515" "HGSV_218106" "HGSV_89" "HGSV_54541" "HGSV_149774" "HGSV_245658" "HGSV_220750" "HGSV_161412")
for sv_id in "${sv_ids[@]}"; do
	python -m scripts.split_one -p -id "$sv_id"
done
