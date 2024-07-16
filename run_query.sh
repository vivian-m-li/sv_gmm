#!/bin/bash
#SBATCH --job-name=run_sims
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=vili4418@colorado.edu
#SBATCH -p highmem
#SBATCH -N 1
#SBATCH -c 192
#SBATCH --mem=1500gb                # Memory limit
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=/Users/vili4418/sv/eofiles/%x_%j.out
#SBATCH --error=/Users/vili4418/sv/eofiles/%x_%j.err

source $HOME/.venv/bin/activate
python3 $HOME/sv/sv_gmm/query_gnomad.py
deactivate
