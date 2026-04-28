#!/bin/bash

# # Ensure your python version is loaded (in this case using module load)
# module purge
# module load python/3.11.3

# # Activate your python environment (assuming in the same folder as this script)
source venv/bin/activate
which python
python --version

pip install -r min_requirements.txt
pip install -e .

# test split_one
python -m scripts.split_one \
--config data/assets/default_config.toml \
-l 3:173522965 \
-r 3:173524108 \
-p \
-d

#-p \ # true to plot the length and L-coordinate of each sample
#-d \ # keep rerunning algorithm until >= 80% confident in outcome

# test split_all
python -m scripts.split_all \
--config data/assets/default_config.toml