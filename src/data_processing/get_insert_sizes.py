import os
import subprocess
import time
import pandas as pd
from helper import get_sample_ids
from timeout import break_after

FILE_DIR = "/Users/vili4418/sv/sv_gmm/1kgp/insert_size_files"
TEMP_DIR = "/scratch/Users/vili4418/insert_size_files"


def concat_mean_insert_sizes():
    files = os.listdir(f"{FILE_DIR}")
    df = pd.DataFrame(columns=["sample_id", "mean_insert_size"])
    for i, file in enumerate(files):
        sample_id = file.strip(".txt")
        with open(f"{FILE_DIR}/{file}") as f:
            mean_insert_size = int(float(f.readlines()[0].strip("\n")))
            df.loc[i] = [sample_id, mean_insert_size]
    df.to_csv("1kgp/insert_sizes.csv", index=False)


@break_after(hours=5, minutes=55)
def get_insert_sizes():
    sample_ids = get_sample_ids()
    df = pd.read_csv("1kgp/high_cov_grch38_samples.tsv", sep="\t")
    processed_samples = set([f.strip(".txt") for f in os.listdir(FILE_DIR)])

    for sample_id in sample_ids:
        if sample_id in processed_samples:
            continue
        start = time.time()
        row = df[df["Sample"] == sample_id]
        url = row["url"].values[0]

        subprocess.run(
            ["bash", "get_insert_size.sh"] + [sample_id, url],
            capture_output=True,
            text=True,
        )

        end = time.time()
        print(f"{sample_id} - time to get mean insert size={end - start}")

    concat_mean_insert_sizes()


if __name__ == "__main__":
    get_insert_sizes()
