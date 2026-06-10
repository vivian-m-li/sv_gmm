import argparse
import os
import subprocess
import time

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.helper import get_sample_ids
from src.utils.timeout import break_after


def concat_mean_insert_sizes(output_dir: str, filename: str):
    files = os.listdir(output_dir)
    df = pd.DataFrame(
        columns=["sample_id", "mean_insert_size", "insert_size_sd"]
    )
    for i, file in enumerate(files):
        sample_id = file.strip(".txt")
        with open(os.path.join(output_dir, file)) as f:
            mean_insert_size = int(float(f.readlines()[0].strip("\n")))
            insert_size_sd = int(float(f.readlines()[1].strip("\n")))
            df.loc[i] = [sample_id, mean_insert_size, insert_size_sd]
    df.to_csv(filename, index=False)


@break_after(hours=5, minutes=55)
def get_insert_sizes(
    cfg,
    samples_file: str,
):
    output_dir = cfg["paths"]["input_dir"]
    insert_files_dir = os.path.join(output_dir, "insert_size_files")
    os.makedirs(insert_files_dir, exist_ok=True)

    temp_file_dir = cfg["paths"]["intermediate_output_dir"]
    os.makedirs(temp_file_dir, exist_ok=True)

    df = pd.read_csv(
        samples_file, sep="\t" if samples_file.endswith(".tsv") else ","
    )
    processed_samples = set(
        [f.strip(".txt") for f in os.listdir(insert_files_dir)]
    )

    sample_ids = get_sample_ids(
        os.path.join(output_dir, cfg["input_files"]["sample_id_file"])
    )
    if sample_ids is None:
        sample_ids = df["Sample"].unique()

    for sample_id in sample_ids:
        if sample_id in processed_samples:
            continue
        start = time.time()
        row = df[df["Sample"] == sample_id]
        url = row["url"].values[0]

        subprocess.run(
            ["bash", "src/data/bash/get_insert_size.sh"]
            + [
                sample_id,
                url,
                temp_file_dir,
                insert_files_dir,
                cfg["samtools"]["bin"],
            ],
            capture_output=True,
            text=True,
        )

        end = time.time()
        print(f"{sample_id} - time to get mean insert size={end - start}", flush=True)

    concat_mean_insert_sizes(
        insert_files_dir,
        os.path.join(output_dir, cfg["input_files"]["insert_size_file"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get the mean insert size and standard deviation for each sample."
    )
    parser.add_argument(
        "--samples",
        type=str,
        help="Path to the file containing sample information. File must contain columns 'Sample' and 'url'",
    )
    args = parser.parse_args()

    cfg = load_config()
    get_insert_sizes(cfg, samples_file=args.samples)
