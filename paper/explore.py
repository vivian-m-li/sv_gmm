import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.model.gmm_trial import process_data
from src.utils.helper import stix_output_to_df, get_sample_ids
from src.utils.model_helper import giggle_format, get_insert_size_lookup


def check_kmeans_centers():
    filename = "output/results/1_mode_svs.csv"
    svs = pd.read_csv(filename)

    sample_ids = get_sample_ids("data/1kg/sample_ids.txt")
    insert_size_lookup = get_insert_size_lookup(
        "data/1kg", "insert_sizes.csv", 450, sample_ids
    )

    shifts = pd.DataFrame(columns=["id", "L_shift", "R_shift", "len_shift", "n_samples"])
    for i, row in svs.iterrows():
        chr = str(row["chr"])
        L = row["start"]
        R = row["stop"]

        stix_output_file = (
            f"{giggle_format(chr, L)}_{giggle_format(chr, R)}.txt"
        )
        reads_file = f"output/stix_output/{stix_output_file}"
        reads = stix_output_to_df(reads_file)

        x, _ = process_data(
            reads,
            L=row["start"],
            R=row["stop"],
            insert_size_lookup=insert_size_lookup,
        )

        kmeans = KMeans(n_clusters=1, init="k-means++")
        kmeans.fit(x)
        mu = kmeans.cluster_centers_  # (length, L)

        L_shift = mu[0][1]
        cluster_L = L + L_shift
        len_shift = mu[0][0] + (R - L)
        R_shift = (cluster_L + len_shift) - R
        shifts.loc[i] = [row["id"], L_shift, R_shift, len_shift, reads["sample_id"].nunique()]
        print(f"Processed {i}/{len(svs)} SVs...", end="\r", flush=True)

    shifts.to_csv("output/results/1_mode_shifts.csv", index=False)

if __name__ == "__main__":
    check_kmeans_centers()
