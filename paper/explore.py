import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.model.gmm_trial import process_data
from src.utils.helper import stix_output_to_df, get_sample_ids
from src.utils.model_helper import giggle_format, get_insert_size_lookup


def check_kmeans_centers():
    filename = "output/results/1_mode_svs.csv"
    svs = pd.read_csv(filename)

    L_shifts = []
    R_shifts = []
    for _, row in svs.iterrows():
        sample_ids = get_sample_ids("data/1kg/sample_ids.txt")
        insert_size_lookup = get_insert_size_lookup(
            "data/1kg", "insert_sizes.csv", 450, sample_ids
        )

        chr = str(row["chr"])
        L = row["start"]
        R = row["stop"]

        stix_output_file = (
            f"{giggle_format(chr, L)}_{giggle_format(chr, R)}.txt"
        )
        reads_file = f"output/stix_output/{stix_output_file}"
        reads = stix_output_to_df(reads_file)

        x = process_data(
            reads,
            L=row["start"],
            R=row["stop"],
            insert_size_lookup=insert_size_lookup,
        )

        kmeans = KMeans(n_clusters=1, init="k-means++")
        kmeans.fit(x)
        mu = kmeans.cluster_centers_  # (length, L)

        L_shift = mu[0][1]
        R_shift = (mu[0][0] + mu[0][1]) - R
        L_shifts.append(L_shift)
        R_shifts.append(R_shift)

    L_shifts = np.array(L_shifts)
    R_shifts = np.array(R_shifts)

    print(f"Average L shift={np.mean(L_shifts)}, stdev={np.std(L_shifts)}")
    print(f"Average R shift={np.mean(R_shifts)}, stdev={np.std(R_shifts)}")


if __name__ == "__main__":
    check_kmeans_centers()
