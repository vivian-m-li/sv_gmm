import os
import time
import datetime
import csv
import requests
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from gmm_types import CHR_LENGTHS
from query_sv import (
    giggle_format,
    query_stix,
    load_squiggle_data,
    PROCESSED_FILE_DIR,
)
from process_data import get_intercepts
from typing import Dict

# for human genome assembly GRCh37, https://www.ncbi.nlm.nih.gov/grc/human/data?asm=GRCh37
URL = "https://gnomad.broadinstitute.org/api/"
FILE_DIR = "gnomad_svs"


def query_gnomad(chr: str, start: int, stop: int):
    payload = {
        "operationName": "StructuralVariantsInRegion",
        "query": "\n    query StructuralVariantsInRegion($datasetId: StructuralVariantDatasetId!, $chrom: String!, $start: Int!, $stop: Int!, $referenceGenome: ReferenceGenomeId!) {\n      region(chrom: $chrom, start: $start, stop: $stop, reference_genome: $referenceGenome) {\n        structural_variants(dataset: $datasetId) {\n          ac\n          ac_hemi\n          ac_hom\n          an\n          af\n          chrom\n          chrom2\n          end\n          end2\n          consequence\n          filters\n          length\n          pos\n          pos2\n          type\n          variant_id\n        }\n      }\n    }\n  ",
        "variables": {
            "chrom": chr,
            "datasetId": "gnomad_sv_r2_1",
            "referenceGenome": "GRCh37",
            "start": start,
            "stop": stop,
        },
    }

    response = requests.post(URL, json=payload)
    if response.status_code != 200:
        print(f"Failed to retrieve data for chr {chr}.")
        return

    sv_data = response.json()["data"]["region"]["structural_variants"]
    fields = list(sv_data[0].keys())
    dels = [
        sv for sv in sv_data if sv["type"] == "DEL" and len(sv["filters"]) == 0
    ]
    dels = sorted(
        dels, key=lambda x: x["af"], reverse=True
    )  # sort by allele frequency

    filename = f"{FILE_DIR}/{chr}.csv"
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in dels:
            writer.writerow(row)


def get_all_sv():
    for chr, length in CHR_LENGTHS.items():
        query_gnomad(chr, 1, length)
        # add a short delay so the POST requests aren't directly back to back
        time.sleep(5)


def query_gnomad_stix():
    for chr in CHR_LENGTHS.keys():
        start = time.time()
        df = pd.read_csv(f"{FILE_DIR}/{chr}.csv")
        with multiprocessing.Manager():
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            args = []
            for _, row in df.iterrows():
                args.append(
                    (
                        giggle_format(chr, row.pos),
                        giggle_format(chr, row.end),
                        False,
                    )
                )

            p.starmap(query_stix, args)
            p.close()
            p.join()
        end = time.time()
        print(
            f"Time to query chromosome {chr}: {datetime.timedelta(seconds=end - start)}"
        )


def get_num_samples(row_index: int, row, chr: str, lookup: Dict[int, int]):
    start = giggle_format(chr, row.pos)
    end = giggle_format(chr, row.end)
    squiggle_data = load_squiggle_data(
        f"{PROCESSED_FILE_DIR}/{start}_{end}.csv"
    )
    if len(squiggle_data) > 0:
        intercepts, _ = get_intercepts(
            squiggle_data, file_name=None, L=row.pos, R=row.end
        )
        lookup[row_index] = len(intercepts)


def plot_af_num_samples(chr: str, df: pd.DataFrame):
    plt.figure()
    plt.scatter(df["af"], df["num_samples"])
    plt.title(f"Chromosome {chr}")
    plt.xlabel("Allele Frequency")
    plt.ylabel("Number of Samples in STIX")
    plt.show()


def get_num_sv():
    for chr in CHR_LENGTHS.keys():
        filename = f"{FILE_DIR}/{chr}.csv"
        df = pd.read_csv(filename)
        if "num_samples" not in df.columns:
            df["num_samples"] = 0
            with multiprocessing.Manager() as manager:
                p = multiprocessing.Pool(multiprocessing.cpu_count())
                lookup = manager.dict()
                args = []
                for i, row in df.iterrows():
                    args.append((i, row, chr, lookup))

                p.starmap(get_num_samples, args)
                p.close()
                p.join()

                for row_index, num_samples in lookup.items():
                    df.loc[row_index, "num_samples"] = num_samples
            df.to_csv(filename, index=False)

        plot_af_num_samples(chr, df)


def main():
    if not os.path.exists(FILE_DIR):
        os.mkdir(FILE_DIR)
        get_all_sv()
        query_gnomad_stix()
    get_num_sv()


if __name__ == "__main__":
    main()
