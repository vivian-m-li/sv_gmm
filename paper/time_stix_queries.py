import os
import time

import pandas as pd

from src.data.query_stix import query_stix
from src.utils.model_helper import get_query_region
from src.utils.helper import stix_output_to_df

if __name__ == "__main__":
    df = pd.read_csv("calibration/sv_subset.csv")
    df = df.sample(n=20)
    for _, row in df.iterrows():
        for q in [0.5, 1.0]:
            query_region = get_query_region(
                f"{row.chr}:{row.start}",
                f"{row.chr}:{row.stop}",
                q,
            )
            filename = f"{query_region.file_name}.txt"
            output_dir = "test_stix_queries"
            file = os.path.join(output_dir, filename)
            if os.path.isfile(file):
                continue

            start = time.time()
            stix_file = query_stix(
                query_region,
                output_dir,
                "/Users/vili4418/sv/stix/bin/stix",
                "/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian/shard",
                "/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian/shard",
                8,
                True,
            )
            reads = stix_output_to_df(stix_file)
            end = time.time()
            print(
                f"Time to download {reads.shape[0]} reads for {reads['sample_id'].nunique()} samples: {(end - start):.2f} seconds. file={stix_file}",
                flush=True,
            )
