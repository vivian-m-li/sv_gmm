import os
import subprocess

from src.utils.types import StixQueryRegion


def query_stix(
    query_region: StixQueryRegion,
    output_dir: str,
    stix_bin: str,
    index_path: str,
    database_path: str,
    num_shards: int,
    parallel: bool = False,
):
    """
    Runs a bash query file to query STIX for all the read (paired-end and split
    data within the defined coordinate space.
    num_shards: number of shards the stix index and database are split into
    - this must be defined properly to pull all of the relevant data
    """
    partial_outputs_dir = os.path.join(output_dir, "partial_outputs")
    if num_shards > 1:
        if not parallel and not os.path.exists(partial_outputs_dir):
            os.mkdir(partial_outputs_dir)
        stix_output_file = os.path.join(
            partial_outputs_dir, query_region.file_name
        )
    else:
        stix_output_file = os.path.join(output_dir, query_region.file_name)

    stix_path = "/".join(index_path.split("/")[:-1])
    l_query = (
        f"{query_region.chr}:{query_region.left_start}-{query_region.left_stop}"
    )
    r_query = f"{query_region.chr}:{query_region.right_start}-{query_region.right_stop}"
    subprocess.run(
        ["bash", "bash/query_stix.sh"]
        + [  # noqa503
            l_query,
            r_query,
            stix_path,
            index_path,
            database_path,
            str(num_shards),
            stix_output_file,
            stix_bin,
        ],
        capture_output=True,
        text=True,
    )

    # the query output from each shard was written to a different file, and we want to write them to the same file
    if num_shards > 1:
        output_file = os.path.join(output_dir, f"{query_region.file_name}.txt")
        try:
            with open(output_file, "w") as out_file:
                for i in range(num_shards):
                    temp_file = os.path.join(
                        partial_outputs_dir, f"{query_region.file_name}_{i}.txt"
                    )
                    with open(temp_file, "r") as partial_file:
                        # write the partial file to the main file
                        out_file.write(partial_file.read())

                    # remove partial file
                    os.remove(temp_file)
        except FileNotFoundError:
            os.remove(output_file)
            raise Exception(
                f"There was an issue querying STIX for the region {l_query} to {r_query}"
            )
        if not parallel:
            os.rmdir(partial_outputs_dir)
    else:
        output_file = f"{stix_output_file}.txt"

    return output_file
