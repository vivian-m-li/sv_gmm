import subprocess

from src.utils.helper import get_sv_lookup


def copy_stix_output_from_fiji(sv_id: str):
    lookup = get_sv_lookup()
    sv_row = lookup[lookup["id"] == sv_id]
    chr = str(sv_row["chr"].values[0])
    start = str(int(sv_row["start"].values[0]))
    stop = str(int(sv_row["stop"].values[0]))
    subprocess.run(
        [
            "scp",
            f"vili4418@fiji.colorado.edu:/Users/vili4418/sv/sv_gmm/stix_output/{chr}:{start}_{chr}:{stop}.txt",
            "assets/stix_output/",
        ]
    )
