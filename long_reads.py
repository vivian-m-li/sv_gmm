import re
import sys
import argparse
import subprocess
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from helper import get_sv_lookup


def parse_long_read_samples():
    file = "long_reads/raw_1kg_ont_vienna_hg38.txt"
    root = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1KG_ONT_VIENNA/hg38"
    df = pd.DataFrame(columns=["sample_id", "cram_file", "indexed_cram_file"])
    with open(file, "r") as f:
        soup = BeautifulSoup(f, "html.parser")
        table = soup.find("table")
        rows = table.find_all("tr")
        for row in rows:
            if ".cram" in row.text:
                columns = row.find_all("td")
                for col in columns:
                    if ".cram" in col.text:
                        pattern = r"([\S]+)\.hg38\.cram(.crai)?"
                        sample_id = re.search(pattern, col.text).group(1)
                        file_name = f"{root}/{col.text}"
                        if ".crai" in col.text:
                            df.loc[
                                df["sample_id"] == sample_id,
                                "indexed_cram_file",
                            ] = file_name
                        else:
                            df.loc[len(df)] = [sample_id, file_name, ""]
    df.to_csv("long_reads/long_read_samples.csv", index=False)


def parse_cigar(cigar_string):
    """Parse CIGAR string into a list of operations and lengths."""
    # Extract digit-character pairs
    cigar_pattern = re.compile(r"(\d+)([MIDNSHP=X])")
    return [
        (int(length), op) for length, op in cigar_pattern.findall(cigar_string)
    ]


def identify_large_indels(cigar_tuples, min_size=50):
    """Identify large insertions and deletions in CIGAR string."""
    large_indels = []
    current_pos = 0

    for length, op in cigar_tuples:
        if op == "D" and length >= min_size:
            large_indels.append(("DEL", current_pos, length))
        elif op == "I" and length >= min_size:
            large_indels.append(("INS", current_pos, length))

        # Update position (only for operations that consume reference)
        if op in "MDN=X":
            current_pos += length

    return large_indels


def calculate_sv_signature(cigar_tuples):
    """Create a simplified signature of the structural variants in the CIGAR."""
    # Count total operations by type
    op_counts = defaultdict(int)
    # Count large operations (potential SVs)
    large_ops = defaultdict(list)

    current_pos = 0
    for length, op in cigar_tuples:
        op_counts[op] += 1

        # Track large indels (potential SVs)
        if op in "DI" and length >= 50:
            large_ops[op].append((current_pos, length))

        # Update position
        if op in "MDN=X":
            current_pos += length

    return {"op_counts": dict(op_counts), "large_indels": dict(large_ops)}


def compare_cigars(cigar1, cigar2, tolerance=10):
    """Compare two CIGAR strings to see if they represent the same SV."""
    # Parse CIGAR strings
    cigar1_tuples = parse_cigar(cigar1)
    cigar2_tuples = parse_cigar(cigar2)

    # Get large indels from both CIGARs
    indels1 = identify_large_indels(cigar1_tuples)
    indels2 = identify_large_indels(cigar2_tuples)

    # If number of large indels is different, they're different SVs
    if len(indels1) != len(indels2):
        return (
            False,
            f"Different number of large indels: {len(indels1)} vs {len(indels2)}",
        )

    # Compare each large indel
    matches = []
    for i, (type1, pos1, len1) in enumerate(indels1):
        if i >= len(indels2):
            break

        type2, pos2, len2 = indels2[i]

        # Check if types match
        if type1 != type2:
            matches.append(False)
            continue

        # Check if positions are close enough
        pos_diff = abs(pos1 - pos2)
        if pos_diff > tolerance:
            matches.append(False)
            continue

        # Check if lengths are similar
        len_diff = abs(len1 - len2)
        if len_diff > tolerance:
            matches.append(False)
            continue

        matches.append(True)

    if all(matches) and matches:
        return True, "SVs match"
    else:
        return False, "SVs differ"


def summarize_cigar(cigar_string):
    """Provide a summary of a CIGAR string, focusing on large operations."""
    cigar_tuples = parse_cigar(cigar_string)

    # Get counts of each operation type
    op_counts = defaultdict(int)
    for length, op in cigar_tuples:
        op_counts[op] += 1

    # Get large indels
    large_indels = identify_large_indels(cigar_tuples)

    return {
        "total_operations": len(cigar_tuples),
        "operation_counts": dict(op_counts),
        "large_indels": large_indels,
    }


def process_sam_record(line):
    """Process a single SAM/BAM record line and extract relevant information."""
    fields = line.strip().split("\t")
    if len(fields) < 6:
        return None

    read_id = fields[0]
    flag = int(fields[1])
    chrom = fields[2]
    pos = int(fields[3])
    mapq = int(fields[4])
    cigar = fields[5]

    return {
        "read_id": read_id,
        "flag": flag,
        "chrom": chrom,
        "pos": pos,
        "mapq": mapq,
        "cigar": cigar,
    }


def compare_long_reads(sv_id: str, sample1: str, sample2: str, tolerance: int):
    # get sv region
    sv_lookup = get_sv_lookup()
    row = sv_lookup[sv_lookup["id"] == sv_id]
    start = row["start"].values[0] - tolerance
    stop = row["stop"].values[0] + tolerance
    region = f"{row["chr"].values[0]}:{start}-{stop}"

    # check both samples have long read files
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")
    sample_rows = long_read_samples[
        long_read_samples["sample_id"].isin([sample1, sample2])
    ]
    if len(sample_rows) < 2:
        raise ValueError(
            "One or both samples does not have a corresponding long read file."
        )

    output_files = []
    for sample_id in (sample1, sample2):
        output_file = f"long_reads/reads/{sv_id}-{sample_id}.txt"
        cram_file = sample_rows[sample_rows["sample_id"] == sample_id][
            "cram_file"
        ].values[0]
        subprocess.run(
            ["bash", "get_cigar.sh"] + [cram_file, region, output_file],
            capture_output=True,
            text=True,
        )
        output_files.append(output_file)

    # compare outputs

    # summary = summarize_cigar(cigar)

    # print(f"CIGAR Summary:")
    # print(f"  Total operations: {summary['total_operations']}")
    # print(f"  Operation counts:")
    # for op, count in summary["operation_counts"].items():
    #     print(f"    {op}: {count}")

    # print(f"  Large indels (>= {args.min_sv_size} bp):")
    # for sv_type, pos, length in summary["large_indels"]:
    #     print(f"    {sv_type} at relative position {pos} with length {length}")

    # return

    # # Analyze from SAM/BAM files (simplified, would typically use pysam)
    # # This is just a demonstration of the concept
    # if args.file1 and args.file2 and args.region:
    #     print(
    #         f"Comparing CIGAR strings for region {args.region} between samples..."
    #     )
    #     print(
    #         "This would typically use pysam to extract records from BAM/CRAM files"
    #     )
    #     # Implementation would depend on having pysam to read BAM/CRAM files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare between two CIGAR strings to identify differences in SV deletions"
    )
    parser.add_argument("-s1", type=str, help="First BAM/CRAM file")
    parser.add_argument("-s2", type=str, help="Second BAM/CRAM file")
    parser.add_argument(
        "-id", type=str, help="SV ID both samples originate from"
    )
    parser.add_argument(
        "-t",
        type=int,
        default=100,
        help="Tolerance",
    )

    args = parser.parse_args()
    compare_long_reads(
        args.id,
        args.s1,
        args.s2,
        args.t,
    )
