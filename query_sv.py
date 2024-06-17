import sys
import subprocess
import argparse

STIX_SCRIPT = "./query_stix.sh"


def parse_input(input: str) -> str:
    parts = input.split(":")
    if len(parts) != 2:
        print("Input string must be in the format 'chromosome:position'")
        sys.exit(1)

    chromosome = int(parts[0])
    position = int(parts[1])
    return f"{chromosome}:{position}-{position}"


def query_stix(l: str, r: str):
    result = subprocess.run(
        ["bash", STIX_SCRIPT] + [l, r], capture_output=True, text=True
    )


def main():
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument(
        "-l",
        type=str,
        help="Left position of the structural variant, format=chromosome:position",
    )
    parser.add_argument(
        "-r",
        type=str,
        help="Right position of the structural variant, format=chromosome:position",
    )

    args = parser.parse_args()
    l = parse_input(args.l)
    r = parse_input(args.r)
    query_stix(l, r)


if __name__ == "__main__":
    main()
