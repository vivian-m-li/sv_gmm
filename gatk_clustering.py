import uuid
import subprocess
import multiprocessing
import csv
import pysam
import argparse
import pandas as pd
from synthetic_tests import generate_data_r
from generate_data import generate_synthetic_sv_data
from timeout import break_after
from typing import Optional

PLOIDY_TABLE = (
    "/Users/vili4418/sv/sv_gmm/synthetic_data/generated_files/ploidy_table.tsv"
)
REFERENCE_FILE = (
    "/Users/vili4418/sv/sv_gmm/synthetic_data/generated_files/reference.fasta"
)


def process_gatk_output(filename: str):
    """Parse GATK SVCluster output VCF to determine number of clusters
    and extract their positions (L) and lengths."""
    Ls = []
    lengths = []
    vcf = pysam.VariantFile(filename)
    for record in vcf.fetch():
        pos = record.pos  # left breakpoint
        info = record.info

        # skip non-deletion SVs
        svtype = info.get("SVTYPE", None)
        if svtype != "DEL":
            continue

        # END marks the right breakpoint
        if "END" in info:
            end = info["END"]
        else:
            # fallback: attempt to parse ALT
            end = pos
            if record.alts and ":" in record.alts[0]:
                try:
                    end_str = record.alts[0].split(":")[1].rstrip("><")
                    end = int(end_str)
                except ValueError:
                    pass

        Ls.append(pos)
        lengths.append(abs(end - pos))

    num_clusters = len(Ls)
    return num_clusters, Ls, lengths


def write_csv(
    all_results,
    *,
    fixed_n_samples: Optional[int] = None,
    fixed_svlen: Optional[int] = None,
):
    file = f"synthetic_data/results{'' if fixed_n_samples is None else 'n=' + str(fixed_n_samples)}.csv"
    # append to existing file generated in synthetic_tests.py
    with open(
        file,
        mode="a",
        newline="",
    ) as out:
        fieldnames = [
            "case",
            "r",
            "r2",
            "gmm_model",
            "expected_num_modes",
            "expected_lengths",
            "expected_Ls",
            "num_modes",
            "lengths",
            "Ls",
            "num_samples",
            "svlen",
            "weights",
        ]
        csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
        for (
            case,
            rs,
            svs,
            n_samples,
            weights,
            gatk_alg,
            gatk_output_file,
        ) in all_results:
            try:
                n_clusters, lengths, Ls = process_gatk_output(gatk_output_file)
            except FileNotFoundError:
                print(
                    f"File not found: {gatk_output_file}, case={case}, rs={rs}, svs={svs}, n_samples={n_samples}"
                )
                continue
            if type(rs) is tuple:
                r, r2 = rs
            else:
                r = rs
                r2 = None
            csv_writer.writerow(
                {
                    "case": case,
                    "r": r,
                    "r2": r2,
                    "gmm_model": f"gatk_{gatk_alg}",
                    "expected_num_modes": len(svs),
                    "expected_lengths": [sv[1] - sv[0] for sv in svs],
                    "expected_Ls": [sv[0] for sv in svs],
                    "num_modes": n_clusters,
                    "lengths": lengths,
                    "Ls": Ls,
                    "num_samples": n_samples,
                    "svlen": fixed_svlen,
                    "weights": weights,
                }
            )

    df = pd.read_csv(file)
    df = df.sort_values(by=["case", "gmm_model", "r", "r2"])
    df.to_csv(file, index=False)


def gatk_cluster_inner(case, r, svs, weights, n_samples, gatk_alg, results):
    """Generates synthetic data and runs the GMM on it. Appends the results to the multiprocessing-managed list to be written to a CSV later. File I/O is done on scratch."""
    # generates synthetic data and writes to a vcf file
    run_id = uuid.uuid4()
    r_str = ",".join([str(x) for x in r]) if type(r) is tuple else str(r)
    filename = f"/scratch/Users/vili4418/synthetic_data/data/{case}_r{r_str}_svlen{str(svs[0][1] - svs[0][0])}_n{n_samples}_{run_id}.vcf"
    generate_synthetic_sv_data(
        1,
        svs,
        n_samples=n_samples,
        p=weights,
        run_gmm=False,
        vcf_filename=filename,
    )

    # run GATK's SVCluster on the generated vcf
    output_file = (
        f"/scratch/Users/vili4418/synthetic_data/clustered/{run_id}.vcf"
    )
    result = subprocess.run(  # noqa: F841
        ["bash", "gatk_svcluster.sh"]
        + [  # noqa: W503
            filename,
            output_file,
            PLOIDY_TABLE,
            REFERENCE_FILE,
            gatk_alg,
        ],
        capture_output=True,
        text=True,
    )
    # for debugging gatk
    # print(f"GATK SVCluster output for case {case}, r={r}, svs={svs}, n_samples={n_samples}, filename={filename}")
    # print(result.stdout)
    # print(result.stderr)

    results.append([case, r, svs, n_samples, weights, gatk_alg, output_file])


@break_after(hours=71, minutes=55)
def gatk_cluster(
    n_samples: int,
    svlen: int,
    test_case: Optional[str] = None,
    gatk_alg: Optional[str] = None,
):
    """Parallelized synthetic data tests with varying r and (optional) cluster weights. Each set of parameters is repeated 50 times. Run with the bash script run_gatk_clustering.sh to test different sample sizes and sv lengths."""
    # generate synthetic data
    if test_case is None:
        data = []
        for case in ["A", "B", "C", "D"]:
            data.extend(generate_data_r(case, svlen))
    else:
        data = generate_data_r(test_case, svlen)

    with multiprocessing.Manager() as manager:
        # assign more cores to the job than the pool size
        p = multiprocessing.Pool(10)
        results = manager.list()
        args = []
        for case, r, svs in data:
            weights = [[1.0 / len(svs) for _ in range(len(svs))]]
            for weight in weights:
                for _ in range(10):
                    args.append(
                        (case, r, svs, weight, n_samples, gatk_alg, results)
                    )

        # add a chunk size; each worker will be given 10 tasks to run
        p.starmap(gatk_cluster_inner, args, chunksize=10)
        p.close()
        p.join()

        write_csv(
            results,
            fixed_n_samples=n_samples,
            fixed_svlen=svlen,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Runs GATK SVCluster on synthetic data."
    )
    parser.add_argument(
        "-n",
        type=int,
        help="Number of samples to generate",
        default=30,
        nargs="?",
    )
    parser.add_argument(
        "-l",
        type=int,
        help="Length of the default SV generated",
        default=802,
        nargs="?",
    )
    parser.add_argument(
        "-c",
        type=str,
        help="Test case to run (A, B, C, D)",
        default=None,
        nargs="?",
    )

    # Default value: SINGLE_LINKAGE. Possible values: {DEFRAGMENT_CNV, SINGLE_LINKAGE, MAX_CLIQUE}
    parser.add_argument(
        "-g",
        type=str,
        help="GATK SVCluster algorithm to use",
        default="SINGLE_LINKAGE",
        nargs="?",
    )
    args = parser.parse_args()
    gatk_cluster(
        n_samples=args.n,
        svlen=args.l,
        test_case=args.c,
        gatk_alg=args.g,
    )


if __name__ == "__main__":
    main()
