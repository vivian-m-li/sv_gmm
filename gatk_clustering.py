import sys
import uuid
import subprocess
import multiprocessing
import csv
import pandas as pd
from synthetic_tests import generate_data_r
from generate_data import generate_synthetic_sv_data
from timeout import break_after
from typing import Optional


def process_gatk_output(filename: str):
    """Parse GATK SVCluster output VCF to determine number of clusters
    and extract their positions (L) and lengths."""
    Ls = []
    lengths = []

    with open(filename, "r") as f:
        for line in f:
            # skip hearder lines
            if line.startswith("#"):
                continue

            # parse vcf file fields
            fields = line.strip().split("\t")
            if len(fields) < 8:
                continue

            pos = int(fields[1])  # this is L (left breakpoint)
            info = fields[7]

            # parse INFO field to get END2 (right breakpoint)
            info_dict = {}
            for item in info.split(";"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    info_dict[key] = value

            # get the right breakpoint position
            if "END2" in info_dict:
                end2 = int(info_dict["END2"])
                length = abs(end2 - pos)
            else:
                # fallback: try to parse from ALT field
                alt = fields[4]
                # ALT format: N]chr:pos] or N[chr:pos[
                if ":" in alt:
                    end2_str = alt.split(":")[1].rstrip("][")
                    end2 = int(end2_str)
                    length = abs(end2 - pos)
                else:
                    length = 0

            Ls.append(pos)
            lengths.append(length)

    num_clusters = len(Ls)

    return num_clusters, Ls, lengths


def write_csv(
    all_results,
    *,
    fixed_n_samples: Optional[int] = None,
    fixed_svlen: Optional[int] = None,
):
    file = f"synthetic_data/results{'' if fixed_n_samples is None else 'n=' + str(fixed_n_samples)}.csv"
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
            gatk_output_file,
        ) in all_results:
            lengths, Ls = process_gatk_output(gatk_output_file)
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
                    "gmm_model": "gatk_svcluster",
                    "expected_num_modes": len(svs),
                    "expected_lengths": [sv[1] - sv[0] for sv in svs],
                    "expected_Ls": [sv[0] for sv in svs],
                    "num_modes": len(lengths),
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


def gatk_cluster_inner(case, r, svs, weights, n_samples, results):
    """Generates synthetic data and runs the GMM on it. Appends the results to the multiprocessing-managed list to be written to a CSV later."""
    # generates synthetic data and writes to a vcf file
    run_id = uuid.uuid4()
    filename = f"synthetic_data/data/{case}_r{r}_svlen{svs[0][1] - svs[0][0]}_n{n_samples}_{run_id}.vcf"
    generate_synthetic_sv_data(
        1,
        svs,
        n_samples=n_samples,
        p=weights,
        run_gmm=False,
        vcf_filename=filename,
    )

    # run GATK's SVCluster on the generated vcf
    output_file = f"synthetic_data/clustered/{run_id}.vcf"
    subprocess.run(
        ["bash", "gatk_svcluster.sh"] + [filename, output_file],
        capture_output=True,
        text=True,
    )

    results.append([case, r, svs, n_samples, weights, output_file])


@break_after(hours=31, minutes=55)
def gatk_cluster(n_samples: int, svlen: int, test_case: Optional[str] = None):
    """Parallelized synthetic data tests with varying r and (optional) cluster weights. Each set of parameters is repeated 50 times. Run with the bash script run_gatk_clustering.sh to test different sample sizes and sv lengths."""
    # generate synthetic data
    if test_case is None:
        data = []
        for case in ["A", "B", "C", "D"]:
            data.extend(generate_data_r(case, svlen))
    else:
        data = generate_data_r(test_case, svlen)

    with multiprocessing.Manager() as manager:
        cpu_count = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpu_count)
        results = manager.list()
        args = []
        for case, r, svs in data:
            weights = [[1.0 / len(svs) for _ in range(len(svs))]]
            for weight in weights:
                for _ in range(50):
                    args.append((case, r, svs, weight, n_samples, results))

        p.starmap(gatk_cluster_inner, args)
        p.close()
        p.join()

        write_csv(
            results,
            write_new_file=False,
            fixed_n_samples=n_samples,
            fixed_svlen=svlen,
        )


if __name__ == "__main__":
    n_samples = int(sys.argv[1])
    svlen = int(sys.argv[2]) if len(sys.argv) > 2 else 802
    gatk_cluster(n_samples=n_samples, svlen=svlen)
