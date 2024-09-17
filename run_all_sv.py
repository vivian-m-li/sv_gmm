import pandas as pd
import multiprocessing
from dataclasses import fields, asdict
from query_sv import *
from viz import *
from gmm_types import *

FILE_DIR = "1000genomes"
FILE_NAME = "sv_stats.csv"


def get_population_size():
    df = pd.read_csv("1000genomes/deletions_df.csv", nrows=0, low_memory=False)
    return df.shape[1] - 12


def remove_missing_samples(squiggle_data: Dict[str, np.ndarray[float]]):
    df = pd.read_csv(
        "1000genomes/deletions_df.csv", nrows=0, low_memory=False
    )  # 2504 samples
    missing_keys = set(squiggle_data.keys()) - set(df.columns[11:-1])
    for key in missing_keys:
        squiggle_data.pop(key, None)


def get_sv_stats(row, svs: List[SVStatGMM]) -> None:
    sv_stat = SVStatGMM(
        id=row.id,
        chr=row.chr,
        start=row.start,
        stop=row.stop,
        svlen=row.svlen,
        ref=row.ref,
        alt=row.alt,
        qual=row.qual,
        # filter=row.filter,
        af=row.af,
        # info=row.info,
        num_samples=0,
        num_pruned=0,
        num_reference=0,
        svlen_post=0,
        num_modes=0,
        num_iterations=0,
        overlap_between_modes=False,
        modes=[],
    )

    start = giggle_format(str(row.chr), row.start)
    end = giggle_format(str(row.chr), row.stop)

    squiggle_data = query_stix(start, end, False, filter_reference=False)
    sv_stat.num_samples = len(squiggle_data)

    reference_samples = get_reference_samples(
        squiggle_data, row.chr, row.start, row.stop
    )
    sv_stat.num_reference = len(reference_samples)
    for ref in reference_samples:
        squiggle_data.pop(ref, None)

    remove_missing_samples(squiggle_data)

    if len(squiggle_data) == 0:
        svs.append(sv_stat)
        return

    gmm, evidence_by_mode = run_viz_gmm(
        squiggle_data,
        file_name=None,
        chr=row.chr,
        L=row.start,
        R=row.stop,
        plot=False,
        plot_bokeh=False,
    )
    num_samples = sum([len(mode) for mode in gmm.x_by_mode])
    sv_stat.num_pruned = sum([pruned for pruned in gmm.num_pruned]) + len(gmm.outliers)
    sv_stat.num_modes = gmm.num_modes
    sv_stat.num_iterations = gmm.num_iterations

    all_svlen = get_svlen(evidence_by_mode)
    sv_stat.svlen_post = int(
        np.mean(
            [sv.length - 450 for lst in all_svlen for sv in lst]
        )  # 450 is the length of the read
    )  # TODO: how do we calculate this? It can be negative

    population_size = get_population_size()
    mode_coords = []
    for i, mode in enumerate(evidence_by_mode):
        sample_ids = [e.sample.id for e in mode]
        num_samples = len(sample_ids)
        num_homozygous = len([e.sample for e in mode if e.sample.allele == "(1, 1)"])

        lengths = []
        starts = []
        ends = []
        min_start = float("inf")
        max_end = float("-inf")
        for evidence in mode:
            max_l = max([paired_end[0] for paired_end in evidence.paired_ends])
            min_r = min([paired_end[1] for paired_end in evidence.paired_ends])
            lengths.append(min_r - max_l - 450)  # 450 is the length of the read
            starts.append(max_l)
            ends.append(min_r)
            min_start = min(min_start, max_l)
            max_end = max(max_end, min_r)
        mode_coords.append((min_start, max_end))

        mode_stat = ModeStat(
            length=np.mean(lengths),
            length_sd=np.std(lengths),
            start=int(np.mean(starts)),
            start_sd=np.std(starts),
            end=int(np.mean(ends)),
            end_sd=np.std(ends),
            num_samples=num_samples,
            num_heterozygous=num_samples - num_homozygous,
            num_homozygous=num_homozygous,
            sample_ids=sample_ids,
            num_pruned=gmm.num_pruned[i],
            af=num_samples / population_size,
        )
        sv_stat.modes.append(mode_stat)

    if gmm.num_modes > 1:
        for i in range(len(evidence_by_mode) - 1):
            if mode_coords[i][1] > mode_coords[i + 1][0]:
                sv_stat.overlap_between_modes = True

    svs.append(sv_stat)


def dataclass_to_columns(dataclass_type):
    return [field.name for field in SVStatGMM(dataclass_type)]


def create_sv_stats_file():
    df_fields = [field.name for field in fields(SVStatGMM)]
    df = pd.DataFrame(columns=df_fields)
    return df


def run_all_sv(
    *,
    query_chr: Optional[str] = None,
    subset: Optional[List[Tuple[str, int, int]]] = None,
):
    deletions_df = pd.read_csv(f"{FILE_DIR}/deletions_df.csv", low_memory=False)
    sv_stats_df = create_sv_stats_file()
    if subset is not None:
        sv_stats = []
        for chr, start, stop in subset:
            row = deletions_df[
                (deletions_df["chr"] == chr)
                & (deletions_df["start"] == start)
                & (deletions_df["stop"] == stop)
            ].iloc[0]
            get_sv_stats(row, sv_stats)
            sv_stats_df.loc[-1] = asdict(sv_stats[0])
    else:
        with multiprocessing.Manager() as manager:
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            svs = manager.list()
            args = []
            if query_chr is not None:
                for _, row in deletions_df[deletions_df["chr"] == query_chr].iterrows():
                    args.append((row, svs))
            else:
                for _, row in deletions_df.iterrows():
                    args.append((row, svs))
            p.starmap(get_sv_stats, args)
            p.close()
            p.join()

            for sv in svs:
                sv_stats_df.loc[-1] = asdict(sv)

        sv_stats_df.to_csv(f"{FILE_DIR}/{FILE_NAME}", index=False)


if __name__ == "__main__":
    run_all_sv(subset=[("18", 45379612, 45379807)])
    # TODO/issues:
    # pruning too many samples
    # many samples queried from STIX but not in deletions_df.csv -- check STIX database for the index
    # figure out length of SV ~ length of read (-450?)
