from dataclasses import dataclass

from src.synthetic.generate_data import (  # noqa:F401
    generate_and_split_sample_reads,
    generate_sv_coordinates,
)
from src.utils.config_loader import load_config


@dataclass
class SVLookup:
    sv_id: str
    clustered_coords: list[tuple[int, int]]
    svlen: int
    n_samples: int
    n_clusters_expected: int


test_sv_lookup = {
    "HGSV_15262": SVLookup("HGSV_15262", [(199140920, 199144677)], 774, 772, 1),
    "HGSV_143868": SVLookup("HGSV_143868", [(3168426, 3168623)], 2210, 73, 1),
    "HGSV_39753": SVLookup("HGSV_39753", [(236446871, 236447567)], 1152, 87, 1),
    "HGSV_204881": SVLookup("HGSV_204881", [(746193, 746683)], 2488, 2425, 1),
    "HGSV_226693": SVLookup(
        "HGSV_226693",
        [(71179505, 71179822), (71179427, 71179909)],
        2500,
        2458,
        2,
    ),
    "HGSV_5515": SVLookup("HGSV_5515", [(58278241, 58279150)], 2488, 2487, 1),
    "HGSV_218106": SVLookup("HGSV_218106", [(56082821, 56095596)], 127, 126, 1),
    "HGSV_89": SVLookup(
        "HGSV_89", [(964504, 964937), (964467, 965008)], 2488, 274, 2
    ),
    "HGSV_54541": SVLookup(
        "HGSV_54541",
        [(173522939, 173524155), (173522635, 173524202)],
        83,
        72,
        2,
    ),
    "HGSV_149774": SVLookup("HGSV_149774", [(68570263, 68571638)], 25, 24, 1),
    "HGSV_245658": SVLookup(
        "HGSV_245658",
        [(24660239, 24661204), (24659891, 24661496), (24660276, 24661218)],
        382,
        91,
        3,
    ),
    "HGSV_220750": SVLookup(
        "HGSV_220750", [(82502735, 82505050), (82502449, 82505420)], 14, 13, 2
    ),
    "HGSV_161412": SVLookup(
        "HGSV_161412", [(66536554, 66537777), (66536272, 66537614)], 28, 19, 2
    ),
}


def test_synthetic_data_generation(cfg: dict):
    # svs = generate_sv_coordinates(case="C", svlen=1000, r=0.8)[0][2]
    for sv in test_sv_lookup.values():
        print(f"\n{sv.sv_id}: n svs expected: {sv.n_clusters_expected}")
        generate_and_split_sample_reads(
            chr=1,
            svs=sv.clustered_coords,
            input_dir=cfg["paths"]["input_dir"],
            insert_size_file=cfg["input_files"]["insert_size_file"],
            model_params=cfg["model"],
            n_samples=sv.n_samples,
            include_split_reads=True,
            include_paired_reads=True,
            gmm_model="2d",
            run_split=True,
            plot=False,
            plot_reads=False,
            plot_sample_summary_reads=False,
            vcf_filename=f"output/synthetic_data_reads/{sv.sv_id}.vcf",
        )


if __name__ == "__main__":
    cfg = load_config()
    test_synthetic_data_generation(cfg)
