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


test_sv_lookup = {
    "HGSV_15262": SVLookup("HGSV_15262", [(199140920, 199144677)], 3758, 772),
    "HGSV_143868": SVLookup("HGSV_143868", [(3168426, 3168623)], 198, 72),
    "HGSV_39753": SVLookup("HGSV_39753", [(236446871, 236447567)], 697, 87),
    "HGSV_204881": SVLookup("HGSV_204881", [(746195, 746677)], 309, 2419),
    "HGSV_226693": SVLookup("HGSV_226693", [(71179505, 71179822)], 318, 2458),
    "HGSV_5515": SVLookup("HGSV_5515", [(58278241, 58279150)], 910, 2487),
    "HGSV_218106": SVLookup("HGSV_218106", [(56082821, 56095596)], 12776, 126),
    "HGSV_89": SVLookup("HGSV_89", [(964497, 964926)], 370, 247),
    "HGSV_54541": SVLookup(
        "HGSV_54541", [(173522939, 173524148), (173522649, 173524190)], 1143, 69
    ),
    "HGSV_149774": SVLookup(
        "HGSV_149774", [(68570275, 68571634), (68570116, 68571887)], 1290, 23
    ),
    "HGSV_245658": SVLookup(
        "HGSV_245658", [(24660235, 24661195), (24659897, 24661532)], 841, 87
    ),
    "HGSV_220750": SVLookup(
        "HGSV_220750", [(82502735, 82505050), (82502449, 82505420)], 2316, 13
    ),
    "HGSV_161412": SVLookup(
        "HGSV_161412", [(66536575, 66537783), (66536288, 66537614)], 1134, 19
    ),
}


def test_synthetic_data_generation(cfg: dict):
    # svs = generate_sv_coordinates(case="C", svlen=1000, r=0.8)[0][2]
    for sv in test_sv_lookup.values():
        print(f"\n{sv.sv_id}: n svs expected: {len(sv.clustered_coords)}")
        generate_and_split_sample_reads(
            chr=1,
            svs=sv.clustered_coords,
            input_dir=cfg["paths"]["input_dir"],
            insert_size_file=cfg["input_files"]["insert_size_file"],
            model_params=cfg["model"],
            n_samples=sv.n_samples,
            include_split_reads=True,
            include_paired_reads=True,
            run_split=True,
            plot=False,
            plot_reads=False,
            plot_sample_summary_reads=False,
            vcf_filename=f"output/synthetic_data_reads/{sv.sv_id}.vcf",
        )


if __name__ == "__main__":
    cfg = load_config()
    test_synthetic_data_generation(cfg)
