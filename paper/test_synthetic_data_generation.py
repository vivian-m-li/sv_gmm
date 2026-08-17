from src.synthetic.generate_data import (  # noqa:F401
    generate_and_split_sample_reads,
    generate_sv_coordinates,
)
from src.utils.config_loader import load_config


def test_synthetic_data_generation(cfg: dict):
    # svs = generate_sv_coordinates(case="C", svlen=1000, r=0.8)[0][2]
    generate_and_split_sample_reads(
        chr=1,
        svs=[(173522939, 173524155), (173522635, 173524202)],
        input_dir=cfg["paths"]["input_dir"],
        insert_size_file=cfg["input_files"]["insert_size_file"],
        model_params=cfg["model"],
        n_samples=72,
        include_split_reads=True,
        include_paired_reads=True,
        gmm_model="2d",
        run_split=True,
        plot=True,
        plot_reads=False,
        plot_sample_summary_reads=False,
        vcf_filename="output/synthetic_data_reads/HGSV_54541.vcf",
    )


if __name__ == "__main__":

    cfg = load_config()
    test_synthetic_data_generation(cfg)
