import pandas as pd


def get_ancestry_lookup():
    """
    Creates a lookup of sample_id:population
    If multiple populations are listed for a sample, take the first one (only happens for 1 sample).
    """
    ancestry_df = pd.read_csv("data/1kg/ancestry.tsv", delimiter="\t")
    sample_lookup = {
        row["Sample name"]: row["Population code"].split(",")[0]
        for _, row in ancestry_df.iterrows()
    }
    population_lookup = {}
    for i, row in ancestry_df.iterrows():
        population_lookup[row["Population code"]] = row["Superpopulation code"]
    return sample_lookup, population_lookup
