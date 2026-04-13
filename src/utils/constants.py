CHR_LENGTHS = {  # for grch38
    "1": 248956422,
    "2": 242193529,
    "3": 198295559,
    "4": 190214555,
    "5": 181538259,
    "6": 170805979,
    "7": 159345973,
    "8": 145138636,
    "9": 138394717,
    "10": 133797422,
    "11": 135086622,
    "12": 133275309,
    "13": 114364328,
    "14": 107043718,
    "15": 101991189,
    "16": 90338345,
    "17": 83257441,
    "18": 80373285,
    "19": 58617616,
    "20": 64444167,
    "21": 46709983,
    "22": 50818468,
    # ignore X and Y chromosomes in SV analysis
    # "X": 156040895,
    # "Y": 57227415,
}

NONREF_GTS = [(0, 1), (1, 0), (1, 1)]

COLORS = ["#459395", "#EB7C69", "#FDA638"]
SUPERPOPULATIONS = ["AFR", "AMR", "EUR", "EAS", "SAS"]
SUBPOPULATIONS = [  # sorted by superpopulation
    "LWK",
    "YRI",
    "ESN",
    "ASW",
    "ACB",
    "MSL",
    "GWD",
    "PUR",
    "MXL",
    "CLM",
    "PEL",
    "CDX",
    "KHV",
    "CHS",
    "CHB",
    "JPT",
    "CEU",
    "TSI",
    "IBS",
    "FIN",
    "GBR",
    "ITU",
    "GIH",
    "STU",
    "BEB",
    "PJL",
]
ANCESTRY_COLORS = {
    "AFR": "#45597e",
    "AMR": "#1d6295",
    "EUR": "#a4def4",
    "EAS": "#ffbf00",
    "SAS": "#ffe69f",
}

GMM_MODELS = ["1d_len", "1d_L", "2d"]
MODEL_NAMES = ["Length-only", "L-only", "Length-L"]

GMM_AXES = {
    "L": lambda x: x[0],
    "R": lambda x: x[1],
    "Length": lambda x: x[1] - x[0],
}

SYNTHETIC_DATA_CENTROIDS = {
    "A": [[100000, 102553], [100500, 102053]],
    "B": [[100000, 102553], [100500, 103053]],
    "C": [[100000, 102553], [103053, 105606]],
    "D": [[100000, 102553], [101000, 103003], [100436, 103452]],
    "E": [[100000, 102553], [101000, 103003], [100699, 103757]],
}
