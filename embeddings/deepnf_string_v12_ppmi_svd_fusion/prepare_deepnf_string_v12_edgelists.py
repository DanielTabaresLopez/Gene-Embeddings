from pathlib import Path
import gzip
import csv

home = Path.home()

RAW_DIR = home / "data/raw_embeddings/deepnf_original_try/string_v12"
OUT_DIR = home / "data/raw_embeddings/deepnf_original_try/string_v12/deepnf_edgelists"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINKS = RAW_DIR / "9606.protein.links.detailed.v12.0.txt.gz"

# STRING evidence channels. We exclude combined_score because it is a derived aggregate.
CHANNELS = [
    "neighborhood",
    "fusion",
    "cooccurence",
    "coexpression",
    "experimental",
    "database",
    "textmining",
]

MIN_SCORE = 150

writers = {}
files = {}

for ch in CHANNELS:
    f = open(OUT_DIR / f"string_v12_{ch}_min{MIN_SCORE}.edgelist", "w")
    files[ch] = f
    writers[ch] = f

counts = {ch: 0 for ch in CHANNELS}
seen = {ch: set() for ch in CHANNELS}

with gzip.open(LINKS, "rt") as fh:
    header = fh.readline().strip().split()
    idx = {name: i for i, name in enumerate(header)}

    for line_i, line in enumerate(fh, start=1):
        parts = line.strip().split()
        if not parts:
            continue

        p1 = parts[idx["protein1"]]
        p2 = parts[idx["protein2"]]

        if p1 == p2:
            continue

        for ch in CHANNELS:
            score = int(parts[idx[ch]])
            if score < MIN_SCORE:
                continue

            key = tuple(sorted((p1, p2)))
            if key in seen[ch]:
                continue
            seen[ch].add(key)

            w = score / 1000.0
            writers[ch].write(f"{p1} {p2} {w:.6f}\n")
            counts[ch] += 1

for f in files.values():
    f.close()

print("Output directory:", OUT_DIR)
print("Minimum score:", MIN_SCORE)
for ch in CHANNELS:
    path = OUT_DIR / f"string_v12_{ch}_min{MIN_SCORE}.edgelist"
    print(f"{ch}: {counts[ch]} edges -> {path}")
