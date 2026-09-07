#!/usr/bin/env python3
"""
dp_01_build_splits.py
 
Step 1 of the Directional Predictability track.
 
Builds ONE global train / validation / test partition of the gene universe,
used by every embedding's PCA and by every pair's model. A single global split
matters for three reasons:
 
  1. No leakage. Each embedding's PCA is fitted on training genes only, so
     test genes never influence the representation the model sees.
  2. Comparability. Every pair is scored on the same held-out genes, so
     R-squared values can be compared across pairs.
  3. Cost. PCA is computed once per embedding rather than once per pair.
 
FAMILY-AWARE SPLITTING
----------------------
A purely random gene split leaks. Paralogues and recently duplicated genes
have near-identical protein sequence embeddings, so a random split places a
gene in training and its near-twin in test. The model can then memorise rather
than generalise, which inflates R-squared specifically for sequence-based
pairs — exactly the comparisons of most interest.
 
This script therefore groups genes into families and assigns whole families to
a single split. Families are derived, in order of preference:
 
  1. From an external cluster file, if supplied (`family_cluster_file`), with
     columns gene_id and family_id. Ensembl paralogue groups or an MMseqs2 /
     CD-HIT clustering of protein sequences are all suitable.
  2. Otherwise from a reference embedding (`family_reference_embedding`,
     default esm2) with non-chaining BIRCH clustering of L2-normalised vectors.
     This is self-contained and needs no external data, but it is a proxy for
     sequence similarity, not a curated family definition.
  3. Otherwise a plain random split, with a loud warning.
 
Both a family-aware and a purely random split are always written, so the
inflation caused by leakage can be measured directly rather than assumed.
 
Outputs
-------
    gene_splits.tsv        gene_id, family_id, split_family, split_random
    family_sizes.tsv       family_id, n_genes
    report.md
    run_metadata.json
"""
 
from __future__ import annotations
 
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
 
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
 
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.cluster import Birch

from dp_manifest import load_manifest, panel_sha256
 
 
# ------------------------------------------------------------- union-find
 
 
class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=np.int32)
 
    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root
 
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
 
 
# ------------------------------------------------------------ family build
 
 
def families_from_embedding_connected_components(
    gene_ids: np.ndarray,
    matrix: np.ndarray,
    threshold: float,
    block_size: int = 512,
) -> np.ndarray:
    """
    Connected components of the graph joining genes whose cosine similarity
    exceeds `threshold`. Computed in row blocks so the full n x n similarity
    matrix is never held in memory.
    """
    normalized = matrix.astype(np.float32)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = normalized / norms
 
    n = normalized.shape[0]
    union_find = UnionFind(n)
    n_edges = 0
 
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        similarity = normalized[start:stop] @ normalized.T
        rows, columns = np.nonzero(similarity >= threshold)
        for row, column in zip(rows, columns):
            global_row = start + int(row)
            if global_row < int(column):  # upper triangle only
                union_find.union(global_row, int(column))
                n_edges += 1
 
    labels = np.array([union_find.find(i) for i in range(n)])
    print(f"    edges above cosine {threshold}: {n_edges:,}")
    return labels


def families_from_embedding_birch(
    matrix: np.ndarray,
    cosine_threshold: float,
    branching_factor: int = 50,
) -> np.ndarray:
    """Cluster normalised vectors without single-linkage chaining.

    The old implementation formed a threshold graph and took connected
    components.  A long chain of individually similar vectors can percolate
    into one enormous component even when its endpoints are unrelated.  BIRCH
    maintains compact subclusters and therefore does not have that failure
    mode.  On the unit sphere, cosine ``c`` corresponds to Euclidean distance
    ``sqrt(2 * (1-c))``.  Half that distance is used as a conservative radius.

    This remains an embedding-derived proxy.  A curated or sequence-cluster
    file is preferred and is always used when ``family_cluster_file`` exists.
    """
    if not 0.0 < cosine_threshold < 1.0:
        raise ValueError("family_similarity_threshold must lie in (0, 1)")
    normalized = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = normalized / norms
    radius = float(np.sqrt(2.0 * (1.0 - cosine_threshold)) / 2.0)
    model = Birch(
        threshold=radius,
        branching_factor=int(branching_factor),
        n_clusters=None,
        compute_labels=True,
    )
    labels = model.fit_predict(normalized).astype(np.int64, copy=False)
    print(f"    BIRCH radius: {radius:.5f}")
    return labels
 
 
def assign_families_to_splits(
    family_ids: np.ndarray, fractions: tuple[float, float, float], seed: int
) -> np.ndarray:
    """
    Assign whole families to train / val / test, greedily filling toward the
    requested gene-count fractions. Families are shuffled first, and the
    largest are placed first.  Each family goes to the split with the largest
    *relative* remaining deficit, so the 70/10/20 targets are respected rather
    than favouring the numerically largest training target.
    """
    unique, counts = np.unique(family_ids, return_counts=True)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique))
    unique, counts = unique[shuffled], counts[shuffled]
    order = np.argsort(-counts)  # largest families first
 
    total = counts.sum()
    targets = np.array(fractions) * total
    current = np.zeros(3)
    family_split = {}
 
    for index in order:
        relative_deficit = (targets - current) / np.maximum(targets, 1.0)
        choice = int(np.argmax(relative_deficit))
        family_split[unique[index]] = choice
        current[choice] += counts[index]
 
    names = np.array(["train", "val", "test"])
    return names[np.array([family_split[f] for f in family_ids])]


def validate_family_partition(
    family_ids: np.ndarray,
    split_family: np.ndarray,
    fractions: tuple[float, float, float],
    max_group_fraction: float,
    max_split_deviation: float,
    min_grouped_gene_fraction: float = 0.05,
) -> dict:
    """Fail before expensive work when a purported family split is unusable."""
    unique, counts = np.unique(family_ids, return_counts=True)
    largest_fraction = float(counts.max() / counts.sum())
    grouped_gene_fraction = float(counts[counts > 1].sum() / counts.sum())
    if largest_fraction > max_group_fraction:
        raise RuntimeError(
            "unsafe family partition: the largest group contains "
            f"{counts.max():,}/{counts.sum():,} genes ({largest_fraction:.1%}), "
            f"above family_max_group_fraction={max_group_fraction:.1%}. "
            "Use a curated/MMseqs2 cluster file or tighten the proxy clustering."
        )
    if grouped_gene_fraction < min_grouped_gene_fraction:
        raise RuntimeError(
            "unsafe family partition: only "
            f"{grouped_gene_fraction:.1%} of genes belong to a non-singleton "
            f"group, below family_min_grouped_gene_fraction="
            f"{min_grouped_gene_fraction:.1%}. This is effectively a random split."
        )

    names = ("train", "val", "test")
    observed = np.array([(split_family == name).mean() for name in names])
    deviations = np.abs(observed - np.asarray(fractions))
    if deviations.max() > max_split_deviation:
        raise RuntimeError(
            "unsafe family split proportions: observed "
            + ", ".join(f"{n}={v:.1%}" for n, v in zip(names, observed))
            + f"; maximum allowed deviation is {max_split_deviation:.1%}."
        )
    family_counts = {
        name: int(len(np.unique(family_ids[split_family == name]))) for name in names
    }
    if min(family_counts.values()) < 2:
        raise RuntimeError(
            f"unsafe family split: family counts by split are {family_counts}"
        )
    return {
        "largest_family_size": int(counts.max()),
        "largest_family_fraction": largest_fraction,
        "grouped_gene_fraction": grouped_gene_fraction,
        "family_counts_by_split": family_counts,
        "split_proportions": {n: float(v) for n, v in zip(names, observed)},
        "n_families": int(len(unique)),
    }
 
 
def random_split(n: int, fractions: tuple[float, float, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    out = np.empty(n, dtype=object)
    out[order[:n_train]] = "train"
    out[order[n_train:n_train + n_val]] = "val"
    out[order[n_train + n_val:]] = "test"
    return out
 
 
# -------------------------------------------------------------------- main
 
 
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
 
    started = time.time()
    config = yaml.safe_load(args.config.read_text())
    panel, manifest_path = load_manifest(config, args.config)
 
    library = Path(os.path.expanduser(config["library_root"]))
    out = Path(os.path.expanduser(config["splits_output"]))
    out.mkdir(parents=True, exist_ok=True)
    array_key = config.get("array_key", "embeddings")
    id_column = config.get("id_column", "ensembl_gene_id")
    exclusions = set(config.get("global_exclusions", []))
    fractions = tuple(config["split_fractions"])
    seed = int(config["seed"])
 
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"split_fractions must sum to 1.0, got {fractions}")
 
    print("=" * 68)
    print("DP STEP 1: GLOBAL FAMILY-AWARE GENE SPLITS")
    print("=" * 68)
    print(f"library   : {library}")
    print(f"fractions : train {fractions[0]:.0%} / val {fractions[1]:.0%} / test {fractions[2]:.0%}")
 
    # ---------------- gene universe from the panel ----------------
    pkg_root = library / "data" / "embeddings"
    panel_ids = panel.embedding_id.tolist()
    excluded_selected = sorted(set(panel_ids) & exclusions)
    if excluded_selected:
        raise ValueError(
            "manifest-selected embeddings cannot also be global_exclusions: "
            f"{excluded_selected}"
        )
    local_ids = {
        d.name for d in pkg_root.iterdir()
        if d.is_dir() and (d / "embeddings.npz").is_file()
    }
    missing = sorted(set(panel_ids) - local_ids)
    if missing:
        raise FileNotFoundError(f"manifest embeddings absent from local library: {missing}")
    ignored = sorted(local_ids - set(panel_ids))
    ids = panel_ids
    print(f"manifest  : {manifest_path}")
    print(f"panel     : {len(ids)} manifest-selected embeddings")
    print(f"ignored   : {len(ignored)} unlisted local packages\n")
 
    universe = set()
    for embedding_id in ids:
        genes = pd.read_csv(pkg_root / embedding_id / "genes.tsv", sep="\t", dtype=str)
        with np.load(pkg_root / embedding_id / "embeddings.npz") as npz:
            matrix = np.asarray(npz[array_key], dtype=np.float32)
        finite = np.isfinite(matrix).all(axis=1)
        nonzero = np.linalg.norm(matrix, axis=1) > 0
        series = pd.Series(genes[id_column].to_numpy())
        valid = finite & nonzero & (~series.duplicated(keep=False).to_numpy())
        universe.update(series[valid].tolist())
        del matrix
 
    gene_ids = np.array(sorted(universe))
    print(f"gene universe: {len(gene_ids):,}")
 
    # ---------------- families ----------------
    family_source = "random"
    family_ids = np.arange(len(gene_ids))  # default: every gene its own family
 
    cluster_file = config.get("family_cluster_file")
    reference = config.get("family_reference_embedding")
 
    if cluster_file and Path(os.path.expanduser(str(cluster_file))).is_file():
        path = Path(os.path.expanduser(str(cluster_file)))
        table = pd.read_csv(path, sep=None, engine="python", dtype=str)
        gene_column = "gene_id" if "gene_id" in table.columns else table.columns[0]
        family_column = "family_id" if "family_id" in table.columns else table.columns[1]
        if table[[gene_column, family_column]].isna().any().any():
            raise ValueError(f"{path}: gene_id and family_id must not be missing")
        if table[gene_column].duplicated().any():
            raise ValueError(f"{path}: duplicate gene IDs are not allowed")
        mapping = dict(zip(table[gene_column], table[family_column]))
        codes, next_code = {}, len(gene_ids)
        assigned = []
        for position, gene in enumerate(gene_ids):
            key = mapping.get(gene)
            if key is None:
                assigned.append(position)  # unclustered gene stays a singleton
            else:
                if key not in codes:
                    codes[key] = next_code
                    next_code += 1
                assigned.append(codes[key])
        family_ids = np.array(assigned)
        family_source = f"external file: {path.name}"
        print(f"families  : from {path.name}")
 
    elif reference and (pkg_root / reference).is_dir():
        method = str(config.get("family_embedding_cluster_method", "birch"))
        print(f"families  : proxy derived from {reference}; method={method}, "
              f"cosine target={config['family_similarity_threshold']}")
        genes = pd.read_csv(pkg_root / reference / "genes.tsv", sep="\t", dtype=str)
        with np.load(pkg_root / reference / "embeddings.npz") as npz:
            matrix = np.asarray(npz[array_key], dtype=np.float32)
        series = pd.Series(genes[id_column].to_numpy())
        finite = np.isfinite(matrix).all(axis=1)
        nonzero = np.linalg.norm(matrix, axis=1) > 0
        valid = finite & nonzero & (~series.duplicated(keep=False).to_numpy())
        reference_genes = series[valid].to_numpy()
        reference_matrix = matrix[valid]
 
        if method == "birch":
            labels = families_from_embedding_birch(
                reference_matrix,
                float(config["family_similarity_threshold"]),
                int(config.get("family_birch_branching_factor", 50)),
            )
        elif method == "connected_components":
            if not config.get("allow_unsafe_connected_components", False):
                raise RuntimeError(
                    "connected-components family clustering is disabled because it "
                    "can percolate into a giant component. Set "
                    "allow_unsafe_connected_components=true only for a diagnostic run."
                )
            labels = families_from_embedding_connected_components(
                reference_genes, reference_matrix,
                float(config["family_similarity_threshold"]),
                int(config.get("family_block_size", 512)),
            )
        else:
            raise ValueError(
                f"unknown family_embedding_cluster_method={method!r}; expected birch"
            )
 
        label_of = dict(zip(reference_genes, labels))
        codes, next_code = {}, len(gene_ids)
        assigned = []
        for position, gene in enumerate(gene_ids):
            key = label_of.get(gene)
            if key is None:
                assigned.append(position)
            else:
                if key not in codes:
                    codes[key] = next_code
                    next_code += 1
                assigned.append(codes[key])
        family_ids = np.array(assigned)
        family_source = (
            f"{reference} {method} proxy at cosine target "
            f"{config['family_similarity_threshold']}"
        )
        del matrix, reference_matrix
 
    else:
        print("  WARNING: no family source available. Falling back to a purely")
        print("  random split. Sequence-based pairs will be leakage-inflated and")
        print("  their R-squared values must be read with that in mind.")
 
    unique, counts = np.unique(family_ids, return_counts=True)
    multi = int((counts > 1).sum())
    print(f"            {len(unique):,} families, {multi:,} with more than one gene, "
          f"largest {counts.max():,}")
 
    # ---------------- splits ----------------
    split_family = assign_families_to_splits(family_ids, fractions, seed)
    guardrails = validate_family_partition(
        family_ids,
        split_family,
        fractions,
        float(config.get("family_max_group_fraction", 0.05)),
        float(config.get("family_max_split_deviation", 0.03)),
        float(config.get("family_min_grouped_gene_fraction", 0.05)),
    )
    split_random = random_split(len(gene_ids), fractions, seed)
 
    table = pd.DataFrame({
        "gene_id": gene_ids,
        "family_id": family_ids,
        "split_family": split_family,
        "split_random": split_random,
    })
    table.to_csv(out / "gene_splits.tsv", sep="\t", index=False)
 
    pd.DataFrame({"family_id": unique, "n_genes": counts}).sort_values(
        "n_genes", ascending=False).to_csv(out / "family_sizes.tsv", sep="\t", index=False)
 
    print("\nsplit sizes")
    for name in ("split_family", "split_random"):
        proportions = table[name].value_counts(normalize=True)
        counts_by = table[name].value_counts()
        print(f"  {name:<14} " + "  ".join(
            f"{s}={counts_by[s]:,} ({proportions[s]:.1%})" for s in ("train", "val", "test")))
 
    # ---------------- report ----------------
    elapsed = time.time() - started
    report = [
        "# DP step 1 — global family-aware gene splits", "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}    Runtime: {elapsed:.1f}s", "",
        "## Universe", "",
        f"- Embeddings in panel: {len(ids)}",
        f"- Genes: {len(gene_ids):,}",
        f"- Family source: {family_source}",
        f"- Families: {len(unique):,} ({multi:,} with more than one gene, "
        f"largest {counts.max():,})", "",
        f"- Largest-family fraction: {guardrails['largest_family_fraction']:.2%}",
        f"- Genes in non-singleton families: {guardrails['grouped_gene_fraction']:.2%}",
        f"- Families by split: {guardrails['family_counts_by_split']}", "",
        "## Split sizes", "",
        "| split | family-aware | random |",
        "|---|---|---|",
    ]
    for s in ("train", "val", "test"):
        report.append(f"| {s} | {int((split_family == s).sum()):,} | "
                      f"{int((split_random == s).sum()):,} |")
    report += [
        "", "## Why two splits", "",
        "`split_family` keeps whole gene families together and is the primary",
        "split. `split_random` is retained so that leakage can be measured",
        "rather than assumed: training the same pair under both and comparing",
        "held-out R-squared quantifies how much a random split inflates the",
        "result. Expect the gap to be largest for protein-sequence pairs.",
        "",
        "## Guardrail", "",
        "Every downstream step — PCA fitting, model training, standardisation",
        "statistics — must use ONLY the training genes of the chosen split.",
        "",
    ]
    (out / "report.md").write_text("\n".join(report))
 
    (out / "run_metadata.json").write_text(json.dumps({
        "script": Path(__file__).name,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "manifest_file": str(manifest_path),
        "panel_sha256": panel_sha256(panel),
        "embedding_ids": ids,
        "ignored_local_packages": ignored,
        "config": config, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(elapsed, 2), "python_version": sys.version,
        "platform": platform.platform(), "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "n_genes": int(len(gene_ids)), "n_families": int(len(unique)),
        "family_source": family_source, "family_guardrails": guardrails,
        "status": "PASS",
    }, indent=2))
 
    print(f"\nDONE in {elapsed:.1f}s -> {out}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 
