#!/usr/bin/env python3
"""
dp_03_train_sweep.py
 
Directional Predictability, version 2.
 
Changes from v1, all following the review:
 
1. ASYMMETRIC PCA TRUNCATION. PCA with all components kept is a rotation, so
   an MLP's first linear layer can undo it exactly; nothing is lost by feeding
   principal components rather than raw dimensions. The loss comes only from
   TRUNCATION, and input and output truncation are different problems:
 
     output  must be truncated (a 20,421-unit output layer is not sensible)
             and truncation changes what R-squared means, so it is held to a
             90% variance target and the retained fraction is reported;
     input   is cheap to widen (one matrix, not the whole network), so every
             cached component is used, out to a 99% variance target.
 
2. DEPTH SWEEP. 0, 1, 2, 3 and 4 hidden layers. Depth 0 is a linear map fitted
   by gradient descent with the same input dropout and optimiser family.  The
   target is centred and divided by one global training-set scale, preserving
   variance-weighted R-squared while preventing scale-driven divergence.
   Ridge is still fitted for continuity with v1 where tractable.
 
3. WIDTH RULE. Hidden width is the lowest of 32, 64, 128, 256, 512 that is at
   least max(n_input_components, n_output_components), chosen per pair. A
   16-component embedding no longer gets a 1,024-wide network.
 
4. BOTH SPLITS. split_family and split_random. The random split is a
   capability control: if a pair fails on the family split but succeeds on the
   random one, the architecture can learn and the failure is genuine absence
   of shared information rather than a modelling failure.
 
5. SIGNAL FILTER. Restricted to pairs where v1 found any signal, which more
   than halves the work.
 
6. RAW INPUT MODE. --input-mode raw feeds the untruncated embedding, to
   measure empirically what input truncation costs. Intended for a subset.

7. FULL-TARGET ACCOUNTING. No target direction is discarded because a fixed
   PCA cap misses 90%. Alongside retained-subspace R2, the run reports the
   exact full-raw-target R2 of the fitted predictor when omitted target
   directions are predicted at their training mean.
 
Resumable: one JSON per (pair, split, input mode), holding every depth and
both directions.
 
    python dp_03_train_sweep.py --config ... --estimate
    python dp_03_train_sweep.py --config ... --split-column split_random
    python dp_03_train_sweep.py --config ... --input-mode raw --limit 100
"""
 
from __future__ import annotations
 
import argparse, hashlib, json, os, platform, sys, time, warnings
from itertools import combinations
from pathlib import Path
 
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("DP_THREADS", "1"))
 
import numpy as np
import pandas as pd
import yaml
 
warnings.filterwarnings("ignore", message=".*Stochastic Optimizer.*")
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dp_metrics import evaluate
from dp_manifest import canonicalize_id, id_aliases, load_manifest, panel_sha256
 
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base_seed: int, *parts: str) -> int:
    """Stable across processes and Python versions (unlike built-in hash())."""
    payload = "\x1f".join(map(str, parts)).encode()
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 1_000_000
    return int(base_seed + offset)


def resolve_pca_root(config: dict, split_column: str) -> Path:
    raw = os.path.expanduser(str(config["pca_output"]))
    if "{split_column}" in raw:
        return Path(raw.format(split_column=split_column))
    return Path(raw) / split_column


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
 
 
# ============================================================ data access
 
 
class Source:
    """Serves PCA coordinates or raw vectors, with per-embedding metadata."""
 
    def __init__(self, pca_root: Path, library: Path, mode: str,
                 array_key: str, id_column: str, output_variance: float,
                 capacity: int = 4, raw_max_dim: int | None = None,
                 split_column: str | None = None,
                 min_target_variance: float = 0.0,
                 compute_full_target_r2: bool = True,
                 aliases: dict[str, str] | None = None):
        self.pca_root, self.library, self.mode = pca_root, library, mode
        self.array_key, self.id_column = array_key, id_column
        self.output_variance = output_variance
        self.capacity = capacity
        # Raw input on a 20,000-dimensional embedding gives a first layer of
        # ~10M parameters against ~13,000 training genes. Above this width the
        # source falls back to cached components and says so, rather than
        # stalling the run. Set to null to disable the fallback entirely.
        self.raw_max_dim = raw_max_dim
        self.split_column = split_column
        self.min_target_variance = min_target_variance
        self.compute_full_target_r2 = compute_full_target_r2
        self.aliases = aliases or {}
        self._store, self._order = {}, []
 
    def get(self, embedding_id: str):
        """
        Returns (lookup, input_matrix, output_matrix, n_in, n_out, retained,
                 used_raw, n_intrinsic).
 
        n_in         actual width fed to the network (raw dimension, or the
                     number of cached components)
        n_intrinsic  number of components needed for the cache's variance
                     target. This is the embedding's effective width and is
                     what the hidden-layer rule should use: a 5,313-dimensional
                     package whose variance lives in 108 components does not
                     need a 512-wide network.
        """
        if embedding_id in self._store:
            self._order.remove(embedding_id); self._order.append(embedding_id)
            return self._store[embedding_id]
 
        directory = self.pca_root / embedding_id
        if not directory.is_dir():
            legacy = sorted(
                old for old, new in self.aliases.items()
                if new == embedding_id and (self.pca_root / old).is_dir()
            )
            if len(legacy) == 1:
                directory = self.pca_root / legacy[0]
            elif len(legacy) > 1:
                raise RuntimeError(
                    f"{embedding_id}: multiple legacy PCA cache directories: {legacy}"
                )
        cache_meta = json.loads((directory / "meta.json").read_text())
        if self.split_column and cache_meta.get("split_column") != self.split_column:
            raise RuntimeError(
                f"{embedding_id}: PCA cache was fitted for "
                f"{cache_meta.get('split_column')!r}, expected {self.split_column!r}"
            )
        coordinates = np.load(directory / "coordinates.npy")
        genes = pd.read_csv(directory / "genes.tsv", sep="\t", dtype=str)["gene_id"].to_numpy()
        with np.load(directory / "basis.npz") as z:
            ratio = z["explained_variance_ratio"].astype(np.float64)
 
        # Output: the smallest prefix reaching the 90% target.
        cumulative = np.cumsum(ratio)
        n_out = int(np.searchsorted(cumulative, self.output_variance) + 1)
        n_out = int(min(max(n_out, 2), coordinates.shape[1]))
        retained = float(cumulative[n_out - 1])
        target_eligible = retained + 1e-12 >= self.min_target_variance
 
        output_matrix = coordinates[:, :n_out]
 
        used_raw = False
        raw_target_matrix = None
        if self.mode == "raw" or self.compute_full_target_r2:
            package = self.library / "data" / "embeddings" / embedding_id
            raw_genes = pd.read_csv(package / "genes.tsv", sep="\t", dtype=str)[self.id_column].to_numpy()
            with np.load(package / "embeddings.npz") as npz:
                raw = np.asarray(npz[self.array_key], dtype=np.float32)
            position = {g: i for i, g in enumerate(raw_genes)}
            rows = np.array([position.get(g, -1) for g in genes])
            keep = rows >= 0
            if not keep.all():  # keep alignment with the PCA gene order
                genes, output_matrix = genes[keep], output_matrix[keep]
                coordinates = coordinates[keep]
                rows = rows[keep]
            raw_target_matrix = raw[rows]

        if self.mode == "raw":
            if self.raw_max_dim is not None and raw.shape[1] > self.raw_max_dim:
                input_matrix = coordinates
            else:
                input_matrix = raw_target_matrix
                used_raw = True
        else:
            input_matrix = coordinates  # every cached component
 
        lookup = {g: i for i, g in enumerate(genes)}
        record = (lookup, input_matrix, output_matrix,
                  int(input_matrix.shape[1]), n_out, retained, used_raw,
                  int(coordinates.shape[1]), target_eligible,
                  cache_meta.get("cache_signature"), raw_target_matrix)
        self._store[embedding_id] = record
        self._order.append(embedding_id)
        while len(self._order) > self.capacity:
            del self._store[self._order.pop(0)]
        return record
 
 
def choose_width(n_intrinsic: int, n_out: int, options: list[int]) -> int:
    """
    Lowest option at least as wide as the larger of the two sides.
 
    Uses INTRINSIC dimensionality, not the raw input width. Feeding a raw
    20,421-dimensional vector does not mean the network needs 20,421 units of
    capacity; it needs enough to carry the information, which is bounded by the
    number of components the embedding actually uses.
    """
    need = max(n_intrinsic, n_out)
    for width in sorted(options):
        if width >= need:
            return width
    return max(options)


def target_variance_fractions(
    retained: np.ndarray,
    full: np.ndarray | None,
    index: dict[str, np.ndarray],
) -> dict[str, float]:
    """Fraction of full target sum-of-squares represented by retained PCs.

    Both spaces are centred on their own training mean. Because PCA loadings
    are orthonormal, the ratio is exact for the selected rows. Multiplying a
    retained-space R2 by this fraction gives the full-target R2 of the same
    predictor when every omitted direction is predicted at its training mean.
    """
    if full is None:
        return {name: 1.0 for name in ("train", "val", "test")}
    retained_mean = retained[index["train"]].mean(axis=0, dtype=np.float64)
    full_mean = full[index["train"]].mean(axis=0, dtype=np.float64)
    fractions = {}
    for name in ("train", "val", "test"):
        rows = index[name]
        retained_ss = float(np.sum(
            (retained[rows] - retained_mean) ** 2, dtype=np.float64
        ))
        full_ss = float(np.sum(
            (full[rows] - full_mean) ** 2, dtype=np.float64
        ))
        if not np.isfinite(full_ss) or full_ss <= 0:
            raise RuntimeError(f"invalid full-target sum of squares for {name}: {full_ss}")
        fraction = retained_ss / full_ss
        # Tiny randomized-PCA/numerical deviations above one are harmless;
        # anything material indicates an alignment bug.
        if fraction > 1.0 + 1e-5 or fraction < -1e-12:
            raise RuntimeError(
                f"invalid retained/full target variance fraction for {name}: {fraction}"
            )
        fractions[name] = float(np.clip(fraction, 0.0, 1.0))
    return fractions
 
 
# ================================================================== models
 
 
def build_mlp(n_in: int, n_out: int, width: int, depth: int, dropout: float):
    """Depth is hidden-layer count; depth 0 remains linear at evaluation."""
    if depth == 0:
        model = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_in, n_out))
        final = model[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        return model
    layers, current = [nn.Dropout(dropout)], n_in
    for _ in range(depth):
        layers += [nn.Linear(current, width), nn.LayerNorm(width),
                   nn.GELU(), nn.Dropout(dropout)]
        current = width
    layers.append(nn.Linear(current, n_out))
    model = nn.Sequential(*layers)
    # Every depth begins at the null (training-mean) predictor after target
    # unscaling.  This removes a severe scale-dependent initialisation failure
    # observed for very wide raw inputs.
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    return model
 
 
def fit_network(X, Y, index, config, width, depth, seed, device_name):
    train, val, test = index["train"], index["val"], index["test"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(device_name)
 
    model = build_mlp(X.shape[1], Y.shape[1], width, depth,
                      float(config["mlp_dropout"])).to(device)
 
    train_mean = Y[train].mean(axis=0, dtype=np.float64)
    centred = Y - train_mean
    target_scale = float(np.sqrt(np.mean(centred[train] ** 2, dtype=np.float64)))
    if not np.isfinite(target_scale) or target_scale < 1e-12:
        raise RuntimeError(f"invalid target scale: {target_scale}")
    scaled = np.asarray(centred / target_scale, dtype=np.float32)

    xt = torch.tensor(X[train], dtype=torch.float32, device=device)
    yt = torch.tensor(scaled[train], dtype=torch.float32, device=device)
    xv = torch.tensor(X[val], dtype=torch.float32, device=device)
    yv = torch.tensor(scaled[val], dtype=torch.float32, device=device)
    ss_tot = float((yv ** 2).sum().item())
 
    learning_rate = float(config.get("linear_lr", config["mlp_lr"])) \
        if depth == 0 else float(config["mlp_lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=float(config["mlp_weight_decay"]))
    epochs = int(config["mlp_max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    batch = int(config["mlp_batch_size"])
    patience = int(config["mlp_patience"])
    gradient_clip = float(config.get("mlp_gradient_clip", 1.0))
 
    model.eval()
    with torch.no_grad():
        initial_ss_res = float(((model(xv) - yv) ** 2).sum().item())
    best = 1.0 - initial_ss_res / ss_tot if ss_tot > 0 else -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    waited = 0
    n = xt.shape[0]
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            optimizer.zero_grad()
            loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            ss_res = float(((model(xv) - yv) ** 2).sum().item())
        score = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if not np.isfinite(score):
            raise RuntimeError(f"non-finite validation R2 at epoch {epoch + 1}")
        if score > best + 1e-5:
            best, waited = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(torch.tensor(
            X[test], dtype=torch.float32, device=device
        )).cpu().numpy().astype(np.float64)
    prediction = prediction_scaled * target_scale + train_mean
    if not np.isfinite(prediction).all():
        raise RuntimeError("network produced non-finite test predictions")
    n_params = sum(p.numel() for p in model.parameters())
    return prediction, best, epoch + 1, n_params, target_scale, learning_rate
 
 
def fit_ridge(X, Y, index, alphas):
    train, val, test = index["train"], index["val"], index["test"]
    design = np.hstack([X[train], np.ones((len(train), 1))])
    gram, cross = design.T @ design, design.T @ Y[train]
    penalty = np.eye(X.shape[1] + 1); penalty[-1, -1] = 0.0
    design_val = np.hstack([X[val], np.ones((len(val), 1))])
    train_mean = Y[train].mean(axis=0)
    ss_tot = float(((Y[val] - train_mean) ** 2).sum())
    best = (None, -np.inf, None)
    for alpha in alphas:
        try:
            weights = np.linalg.solve(gram + alpha * penalty, cross)
        except np.linalg.LinAlgError:
            continue
        score = 1.0 - float(((design_val @ weights - Y[val]) ** 2).sum()) / ss_tot
        if score > best[1]:
            best = (weights, score, alpha)
    if best[0] is None:
        raise RuntimeError("ridge failed for every alpha")
    prediction = np.hstack([X[test], np.ones((len(test), 1))]) @ best[0]
    return prediction, best[1], best[2]
 
 
# ============================================================== one direction
 
 
def run_direction(
    X_raw, Y, index, config, width, seed, device_name, ks, thresholds,
    full_target=None,
):
    train = index["train"]
    x_mean = X_raw[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = X_raw[train].std(axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < 1e-8] = 1.0
    X = (X_raw - x_mean) / x_std          # stays float32
    train_mean = Y[train].mean(axis=0, dtype=np.float64)
    variance_fractions = target_variance_fractions(Y, full_target, index)

    def add_target_space_metrics(record: dict, val_r2: float | None = None) -> None:
        retained_test = float(record["r2"])
        record["r2_retained_subspace"] = retained_test
        record["r2_full_target_at_k"] = (
            variance_fractions["test"] * retained_test
        )
        record["target_variance_fraction_train"] = variance_fractions["train"]
        record["target_variance_fraction_val"] = variance_fractions["val"]
        record["target_variance_fraction_test"] = variance_fractions["test"]
        record["omitted_target_policy"] = "training_mean"
        if val_r2 is not None:
            record["val_r2_retained_subspace"] = float(val_r2)
            record["val_r2_full_target_at_k"] = (
                variance_fractions["val"] * float(val_r2)
            )

    results, spectra = {}, {}
 
    # Closed-form ridge needs a (d+1) x (d+1) gram matrix. With raw input at
    # d = 20,421 that is 3.1 GB and about 100 TFLOP of solving per direction,
    # which dominates everything else in the sweep. It is also redundant here:
    # depth 0 IS a linear model, fitted by gradient descent with dropout
    # rather than an L2 penalty. Above the threshold, ridge is skipped and
    # depth 0 serves as the linear baseline.
    ridge_limit = int(config.get("ridge_max_input_dim", 4000))
    if X.shape[1] <= ridge_limit:
        prediction, val_score, alpha = fit_ridge(
            X.astype(np.float64), Y, index, list(config["ridge_alphas"]))
        record, spectrum = evaluate(prediction, Y[index["test"]], train_mean, ks, thresholds)
        add_target_space_metrics(record, val_score)
        record.update({"val_r2": val_score, "chosen_alpha": alpha,
                       "fit_status": "PASS"})
        results["ridge"] = record
        spectra["ridge"] = spectrum
    else:
        results["ridge_skipped"] = {"reason": "input_dim_above_ridge_max_input_dim",
                                    "input_dim": int(X.shape[1]),
                                    "ridge_max_input_dim": ridge_limit}
 
    for depth in config["depths"]:
        prediction, val_score, epochs, n_params, target_scale, learning_rate = fit_network(
            X, Y, index, config, width, int(depth), seed + 7 * int(depth), device_name)
        record, spectrum = evaluate(prediction, Y[index["test"]], train_mean, ks, thresholds)
        add_target_space_metrics(record, val_score)
        floor = float(config.get("unstable_r2_floor", -10.0))
        fit_status = "PASS" if (
            np.isfinite(record["r2"]) and np.isfinite(val_score)
            and record["r2"] >= floor and val_score >= floor
        ) else "UNSTABLE"
        record.update({"val_r2": val_score, "epochs_run": epochs,
                       "n_parameters": n_params, "hidden_width": width,
                       "target_global_scale": target_scale,
                       "learning_rate": learning_rate,
                       "fit_status": fit_status})
        results[f"depth{depth}"] = record
        spectra[f"depth{depth}"] = spectrum
 
    null = np.tile(train_mean, (len(index["test"]), 1))
    record, _ = evaluate(null, Y[index["test"]], train_mean, ks, thresholds)
    add_target_space_metrics(record, 0.0)
    results["null_train_mean"] = record
    return results, spectra
 
 
# ==================================================================== main
 
 
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--input-mode", default=None, choices=["pca", "raw"])
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-embedding", action="append", default=[],
        help="Restrict to pairs touching this embedding (repeatable; useful for pilots).",
    )
    parser.add_argument(
        "--panel-extension", action="store_true",
        help="Restrict to pairs touching panel_extension_new_embeddings from the YAML.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Recompute even when a matching PASS result exists.")
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--chunk-index", type=int, default=0)
    args = parser.parse_args()
 
    if not TORCH_AVAILABLE:
        print("torch is required for the depth sweep."); return 1
 
    started = time.time()
    config = yaml.safe_load(args.config.read_text())
    panel, manifest_path = load_manifest(config, args.config)
    aliases = id_aliases(config)
 
    splits_dir = Path(os.path.expanduser(config["splits_output"]))
    library = Path(os.path.expanduser(config["library_root"]))
    out = Path(os.path.expanduser(config["train_output"]))

    split_column = args.split_column or config.get("split_column", "split_family")
    pca_dir = resolve_pca_root(config, split_column)
    input_mode = args.input_mode or config.get("input_mode", "pca")
    tag = f"{split_column}__{input_mode}"
    pair_dir = out / "sweep_results" / tag
    pair_dir.mkdir(parents=True, exist_ok=True)
 
    requested = config.get("device", "auto")
    if requested in (None, "auto"):
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    elif str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("  WARNING: cuda requested but unavailable; using CPU")
        device_name = "cpu"
    else:
        device_name = requested
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
 
    print("=" * 70)
    print("DP v2: DEPTH SWEEP")
    print("=" * 70)
    print(f"split        : {split_column}")
    if input_mode == "raw":
        cap = config.get("raw_max_dim")
        note = f"  (untruncated; PCA fallback above {cap:,} dims)" if cap else "  (untruncated, no fallback)"
    else:
        note = "  (all cached components)"
    print(f"input mode   : {input_mode}{note}")
    print(f"depths       : {config['depths']}   (0 = linear map)")
    print(f"device       : {device_name}   threads: {os.environ.get('OMP_NUM_THREADS')}")
 
    split_path = splits_dir / "gene_splits.tsv"
    pca_metadata_path = pca_dir / "run_metadata.json"
    if not pca_metadata_path.is_file():
        raise FileNotFoundError(
            f"missing split-specific PCA metadata: {pca_metadata_path}. "
            f"Build the {split_column} cache first."
        )
    pca_run_metadata = json.loads(pca_metadata_path.read_text())
    if pca_run_metadata.get("split_column") != split_column:
        raise RuntimeError(
            f"PCA metadata says {pca_run_metadata.get('split_column')!r}, "
            f"but the requested split is {split_column!r}"
        )
    split_sha256 = sha256_file(split_path)
    if pca_run_metadata.get("split_table_sha256") != split_sha256:
        raise RuntimeError("PCA cache does not match the current gene_splits.tsv")
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    pca_run_signature = pca_run_metadata.get("pca_run_signature")
    if not pca_run_signature:
        raise RuntimeError(
            "PCA metadata lacks pca_run_signature; rebuild it with the repaired cache script"
        )
    signature_payload = {
        "script_sha256": script_sha256,
        "config_sha256": config_sha256,
        "split_table_sha256": split_sha256,
        "pca_run_signature": pca_run_signature,
        "split_column": split_column,
        "input_mode": input_mode,
        "device": str(device_name),
        "torch_version": torch.__version__,
        "threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()

    splits = pd.read_csv(split_path, sep="\t", dtype={"gene_id": str})
    if split_column not in splits.columns:
        raise ValueError(f"{split_path}: missing split column {split_column!r}")
    splits_by_gene = dict(zip(splits.gene_id, splits[split_column]))
 
    summary = pd.read_csv(pca_dir / "pca_summary.tsv", sep="\t")
    summary["embedding_id"] = summary.embedding_id.astype(str).map(
        lambda value: canonicalize_id(value, aliases)
    )
    if summary.embedding_id.duplicated().any():
        duplicate_ids = sorted(
            summary.loc[summary.embedding_id.duplicated(False), "embedding_id"].unique()
        )
        raise RuntimeError(f"PCA summary has duplicate canonical IDs: {duplicate_ids}")
    analysis_exclusions = set(config.get("analysis_exclusions") or [])
    selected_ids = panel.embedding_id.tolist()
    excluded_selected = sorted(set(selected_ids) & analysis_exclusions)
    if excluded_selected:
        raise ValueError(
            "manifest-selected embeddings cannot also be analysis_exclusions: "
            f"{excluded_selected}"
        )
    summary_ids = set(summary.embedding_id)
    missing_cache_rows = sorted(set(selected_ids) - summary_ids)
    if missing_cache_rows:
        raise RuntimeError(
            f"PCA summary is missing manifest embeddings: {missing_cache_rows}"
        )
    embedding_ids = sorted(selected_ids)
    print(f"manifest     : {manifest_path}")
    if analysis_exclusions:
        print(f"analysis exclusions: {sorted(analysis_exclusions)}")
    pairs = list(combinations(embedding_ids, 2))
    requested_values = list(args.include_embedding)
    if args.panel_extension:
        requested_values.extend(config.get("panel_extension_new_embeddings") or [])
        if not requested_values:
            raise ValueError(
                "--panel-extension was requested but panel_extension_new_embeddings is empty"
            )
    if requested_values:
        requested_embeddings = {
            canonicalize_id(value, aliases) for value in requested_values
        }
        unknown = requested_embeddings - set(embedding_ids)
        if unknown:
            raise ValueError(f"unknown --include-embedding values: {sorted(unknown)}")
        pairs = [p for p in pairs if requested_embeddings.intersection(p)]
        print(f"embedding filter: {len(pairs):,} pairs touch "
              f"{sorted(requested_embeddings)}")
 
    # ---- signal filter from the v1 run ----
    signal_path = config.get("signal_filter_table")
    threshold = float(config.get("signal_filter_r2", 0.20))
    if not signal_path:
        print(f"signal filter: disabled, analysing all {len(pairs):,} pairs")
    if signal_path:
        path = Path(os.path.expanduser(str(signal_path)))
        if path.is_file():
            previous = pd.read_csv(path, sep="\t")
            if "r2_max" in previous.columns:
                keep = previous[previous.r2_max >= threshold]
                wanted = {tuple(sorted((str(a), str(b))))
                          for a, b in zip(keep.embedding_x, keep.embedding_y)}
 
                # Filtering on v1 is circular for embeddings that v1 itself
                # truncated badly: they may have scored low because their
                # variance was cut off, which is precisely what v2 fixes.
                # Any pair touching one of these is kept regardless of score.
                always = set(config.get("signal_filter_always_include") or [])
                if config.get("signal_filter_include_capped", True):
                    summary_path = pca_dir / "pca_summary.tsv"
                    v1_summary = config.get("v1_pca_summary")
                    if v1_summary:
                        v1_path = Path(os.path.expanduser(str(v1_summary)))
                        if v1_path.is_file():
                            v1 = pd.read_csv(v1_path, sep="\t")
                            capped = v1[v1.variance_retained < 0.90].embedding_id
                            always |= set(capped.astype(str))
                before = len(pairs)
                pairs = [p for p in pairs
                         if tuple(sorted(p)) in wanted or p[0] in always or p[1] in always]
                rescued = len(pairs) - sum(
                    1 for p in pairs if tuple(sorted(p)) in wanted)
                print(f"signal filter: {len(pairs):,} of {before:,} pairs "
                      f"(v1 R2 >= {threshold})")
                if always:
                    print(f"               {rescued:,} kept despite low v1 score "
                          f"because {len(always)} embeddings were truncated in v1")
        else:
            print(f"  WARNING: signal table not found at {path}; using all pairs")
 
    widths_by_id = dict(zip(summary.embedding_id, summary.original_dimension)) \
        if "original_dimension" in summary.columns else {}
    intrinsic_by_id = dict(zip(summary.embedding_id, summary.n_components))
 
    def pair_cost(pair):
        """Rough relative cost: first-layer work dominates."""
        a, b = pair
        if input_mode == "raw":
            da = widths_by_id.get(a, intrinsic_by_id.get(a, 256))
            db = widths_by_id.get(b, intrinsic_by_id.get(b, 256))
        else:
            da = intrinsic_by_id.get(a, 256)
            db = intrinsic_by_id.get(b, 256)
        hidden = max(intrinsic_by_id.get(a, 256), intrinsic_by_id.get(b, 256))
        return (da + db) * hidden
 
    if args.n_chunks > 1:
        # Cost is dominated by the widest embedding in a pair: a 20,421-d
        # package costs roughly eighty times a 500-d one in the first layer.
        # Naive slicing would leave some array tasks with all the expensive
        # pairs. Sort by a cost proxy, then deal round-robin, so every chunk
        # gets one of the most expensive, one of the next, and so on.
        pairs = sorted(pairs, key=pair_cost, reverse=True)
        pairs = pairs[args.chunk_index::args.n_chunks]
        print(f"chunk        : {args.chunk_index + 1} of {args.n_chunks}, "
              f"{len(pairs):,} pairs (cost-balanced)")
 
    done = set()
    if not args.force:
        for path in pair_dir.glob("*.json"):
            try:
                existing = json.loads(path.read_text())
                if (existing.get("status") == "PASS"
                        and existing.get("run_signature") == run_signature):
                    done.add(path.stem)
            except json.JSONDecodeError:
                pass
    n_models = len(pairs) * 2 * len(config["depths"])
    print(f"pairs        : {len(pairs):,}   networks: {n_models:,}   complete: {len(done):,}")
 
    source = Source(pca_dir, library, input_mode,
                    config.get("array_key", "embeddings"),
                    config.get("id_column", "ensembl_gene_id"),
                    float(config["output_variance_target"]),
                    int(config.get("cache_capacity", 4)),
                    config.get("raw_max_dim"),
                    split_column,
                    float(config.get("min_target_variance_retained", 0.0)),
                    bool(config.get("compute_full_target_r2", True)),
                    aliases=aliases)
    widths = list(config["hidden_width_options"])
    ks = list(config["retrieval_ks"])
    thresholds = list(config["spectrum_thresholds"])
    base_seed = int(config["seed"])
 
    def prepare(a, b):
        (lookup_a, in_a, out_a, nin_a, nout_a, ret_a, raw_a, intr_a,
         eligible_a, cache_signature_a, full_a) = source.get(a)
        (lookup_b, in_b, out_b, nin_b, nout_b, ret_b, raw_b, intr_b,
         eligible_b, cache_signature_b, full_b) = source.get(b)
        shared = sorted(g for g in lookup_a if g in lookup_b and g in splits_by_gene)
        rows_a = np.fromiter((lookup_a[g] for g in shared), dtype=np.int64, count=len(shared))
        rows_b = np.fromiter((lookup_b[g] for g in shared), dtype=np.int64, count=len(shared))
        labels = np.array([splits_by_gene[g] for g in shared])
        index = {n: np.flatnonzero(labels == n) for n in ("train", "val", "test")}
        # float32 throughout: it halves peak memory and the network casts to
        # float32 regardless. Ridge casts up locally, where the matrices are
        # small enough for it to matter.
        return (np.ascontiguousarray(in_a[rows_a], dtype=np.float32),
                np.ascontiguousarray(out_a[rows_a], dtype=np.float64),
                np.ascontiguousarray(in_b[rows_b], dtype=np.float32),
                np.ascontiguousarray(out_b[rows_b], dtype=np.float64),
                (np.ascontiguousarray(full_a[rows_a], dtype=np.float32)
                 if full_a is not None else None),
                (np.ascontiguousarray(full_b[rows_b], dtype=np.float32)
                 if full_b is not None else None),
                index, np.array(shared),
                {"n_in_x": nin_a, "n_out_x": nout_a, "variance_retained_x": ret_a,
                 "n_in_y": nin_b, "n_out_y": nout_b, "variance_retained_y": ret_b,
                 "raw_input_x": bool(raw_a), "raw_input_y": bool(raw_b),
                 "n_intrinsic_x": intr_a, "n_intrinsic_y": intr_b,
                 "target_eligible_x": bool(eligible_a),
                 "target_eligible_y": bool(eligible_b),
                 "pca_cache_signature_x": cache_signature_a,
                 "pca_cache_signature_y": cache_signature_b})
 
    if args.estimate:
        # Sample across the COST distribution, including the most expensive
        # pair. Sampling by list position would miss probe_blast and
        # probe_hmmer entirely and understate the total badly.
        by_cost = sorted(pairs, key=pair_cost, reverse=True)
        picks = [0, len(by_cost) // 8, len(by_cost) // 3,
                 len(by_cost) // 2, (3 * len(by_cost)) // 4, len(by_cost) - 1]
        probe = [by_cost[i] for i in sorted(set(picks)) if i < len(by_cost)]
        total_cost = sum(pair_cost(p) for p in pairs)
        print(f"\nTiming {len(probe)} pairs spanning the cost range "
              f"(all depths, both directions):\n")
        times, probe_costs = [], []
        for a, b in probe:
            t0 = time.time()
            Xin, Xout, Yin, Yout, Xfull, Yfull, index, genes, meta = prepare(a, b)
            width = choose_width(max(meta["n_intrinsic_x"], meta["n_intrinsic_y"]),
                                 max(meta["n_out_x"], meta["n_out_y"]), widths)
            seed = stable_seed(base_seed, a, b, split_column, input_mode)
            if meta["target_eligible_y"]:
                run_direction(Xin, Yout, index, config, width, seed,
                              device_name, ks, thresholds, Yfull)
            if meta["target_eligible_x"]:
                run_direction(Yin, Xout, index, config, width, seed + 1,
                              device_name, ks, thresholds, Xfull)
            dt = time.time() - t0
            times.append(dt)
            probe_costs.append(pair_cost((a, b)))
            flag = ""
            if input_mode == "raw" and not (meta["raw_input_x"] and meta["raw_input_y"]):
                flag = "  [PCA fallback]"
            print(f"  {dt:>7.1f}s  width={width:<4} "
                  f"in={meta['n_in_x']}/{meta['n_in_y']} "
                  f"intr={meta['n_intrinsic_x']}/{meta['n_intrinsic_y']} "
                  f"out={meta['n_out_x']}/{meta['n_out_y']}  "
                  f"{a[:20]} <-> {b[:20]}{flag}")
        # Extrapolate by total cost rather than by pair count, since cost
        # varies about a hundredfold across the panel.
        seconds_per_unit = float(np.sum(times) / np.sum(probe_costs))
        projected = seconds_per_unit * total_cost
        print(f"\n  cost-weighted rate: {seconds_per_unit:.2e} s per cost unit")
        print(f"  projected total: {projected / 3600:.0f} h single process, "
              f"this split")
        print(f"  both splits    : {2 * projected / 3600:.0f} h")
        print(f"\n  as a SLURM array (makespan ~1.3x ideal with cost balancing):")
        for n in (20, 40, 80, 120):
            print(f"    {n:>4} tasks: {projected / 3600 / n * 1.3:>6.1f} h wall clock per split")
        if input_mode == "raw":
            print("\n  NOTE: raw-input cost scales with the widest embedding in each")
            print("  pair. The probe_blast and probe_hmmer pairs (20,421 dims) are far")
            print("  more expensive than the rest; if the projection is unacceptable,")
            print("  set raw_max_dim in the config so those fall back to components.")
        return 0
 
    n_done = n_fail = 0
    last = time.time()
    for position, (a, b) in enumerate(pairs, 1):
        stem = f"{a}__VS__{b}"
        if len(stem) > 190:
            stem = hashlib.sha256(stem.encode()).hexdigest()[:40]
        if stem in done:
            continue
        target = pair_dir / f"{stem}.json"
        try:
            t0 = time.time()
            Xin, Xout, Yin, Yout, Xfull, Yfull, index, genes, meta = prepare(a, b)
            width = choose_width(max(meta["n_intrinsic_x"], meta["n_intrinsic_y"]),
                                 max(meta["n_out_x"], meta["n_out_y"]), widths)
            seed = stable_seed(base_seed, a, b, split_column, input_mode)
            if meta["target_eligible_y"]:
                forward, spectra_f = run_direction(
                    Xin, Yout, index, config, width, seed, device_name, ks,
                    thresholds, Yfull
                )
            else:
                forward, spectra_f = ({
                    "target_skipped": {
                        "reason": "target_variance_below_minimum",
                        "variance_retained": meta["variance_retained_y"],
                        "minimum": float(config.get("min_target_variance_retained", 0.0)),
                    }
                }, {})
            if meta["target_eligible_x"]:
                backward, spectra_b = run_direction(
                    Yin, Xout, index, config, width, seed + 1, device_name, ks,
                    thresholds, Xfull
                )
            else:
                backward, spectra_b = ({
                    "target_skipped": {
                        "reason": "target_variance_below_minimum",
                        "variance_retained": meta["variance_retained_x"],
                        "minimum": float(config.get("min_target_variance_retained", 0.0)),
                    }
                }, {})
            payload = {
                "pair_id": f"{a}__VS__{b}", "embedding_x": a, "embedding_y": b,
                "split_column": split_column, "input_mode": input_mode,
                "hidden_width": width, **meta,
                "n_shared_genes": int(len(genes)),
                "n_train": int(len(index["train"])), "n_val": int(len(index["val"])),
                "n_test": int(len(index["test"])),
                "runtime_seconds": round(time.time() - t0, 2),
                "seed": seed,
                "run_signature": run_signature,
                "script_sha256": script_sha256,
                "config_sha256": config_sha256,
                "split_table_sha256": split_sha256,
                "pca_run_signature": pca_run_signature,
                "manifest_file": str(manifest_path),
                "panel_sha256": panel_sha256(panel),
                "status": "PASS",
                "x_to_y": forward, "y_to_x": backward,
            }
            if config.get("save_spectra", True):
                atomic_savez_compressed(
                    pair_dir / f"{stem}__spectra.npz",
                    **{f"x_to_y__{k}": v for k, v in spectra_f.items()},
                    **{f"y_to_x__{k}": v for k, v in spectra_b.items()})
            # JSON is the completion marker and is written last.
            atomic_write_text(target, json.dumps(payload, indent=2))
            n_done += 1
        except Exception as error:  # noqa: BLE001
            n_fail += 1
            atomic_write_text(target, json.dumps({
                "pair_id": f"{a}__VS__{b}", "embedding_x": a, "embedding_y": b,
                "split_column": split_column, "input_mode": input_mode,
                "run_signature": run_signature,
                "script_sha256": script_sha256,
                "config_sha256": config_sha256,
                "split_table_sha256": split_sha256,
                "pca_run_signature": pca_run_signature,
                "status": "FAIL", "error_type": type(error).__name__,
                "error": str(error)}, indent=2))
 
        if time.time() - last > 120:
            elapsed = time.time() - started
            rate = n_done / elapsed if elapsed else 0
            eta = (len(pairs) - position) / rate / 3600 if rate else float("inf")
            print(f"  [{position:>5,}/{len(pairs):,}] done={n_done:,} fail={n_fail:,} "
                  f"eta={eta:.1f}h", flush=True)
            last = time.time()
        if args.limit and n_done >= args.limit:
            print(f"\n--limit {args.limit} reached."); break
 
    elapsed = time.time() - started
    metadata_dir = out / "run_metadata" / tag
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"chunk_{args.chunk_index:05d}.json"
    atomic_write_text(metadata_path, json.dumps({
        "script": Path(__file__).name,
        "script_sha256": script_sha256,
        "config_sha256": config_sha256,
        "split_table_sha256": split_sha256,
        "pca_run_signature": pca_run_signature,
        "run_signature": run_signature,
        "manifest_file": str(manifest_path),
        "panel_sha256": panel_sha256(panel),
        "config": config, "split_column": split_column, "input_mode": input_mode,
        "device": device_name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(elapsed, 2), "python_version": sys.version,
        "platform": platform.platform(),
        "chunk_index": args.chunk_index, "n_chunks": args.n_chunks,
        "include_embedding": sorted(set(requested_values)),
        "panel_extension": bool(args.panel_extension),
        "n_pairs": len(pairs), "n_computed": n_done, "n_failed": n_fail,
        "status": "PASS" if n_fail == 0 else "PASS_WITH_FAILURES",
    }, indent=2))
    print(f"\nDONE in {elapsed/3600:.2f}h  computed={n_done:,}  failed={n_fail:,}")
    print(f"Results: {pair_dir}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
