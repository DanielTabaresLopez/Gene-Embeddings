from pathlib import Path
from collections import defaultdict
import gzip
import json
import re
import tempfile

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.preprocessing import StandardScaler


home = Path.home()

MASTER_PATH = home / "metadata/master_gene_table_v1_1_enriched.csv"

RAW_DIR = home / "data/raw_embeddings/opa2vec_original_try/go_human_inputs"
OBO_PATH = RAW_DIR / "go-basic.obo"
GAF_PATH = RAW_DIR / "goa_human.gaf.gz"

OUT_STEM = "opa2vec_goa_human_v1_1"
DISPLAY_NAME = "OPA2Vec-GOA-human-v1.1"

OUT_DIR = home / "data/processed_embeddings" / OUT_STEM
REPORT_DIR = home / "reports/mapping_reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / f"{OUT_STEM}_embeddings.npz"
OUT_GENES = OUT_DIR / f"{OUT_STEM}_genes.tsv"
OUT_SOURCE_ENTITIES = OUT_DIR / f"{OUT_STEM}_source_uniprot_entities.tsv"
OUT_META = OUT_DIR / f"{OUT_STEM}_metadata.json"

REPORT_TXT = REPORT_DIR / f"{OUT_STEM}_mapping_report.txt"
UNMAPPED_TSV = REPORT_DIR / f"{OUT_STEM}_unmapped_source_identifiers.tsv"
AMBIGUOUS_TSV = REPORT_DIR / f"{OUT_STEM}_ambiguous_source_identifiers.tsv"
DUPLICATES_TSV = REPORT_DIR / f"{OUT_STEM}_duplicate_final_mappings.tsv"

SEED = 42
VECTOR_SIZE = 200
WINDOW = 10
MIN_COUNT = 1
EPOCHS = 10
WORKERS = 8


def split_values(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return []
    return [t.strip() for t in re.split(r"[|,;\s]+", s) if t.strip()]


def clean_uniprot(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    s = s.split("|")[-1]
    s = s.split("-")[0]
    return s.upper()


def go_token(go_id):
    return "GO_" + go_id.replace("GO:", "")


def entity_token(acc):
    return "UNIPROT_" + clean_uniprot(acc)


def text_tokens(s):
    s = str(s).lower()
    return [t for t in re.findall(r"[a-z0-9]+", s) if len(t) >= 2]


def first_existing(row, cols):
    for c in cols:
        if c in row.index and str(row[c]).strip():
            return str(row[c]).strip()
    return ""


def parse_go_obo(path):
    terms = {}
    current = None

    def finish(term):
        if not term:
            return
        if term.get("id") and not term.get("is_obsolete", False):
            terms[term["id"]] = term

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")

            if line == "[Term]":
                finish(current)
                current = {
                    "parents": set(),
                    "synonyms": [],
                    "name": "",
                    "namespace": "",
                    "definition": "",
                    "is_obsolete": False,
                }
                continue

            if line.startswith("["):
                finish(current)
                current = None
                continue

            if current is None:
                continue

            if line.startswith("id: "):
                current["id"] = line.split("id: ", 1)[1].strip()
            elif line.startswith("name: "):
                current["name"] = line.split("name: ", 1)[1].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line.split("namespace: ", 1)[1].strip()
            elif line.startswith("def: "):
                current["definition"] = line.split("def: ", 1)[1].strip()
            elif line.startswith("synonym: "):
                m = re.search(r'"([^"]+)"', line)
                if m:
                    current["synonyms"].append(m.group(1))
            elif line.startswith("is_a: "):
                parent = line.split("is_a: ", 1)[1].split()[0]
                current["parents"].add(parent)
            elif line.startswith("is_obsolete: true"):
                current["is_obsolete"] = True

    finish(current)
    return terms


def compute_ancestors(terms):
    memo = {}

    def dfs(go_id):
        if go_id in memo:
            return memo[go_id]
        out = set()
        for p in terms.get(go_id, {}).get("parents", []):
            if p in terms:
                out.add(p)
                out.update(dfs(p))
        memo[go_id] = out
        return out

    for go_id in terms:
        dfs(go_id)

    return memo


def parse_gaf(path, valid_go_ids):
    acc_to_go = defaultdict(set)
    acc_to_symbols = defaultdict(set)
    evidence_counts = defaultdict(int)

    total = 0
    kept = 0
    skipped_not = 0
    skipped_bad = 0

    with gzip.open(path, "rt") as f:
        for line in f:
            if not line or line.startswith("!"):
                continue

            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                skipped_bad += 1
                continue

            db = parts[0]
            acc = clean_uniprot(parts[1])
            symbol = parts[2].strip()
            qualifier = parts[3]
            go_id = parts[4]
            evidence = parts[6]

            if "NOT" in qualifier.split("|"):
                skipped_not += 1
                continue

            if db != "UniProtKB" or not acc or go_id not in valid_go_ids:
                skipped_bad += 1
                continue

            acc_to_go[acc].add(go_id)
            if symbol:
                acc_to_symbols[acc].add(symbol)
            evidence_counts[evidence] += 1
            kept += 1

    stats = {
        "gaf_total_annotation_rows": total,
        "gaf_kept_non_NOT_uniprot_go_rows": kept,
        "gaf_skipped_NOT_rows": skipped_not,
        "gaf_skipped_bad_or_nonuniprot_rows": skipped_bad,
        "gaf_unique_uniprot_entities": len(acc_to_go),
        "gaf_unique_direct_go_terms": len(set(g for gs in acc_to_go.values() for g in gs)),
        "gaf_evidence_counts": dict(sorted(evidence_counts.items())),
    }

    return acc_to_go, acc_to_symbols, stats


def build_master_uniprot_map(master):
    uniprot_to_rows = defaultdict(set)
    cols = ["uniprot_id", "hgnc_uniprot_ids", "map_uniprot_ids_all"]

    for i, row in master.iterrows():
        for c in cols:
            if c not in master.columns:
                continue
            for tok in split_values(row[c]):
                acc = clean_uniprot(tok)
                if acc:
                    uniprot_to_rows[acc].add(i)

    return uniprot_to_rows


def map_source_acc(acc, uniprot_to_rows):
    rows = set(uniprot_to_rows.get(clean_uniprot(acc), set()))
    if len(rows) == 1:
        return next(iter(rows)), "uniprot_exact", ""
    if len(rows) > 1:
        return None, "uniprot_exact", "maps_to_multiple_master_rows"
    return None, "", "unmapped"


def write_corpus(corpus_path, terms, ancestors, acc_to_go, acc_to_symbols):
    n_lines = 0

    with open(corpus_path, "w") as out:
        # Entity association sentences: UniProt entity + direct GO + ancestor GO terms.
        for acc, direct_gos in sorted(acc_to_go.items()):
            all_gos = set(direct_gos)
            for go_id in direct_gos:
                all_gos.update(ancestors.get(go_id, set()))

            toks = [entity_token(acc)]
            toks += [go_token(g) for g in sorted(all_gos)]

            # Include gene symbol as lightweight metadata token.
            for sym in sorted(acc_to_symbols.get(acc, [])):
                toks.append("symbol_" + re.sub(r"[^A-Za-z0-9]+", "_", sym).lower())

            out.write(" ".join(toks) + "\n")
            n_lines += 1

        # Direct association pair sentences.
        for acc, direct_gos in sorted(acc_to_go.items()):
            for go_id in sorted(direct_gos):
                path_terms = [go_id] + sorted(ancestors.get(go_id, set()))
                toks = [entity_token(acc)] + [go_token(g) for g in path_terms]
                out.write(" ".join(toks) + "\n")
                n_lines += 1

        # GO metadata and graph sentences.
        for go_id, term in sorted(terms.items()):
            toks = [go_token(go_id)]
            if term.get("namespace"):
                toks.append("namespace_" + term["namespace"].replace("_", ""))

            toks += text_tokens(term.get("name", ""))
            toks += text_tokens(term.get("definition", ""))

            for syn in term.get("synonyms", []):
                toks += text_tokens(syn)

            for parent in sorted(term.get("parents", [])):
                if parent in terms:
                    toks.append(go_token(parent))

            if len(toks) > 1:
                out.write(" ".join(toks) + "\n")
                n_lines += 1

            for parent in sorted(term.get("parents", [])):
                if parent in terms:
                    out.write(f"{go_token(go_id)} {go_token(parent)}\n")
                    n_lines += 1

    return n_lines


def standardize(X):
    return StandardScaler().fit_transform(X.astype(np.float32)).astype(np.float32)


def main():
    print("Loading master table...")
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    symbol_col = "gene_symbol" if "gene_symbol" in master.columns else "hgnc_approved_symbol"

    print("Parsing GO ontology...")
    terms = parse_go_obo(OBO_PATH)
    ancestors = compute_ancestors(terms)
    print("GO terms:", len(terms))

    print("Parsing GOA human annotations...")
    acc_to_go, acc_to_symbols, gaf_stats = parse_gaf(GAF_PATH, set(terms))
    print("UniProt entities with GO annotations:", len(acc_to_go))

    print("Building corpus...")
    corpus_path = OUT_DIR / f"{OUT_STEM}_training_corpus.txt"
    n_corpus_lines = write_corpus(corpus_path, terms, ancestors, acc_to_go, acc_to_symbols)
    print("Corpus lines:", n_corpus_lines)
    print("Corpus:", corpus_path)

    print("Training Word2Vec / OPA2Vec-derived model...")
    model = Word2Vec(
        sentences=LineSentence(str(corpus_path)),
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,
        workers=WORKERS,
        epochs=EPOCHS,
        seed=SEED,
    )

    print("Mapping UniProt entities to master genes...")
    uniprot_to_rows = build_master_uniprot_map(master)

    mapped_records = []
    unmapped_records = []
    ambiguous_records = []

    for acc in sorted(acc_to_go):
        tok = entity_token(acc)
        if tok not in model.wv:
            unmapped_records.append({
                "source_identifier": acc,
                "reason": "entity_token_not_in_word2vec_model",
                "go_term_count": len(acc_to_go[acc]),
            })
            continue

        master_i, method, reason = map_source_acc(acc, uniprot_to_rows)

        if master_i is not None:
            mapped_records.append({
                "source_identifier": acc,
                "source_token": tok,
                "master_row_index": master_i,
                "mapping_method": method,
                "go_term_count_direct": len(acc_to_go[acc]),
                "go_terms_direct_examples": ";".join(sorted(acc_to_go[acc])[:20]),
                "symbols_examples": ";".join(sorted(acc_to_symbols.get(acc, []))[:10]),
            })
        elif reason == "maps_to_multiple_master_rows":
            ambiguous_records.append({
                "source_identifier": acc,
                "reason": reason,
                "mapping_method": method,
                "go_term_count_direct": len(acc_to_go[acc]),
                "go_terms_direct_examples": ";".join(sorted(acc_to_go[acc])[:20]),
                "symbols_examples": ";".join(sorted(acc_to_symbols.get(acc, []))[:10]),
            })
        else:
            unmapped_records.append({
                "source_identifier": acc,
                "reason": "no_unique_mapping_to_master",
                "go_term_count_direct": len(acc_to_go[acc]),
                "go_terms_direct_examples": ";".join(sorted(acc_to_go[acc])[:20]),
                "symbols_examples": ";".join(sorted(acc_to_symbols.get(acc, []))[:10]),
            })

    master_to_source = defaultdict(list)
    for rec in mapped_records:
        master_to_source[rec["master_row_index"]].append(rec)

    final_master_indices = sorted(master_to_source.keys())

    X_final = []
    gene_rows = []
    duplicate_rows = []

    for master_i in final_master_indices:
        source_items = master_to_source[master_i]
        accs = [r["source_identifier"] for r in source_items]
        toks = [r["source_token"] for r in source_items]
        methods = sorted(set(r["mapping_method"] for r in source_items))

        row = master.loc[master_i]
        X = np.vstack([model.wv[tok] for tok in toks]).astype(np.float32)
        X_final.append(X.mean(axis=0).astype(np.float32))

        if len(accs) > 1:
            duplicate_rows.append({
                "ensembl_gene_id": row["ensembl_gene_id"],
                "gene_symbol": row.get(symbol_col, ""),
                "source_key_count": len(accs),
                "source_keys": ";".join(accs),
                "mapping_methods": ";".join(methods),
                "action": "averaged_multiple_UniProt_entities_mapping_to_same_master_gene",
            })

        gene_rows.append({
            "ensembl_gene_id": row["ensembl_gene_id"],
            "gene_symbol": row.get(symbol_col, ""),
            "gene_type": row.get("gene_type", ""),
            "entrez_id": first_existing(row, ["entrez_id", "hgnc_entrez_id"]),
            "uniprot_id": first_existing(row, ["uniprot_id", "hgnc_uniprot_ids"]),
            "canonical_transcript_id": row.get("canonical_transcript_id", ""),
            "canonical_protein_id": row.get("canonical_protein_id", ""),
            "source_embedding": DISPLAY_NAME,
            "source_identifier_type": "UniProt accession",
            "source_key_count": len(accs),
            "source_keys_examples": ";".join(accs[:10]),
            "mapping_methods": ";".join(methods),
        })

    X_final = np.vstack(X_final).astype(np.float32)
    X_final = standardize(X_final)

    genes_df = pd.DataFrame(gene_rows)

    print("Final X:", X_final.shape)

    np.savez_compressed(
        OUT_NPZ,
        X=X_final,
        ensembl_gene_id=genes_df["ensembl_gene_id"].to_numpy(dtype=str),
        gene_symbol=genes_df["gene_symbol"].to_numpy(dtype=str),
    )

    genes_df.to_csv(OUT_GENES, sep="\t", index=False)

    source_df = pd.DataFrame({
        "uniprot_accession": sorted(acc_to_go),
        "source_token": [entity_token(acc) for acc in sorted(acc_to_go)],
        "symbols": [";".join(sorted(acc_to_symbols.get(acc, []))) for acc in sorted(acc_to_go)],
        "direct_go_term_count": [len(acc_to_go[acc]) for acc in sorted(acc_to_go)],
        "direct_go_terms_examples": [";".join(sorted(acc_to_go[acc])[:20]) for acc in sorted(acc_to_go)],
    })
    source_df.to_csv(OUT_SOURCE_ENTITIES, sep="\t", index=False)

    pd.DataFrame(unmapped_records).to_csv(UNMAPPED_TSV, sep="\t", index=False)
    pd.DataFrame(ambiguous_records).to_csv(AMBIGUOUS_TSV, sep="\t", index=False)
    pd.DataFrame(duplicate_rows).to_csv(DUPLICATES_TSV, sep="\t", index=False)

    metadata = {
        "embedding_name": DISPLAY_NAME,
        "source_embedding": DISPLAY_NAME,
        "modality": "Gene Ontology / ontology-annotation semantic embedding",
        "source_identifier_type": "UniProt accession",
        "download_source": "Gene Ontology go-basic.obo and GOA human GAF",
        "original_model_or_method": "OPA2Vec",
        "embedding_generated_by": "Daniel locally using an OPA2Vec-derived Python 3 workflow",
        "exact_original_reproduction": False,
        "exact_original_reproduction_note": "The official OPA2Vec repository requires old Groovy/Python2-style dependencies and a bundled old gensim/PubMed model object. The workstation lacked Groovy/Python2, so this version regenerates an OPA2Vec-derived embedding from GO ontology metadata, GO graph relations, and GOA human UniProt-GO associations using gensim Word2Vec in Python 3.",
        "input_data_or_sequence_source": "GO basic ontology plus GOA human annotations",
        "source_files": {
            "go_basic_obo": str(OBO_PATH),
            "goa_human_gaf": str(GAF_PATH),
            "training_corpus": str(corpus_path),
        },
        "word2vec": {
            "implementation": "gensim.models.Word2Vec",
            "vector_size": VECTOR_SIZE,
            "window": WINDOW,
            "min_count": MIN_COUNT,
            "sg": 1,
            "epochs": EPOCHS,
            "workers": WORKERS,
            "seed": SEED,
        },
        "pipeline": [
            "Parse GO basic ontology terms, names, namespaces, definitions, synonyms, and is_a parents",
            "Parse GOA human GAF annotations",
            "Exclude NOT-qualified annotations",
            "Build UniProt-to-GO direct association sentences",
            "Propagate GO ancestors through the GO is_a graph",
            "Add GO metadata and GO parent-child graph sentences",
            "Train skip-gram Word2Vec over the ontology/association corpus",
            "Extract UniProt entity vectors",
            "Map UniProt accessions to the v1.1 master gene table",
            "Average multiple UniProt entities mapping to the same master gene",
            "Standardize final gene dimensions",
        ],
        "go_counts": {
            "go_terms_parsed": int(len(terms)),
            "corpus_lines": int(n_corpus_lines),
            **gaf_stats,
        },
        "counts": {
            "source_uniprot_entities": int(len(acc_to_go)),
            "mapped_source_entities_before_duplicate_collapse": int(len(mapped_records)),
            "unmapped_source_entities": int(len(unmapped_records)),
            "ambiguous_source_entities": int(len(ambiguous_records)),
            "final_mapped_genes": int(X_final.shape[0]),
            "final_dimensions": int(X_final.shape[1]),
            "master_rows": int(len(master)),
            "duplicate_final_mappings_averaged": int(len(duplicate_rows)),
        },
        "coverage": {
            "source_entity_mapping_fraction": float(len(mapped_records) / len(acc_to_go)) if acc_to_go else 0.0,
            "master_gene_coverage_fraction": float(X_final.shape[0] / len(master)),
        },
        "outputs": {
            "npz": str(OUT_NPZ),
            "genes_tsv": str(OUT_GENES),
            "source_entities_tsv": str(OUT_SOURCE_ENTITIES),
            "metadata_json": str(OUT_META),
            "mapping_report": str(REPORT_TXT),
            "unmapped_tsv": str(UNMAPPED_TSV),
            "ambiguous_tsv": str(AMBIGUOUS_TSV),
            "duplicates_tsv": str(DUPLICATES_TSV),
        },
    }

    with open(OUT_META, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(REPORT_TXT, "w") as f:
        f.write(f"{DISPLAY_NAME} mapping report\n")
        f.write("=" * (len(DISPLAY_NAME) + 16) + "\n\n")
        f.write("Provenance note:\n")
        f.write("This is a locally generated OPA2Vec-derived GOA-human embedding.\n")
        f.write("It is not an exact original OPA2Vec run because the official repo depends on old Groovy/Python2-era tooling and the workstation lacked Groovy/Python2.\n")
        f.write("The embedding uses GO ontology metadata, GO is_a relations, GO ancestor propagation, and GOA human UniProt-GO associations.\n\n")

        f.write("GO / GOA counts:\n")
        for k, v in metadata["go_counts"].items():
            f.write(f"{k}: {v}\n")

        f.write("\nWord2Vec:\n")
        for k, v in metadata["word2vec"].items():
            f.write(f"{k}: {v}\n")

        f.write("\nCounts:\n")
        for k, v in metadata["counts"].items():
            f.write(f"{k}: {v}\n")

        f.write("\nCoverage:\n")
        for k, v in metadata["coverage"].items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\nOutputs:\n")
        for k, v in metadata["outputs"].items():
            f.write(f"{k}: {v}\n")

    print("\nDone")
    print("X_final:", X_final.shape)
    print("genes:", len(genes_df))
    print("Report:", REPORT_TXT)


if __name__ == "__main__":
    main()
