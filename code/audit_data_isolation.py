#!/usr/bin/env python3
"""Offline, static evidence audit for the SST-2 canonical pool and train splits."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable

import numpy as np

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

DEFAULT_PARQUET = str(Path(__file__).resolve().parent.parent / "data" / "sst2_validation.parquet")
DEFAULT_POOL = str(Path(__file__).resolve().parent.parent / "data" / "sst2_pool.json")
DEFAULT_PHASE1 = str(Path(__file__).resolve().parent / "cisb_phase1_discovery.py")
DEFAULT_PHASE2 = str(Path(__file__).resolve().parent / "cisb_phase2_scoring.py")
TRAIN_SEEDS = (42, 7, 123)
SEARCH_SAMPLES = 2000
TRAIN_SAMPLES = 5000
ORIGINAL_ID_COLUMN = "__audit_original_row_id"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ordered_hash(values: Iterable[Any]) -> str:
    payload = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    return record


def builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [builtin(item) for item in value]
    return value


def text_ref(text: str, **metadata: Any) -> dict[str, Any]:
    return {"text": text, "text_sha256": sha256_text(text), **builtin(metadata)}


def public_ref(record: dict[str, Any]) -> dict[str, Any]:
    return {key: builtin(value) for key, value in record.items() if key != "text"}


def exact_intersection(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    right_by_text: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for right_index, record in enumerate(right):
        right_by_text.setdefault(record["text"], []).append((right_index, record))
    matches = []
    matched_left = set()
    matched_right = set()
    for left_index, left_record in enumerate(left):
        for right_index, right_record in right_by_text.get(left_record["text"], []):
            matches.append(
                {
                    "text_sha256": left_record["text_sha256"],
                    "left": public_ref(left_record),
                    "right": public_ref(right_record),
                }
            )
            matched_left.add(left_index)
            matched_right.add(right_index)
    return {
        "comparison": "exact Python string equality; hashes are identifiers, not collision-only evidence",
        "match_count": len(matches),
        "unique_matched_text_count": len({match["text_sha256"] for match in matches}),
        "unique_left_record_count": len(matched_left),
        "unique_right_record_count": len(matched_right),
        "matches": matches,
    }


def cache_file_record(dataset: Any) -> dict[str, Any]:
    raw = getattr(dataset, "cache_files", None)
    if raw is None:
        return {"available": False, "files": []}
    files = []
    for item in raw:
        filename = item.get("filename") if isinstance(item, dict) else str(item)
        entry: dict[str, Any] = {"filename": filename}
        if filename:
            path = Path(filename).expanduser()
            entry["exists"] = path.is_file()
            if path.is_file():
                entry["size_bytes"] = path.stat().st_size
        files.append(entry)
    return {"available": True, "files": files}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_string_list(pool: dict[str, Any], key: str) -> list[str]:
    value = pool[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"pool field {key!r} must be a list of strings")
    return value


def require_id_list(pool: dict[str, Any], key: str) -> list[int]:
    value = pool[key]
    if not isinstance(value, list) or not all(
        isinstance(item, (int, np.integer)) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"pool field {key!r} must be a list of integer row IDs")
    return [int(item) for item in value]


def pool_trigger(text: str, trigger: str, position: str) -> str:
    if position == "prefix":
        return f"{trigger} {text}"
    if position == "end":
        return f"{text} {trigger}"
    if position == "middle":
        words = text.split()
        words.insert(max(1, len(words) // 2), trigger)
        return " ".join(words)
    raise ValueError(f"unsupported pool trigger position: {position!r}")


def audit_source_and_pool(
    parquet_path: Path, pool_path: Path, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    public: dict[str, Any] = {
        "source_parquet": file_record(parquet_path),
        "canonical_pool": file_record(pool_path),
        "seed": seed,
    }
    if not parquet_path.is_file() or not pool_path.is_file():
        raise FileNotFoundError("source parquet and canonical pool must both exist")

    pd = importlib.import_module("pandas")
    public["source_parquet"]["pandas_version_used"] = pd.__version__
    frame = pd.read_parquet(parquet_path)
    with pool_path.open("r", encoding="utf-8") as handle:
        pool = json.load(handle)

    required = {
        "cal_ids",
        "test_ids",
        "cal_clean",
        "cal_labels",
        "test_clean",
        "test_labels",
        "test_trig",
        "trigger",
        "position",
    }
    missing = sorted(required.difference(pool))
    if missing:
        raise ValueError(f"pool is missing required fields: {missing}")
    if "sentence" not in frame or "label" not in frame:
        raise ValueError("source parquet must contain sentence and label columns")

    cal_ids = require_id_list(pool, "cal_ids")
    test_ids = require_id_list(pool, "test_ids")
    cal_clean = require_string_list(pool, "cal_clean")
    test_clean = require_string_list(pool, "test_clean")
    test_trig = require_string_list(pool, "test_trig")
    cal_labels = [int(item) for item in pool["cal_labels"]]
    test_labels = [int(item) for item in pool["test_labels"]]
    lengths = {
        "cal_ids": len(cal_ids),
        "test_ids": len(test_ids),
        "cal_clean": len(cal_clean),
        "cal_labels": len(cal_labels),
        "test_clean": len(test_clean),
        "test_labels": len(test_labels),
        "test_trig": len(test_trig),
    }
    lengths_match = (
        len(cal_ids) == len(cal_clean) == len(cal_labels)
        and len(test_ids) == len(test_clean) == len(test_labels) == len(test_trig)
    )
    if not lengths_match:
        public["pool_field_lengths"] = lengths
        raise ValueError(f"pool field lengths disagree: {lengths}")

    source_texts = frame["sentence"].astype(str).tolist()
    source_labels = frame["label"].astype(int).tolist()
    source_dataset_idx = frame["idx"].tolist() if "idx" in frame else None
    public["source_parquet"].update(
        {
            "row_count": len(frame),
            "columns": [str(column) for column in frame.columns],
            "ordered_sentence_sha256": ordered_hash(source_texts),
            "text_field": "sentence",
            "label_field": "label",
            "row_id_convention": "zero-based parquet row position, not a pandas index label",
        }
    )

    cal_set, test_set = set(cal_ids), set(test_ids)
    all_ids = cal_ids + test_ids
    in_bounds = all(0 <= item < len(frame) for item in all_ids)
    permutation = np.random.RandomState(seed).permutation(len(frame))
    expected_cal = [int(item) for item in permutation[:100]]
    expected_test = [int(item) for item in permutation[100:300]]
    split_proof = {
        "cal_ids": cal_ids,
        "test_ids": test_ids,
        "counts": lengths,
        "cal_unique_id_count": len(cal_set),
        "test_unique_id_count": len(test_set),
        "cal_duplicate_ids": sorted(item for item in cal_set if cal_ids.count(item) > 1),
        "test_duplicate_ids": sorted(item for item in test_set if test_ids.count(item) > 1),
        "cal_test_intersection": sorted(cal_set.intersection(test_set)),
        "cal_test_disjoint": cal_set.isdisjoint(test_set),
        "all_ids_within_source_bounds": in_bounds,
        "regeneration_formula": (
            "perm = numpy.random.RandomState(seed).permutation(source_row_count); "
            "cal_ids = perm[:100]; test_ids = perm[100:300]"
        ),
        "expected_cal_ids": expected_cal,
        "expected_test_ids": expected_test,
        "cal_ids_match_regeneration": cal_ids == expected_cal,
        "test_ids_match_regeneration": test_ids == expected_test,
    }
    split_proof["matches_regeneration"] = (
        split_proof["cal_ids_match_regeneration"]
        and split_proof["test_ids_match_regeneration"]
    )
    public["file_internal_split_proof"] = split_proof

    canonical_records = []
    for part, ids, texts in (
        ("cal_clean", cal_ids, cal_clean),
        ("test_clean", test_ids, test_clean),
    ):
        for part_index, (row_id, text) in enumerate(zip(ids, texts)):
            canonical_records.append(
                text_ref(
                    text,
                    canonical_set="clean_300",
                    canonical_position=len(canonical_records),
                    part=part,
                    part_index=part_index,
                    validation_row_id=row_id,
                )
            )
    triggered_records = [
        text_ref(
            text,
            canonical_set="test_trig_200",
            part="test_trig",
            part_index=index,
            validation_row_id=test_ids[index],
        )
        for index, text in enumerate(test_trig)
    ]
    public["canonical_text_hashes"] = {
        "hash_definition": "SHA256 of compact UTF-8 JSON array, preserving order; per-text SHA256 is raw UTF-8",
        "clean_text_count": len(canonical_records),
        "is_exactly_300_clean_texts": len(canonical_records) == 300,
        "cal_clean_ordered_sha256": ordered_hash(cal_clean),
        "test_clean_ordered_sha256": ordered_hash(test_clean),
        "clean_300_ordered_sha256": ordered_hash(cal_clean + test_clean),
        "test_trig_ordered_sha256": ordered_hash(test_trig),
        "unique_clean_text_count": len(set(cal_clean + test_clean)),
        "records": [public_ref(record) for record in canonical_records],
    }

    mismatches: dict[str, list[dict[str, Any]]] = {
        "cal_clean": [],
        "cal_labels": [],
        "test_clean": [],
        "test_labels": [],
        "test_trig": [],
    }
    checked = {key: 0 for key in mismatches}
    if in_bounds:
        for part, ids, texts, labels in (
            ("cal", cal_ids, cal_clean, cal_labels),
            ("test", test_ids, test_clean, test_labels),
        ):
            for index, row_id in enumerate(ids):
                expected_text = source_texts[row_id]
                expected_label = source_labels[row_id]
                text_key = f"{part}_clean"
                label_key = f"{part}_labels"
                checked[text_key] += 1
                checked[label_key] += 1
                if texts[index] != expected_text:
                    mismatches[text_key].append(
                        {"part_index": index, "row_id": row_id, "pool": texts[index], "source": expected_text}
                    )
                if labels[index] != expected_label:
                    mismatches[label_key].append(
                        {"part_index": index, "row_id": row_id, "pool": labels[index], "source": expected_label}
                    )
                if part == "test":
                    checked["test_trig"] += 1
                    expected_trig = pool_trigger(expected_text, pool["trigger"], pool["position"])
                    if test_trig[index] != expected_trig:
                        mismatches["test_trig"].append(
                            {"part_index": index, "row_id": row_id, "pool": test_trig[index], "expected": expected_trig}
                        )
    row_checks = {
        key: {
            "checked_count": checked[key],
            "mismatch_count": len(items),
            "mismatches": items[:50],
            "mismatch_records_truncated": len(items) > 50,
        }
        for key, items in mismatches.items()
    }
    public["pool_source_row_consistency"] = {
        "pool_metadata": {
            key: builtin(pool.get(key)) for key in ("dataset", "trigger", "position")
        },
        "checks": row_checks,
        "all_corresponding_rows_match": bool(
            in_bounds
            and checked["cal_clean"] == len(cal_ids)
            and checked["test_clean"] == len(test_ids)
            and checked["test_trig"] == len(test_ids)
            and all(not items for items in mismatches.values())
        ),
    }
    public["status"] = "completed"
    context = {
        "canonical_clean": canonical_records,
        "canonical_triggered": triggered_records,
        "source_texts": source_texts,
        "source_labels": source_labels,
        "source_dataset_idx": source_dataset_idx,
        "split_proof": split_proof,
        "pool_rows_match": public["pool_source_row_consistency"]["all_corresponding_rows_match"],
    }
    return public, context


def extract_literal_assignments(path: Path, names: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    record = file_record(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    values: dict[str, Any] = {}
    lines: dict[str, int] = {}
    for node in tree.body:
        targets = []
        value_node = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names and value_node is not None:
                if target.id in values:
                    raise ValueError(f"duplicate top-level assignment to {target.id} in {path}")
                values[target.id] = ast.literal_eval(value_node)
                lines[target.id] = node.lineno
    missing = sorted(names.difference(values))
    if missing:
        raise ValueError(f"could not statically extract {missing} from {path}")
    record.update(
        {
            "extraction": "ast.parse plus ast.literal_eval on top-level assignments; module was not imported or executed",
            "assignment_lines": lines,
        }
    )
    return values, record


def phase2_insert_trigger(text: str, trigger: str) -> str:
    words = text.split()
    if len(words) > 2:
        words.insert(len(words) // 2, trigger)
    else:
        words.append(trigger)
    return " ".join(words)


def audit_static_sentences(
    phase1_path: Path, phase2_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    public: dict[str, Any] = {}
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {"phase1": {}, "phase2": {}}

    phase1_values, phase1_record = extract_literal_assignments(
        phase1_path, {"POSITIVE", "NEGATIVE"}
    )
    for variable in ("POSITIVE", "NEGATIVE"):
        values = phase1_values[variable]
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise TypeError(f"phase1 {variable} must be a list of strings")
        records = [
            text_ref(text, source_file=str(phase1_path), variable=variable, variable_index=index)
            for index, text in enumerate(values)
        ]
        groups["phase1"][variable] = records
        phase1_record[variable] = [public_ref(record) | {"text": record["text"]} for record in records]
    groups["phase1"]["ALL"] = groups["phase1"]["POSITIVE"] + groups["phase1"]["NEGATIVE"]
    phase1_record["status"] = "completed"
    public["phase1"] = phase1_record

    phase2_values, phase2_record = extract_literal_assignments(
        phase2_path, {"CANDIDATE_TRIGGERS", "TEST_NEGATIVE", "TEST_POSITIVE"}
    )
    candidates = phase2_values["CANDIDATE_TRIGGERS"]
    if not isinstance(candidates, dict) or not all(isinstance(key, str) for key in candidates):
        raise TypeError("phase2 CANDIDATE_TRIGGERS must be a string-keyed dictionary")
    for variable in ("TEST_NEGATIVE", "TEST_POSITIVE"):
        values = phase2_values[variable]
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise TypeError(f"phase2 {variable} must be a list of strings")
        records = [
            text_ref(text, source_file=str(phase2_path), variable=variable, variable_index=index)
            for index, text in enumerate(values)
        ]
        groups["phase2"][variable] = records
        phase2_record[variable] = [public_ref(record) | {"text": record["text"]} for record in records]

    candidate_records = [
        text_ref(trigger, source_file=str(phase2_path), variable="CANDIDATE_TRIGGERS", category=builtin(category))
        for trigger, category in candidates.items()
    ]
    groups["phase2"]["CANDIDATE_TRIGGERS"] = candidate_records
    phase2_record["CANDIDATE_TRIGGERS"] = [
        public_ref(record) | {"text": record["text"]} for record in candidate_records
    ]
    derived = []
    for trigger in candidates:
        for base_index, base in enumerate(phase2_values["TEST_NEGATIVE"]):
            derived.append(
                text_ref(
                    phase2_insert_trigger(base, trigger),
                    source_file=str(phase2_path),
                    derivation="phase2 insert_trigger(TEST_NEGATIVE[i], trigger)",
                    trigger=trigger,
                    base_variable="TEST_NEGATIVE",
                    base_index=base_index,
                    base_text_sha256=sha256_text(base),
                )
            )
    groups["phase2"]["DERIVED_TRIGGER_INPUTS"] = derived
    groups["phase2"]["ALL_SCORING_INPUTS"] = (
        groups["phase2"]["TEST_NEGATIVE"] + groups["phase2"]["TEST_POSITIVE"] + derived
    )
    phase2_record["DERIVED_TRIGGER_INPUTS"] = [
        public_ref(record) | {"text": record["text"]} for record in derived
    ]
    phase2_record["scoring_input_semantics"] = (
        "ALL_SCORING_INPUTS contains clean TEST_NEGATIVE/TEST_POSITIVE and every "
        "insert_trigger(TEST_NEGATIVE[i], candidate) input. CANDIDATE_TRIGGERS are "
        "audited separately as exact phrases; they are not mislabeled as standalone scoring sentences."
    )
    phase2_record["status"] = "completed"
    public["phase2"] = phase2_record
    return public, groups


def full_train_records(train_dataset: Any) -> list[dict[str, Any]]:
    texts = train_dataset["sentence"]
    dataset_ids = train_dataset["idx"] if "idx" in train_dataset.column_names else [None] * len(texts)
    labels = train_dataset["label"] if "label" in train_dataset.column_names else [None] * len(texts)
    records = []
    for row_id, (text, dataset_idx, label) in enumerate(zip(texts, dataset_ids, labels)):
        if not isinstance(text, str):
            raise TypeError(f"train sentence at original row {row_id} is not a string")
        records.append(
            text_ref(text, original_row_id=row_id, dataset_idx=builtin(dataset_idx), label=builtin(label))
        )
    return records


def segment_public(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "ordered_text_sha256": ordered_hash([record["text"] for record in records]),
        "ordered_original_row_ids_sha256": ordered_hash(
            [record["original_row_id"] for record in records]
        ),
        "original_row_ids": [record["original_row_id"] for record in records],
        "rows": [public_ref(record) for record in records],
    }


def audit_train_seed(
    train_dataset: Any, full_records: list[dict[str, Any]], seed: int
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    indexed = train_dataset.add_column(ORIGINAL_ID_COLUMN, list(range(len(train_dataset))))
    shuffled = indexed.shuffle(seed=seed)
    n_search = min(SEARCH_SAMPLES, len(shuffled) // 2)
    train_end = min(n_search + TRAIN_SAMPLES, len(shuffled))
    search_dataset = shuffled.select(range(n_search))
    train_part_dataset = shuffled.select(range(n_search, train_end))

    def materialize(dataset: Any, segment: str, position_offset: int) -> list[dict[str, Any]]:
        materialized = []
        for segment_position, row in enumerate(dataset.to_list()):
            original_id = int(row[ORIGINAL_ID_COLUMN])
            record = dict(full_records[original_id])
            record["segment"] = segment
            record["segment_position"] = segment_position
            record["shuffled_position"] = position_offset + segment_position
            materialized.append(record)
        return materialized

    search = materialize(search_dataset, "search", 0)
    train = materialize(train_part_dataset, "train", n_search)
    search_ids = {record["original_row_id"] for record in search}
    train_ids = {record["original_row_id"] for record in train}
    public = {
        "seed": seed,
        "algorithm": (
            "add immutable original_row_id before Dataset.shuffle(seed); "
            "search=select(range(min(2000,len//2))); "
            "train=select(range(n_search,min(n_search+5000,len)))"
        ),
        "n_search": n_search,
        "n_train": len(train),
        "search_train_original_row_id_intersection": sorted(search_ids.intersection(train_ids)),
        "search_train_original_row_ids_disjoint": search_ids.isdisjoint(train_ids),
        "cache_files": {
            "indexed": cache_file_record(indexed),
            "shuffled": cache_file_record(shuffled),
            "search": cache_file_record(search_dataset),
            "train": cache_file_record(train_part_dataset),
        },
        "search": segment_public(search),
        "train": segment_public(train),
    }
    return public, {"search": search, "train": train, "union": search + train}


def compare_validation_to_source(validation: Any, source: dict[str, Any]) -> dict[str, Any]:
    texts = validation["sentence"]
    labels = [int(item) for item in validation["label"]]
    dataset_ids = validation["idx"] if "idx" in validation.column_names else None
    count = min(len(texts), len(source["source_texts"]))
    text_mismatches = []
    label_mismatches = []
    idx_mismatches = []
    for index in range(count):
        if texts[index] != source["source_texts"][index]:
            text_mismatches.append(index)
        if labels[index] != source["source_labels"][index]:
            label_mismatches.append(index)
        if (
            dataset_ids is not None
            and source["source_dataset_idx"] is not None
            and dataset_ids[index] != source["source_dataset_idx"][index]
        ):
            idx_mismatches.append(index)
    return {
        "row_count_equal": len(texts) == len(source["source_texts"]),
        "datasets_row_count": len(texts),
        "parquet_row_count": len(source["source_texts"]),
        "compared_row_count": count,
        "text_mismatch_count": len(text_mismatches),
        "text_mismatch_indices_first50": text_mismatches[:50],
        "label_mismatch_count": len(label_mismatches),
        "label_mismatch_indices_first50": label_mismatches[:50],
        "dataset_idx_mismatch_count": len(idx_mismatches),
        "dataset_idx_mismatch_indices_first50": idx_mismatches[:50],
        "datasets_ordered_sentence_sha256": ordered_hash(texts),
        "all_compared_content_matches": bool(
            len(texts) == len(source["source_texts"])
            and not text_mismatches
            and not label_mismatches
            and not idx_mismatches
        ),
    }


def unavailable(error: Exception) -> dict[str, Any]:
    return {"status": "not_available", "error_type": type(error).__name__, "error": str(error)}


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--output", default="results/data_isolation.json")
    parser.add_argument("--seed", type=int, default=2024, help="canonical pool split seed")
    parser.add_argument("--phase1-source", default=DEFAULT_PHASE1)
    parser.add_argument("--phase2-source", default=DEFAULT_PHASE2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    if os.path.lexists(output_path):
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    uncovered = [
        {
            "item": "scope_of_exact_matching",
            "detail": "Exact string intersections do not test normalized, case-folded, token-level, paraphrased, or near-duplicate overlap.",
        },
        {
            "item": "cross_split_row_ids",
            "detail": "Train original_row_id and validation_row_id belong to different splits; numeric equality across splits is not treated as overlap evidence.",
        },
        {
            "item": "historical_provenance",
            "detail": "Current files, local caches, versions, and static source literals do not prove every historical remote execution used the same bytes or environment.",
        },
        {
            "item": "model_and_training",
            "detail": "No model was loaded and no training or remote experiment was run; dynamic behavior outside the extracted literals is not audited.",
        },
    ]
    actual: dict[str, Any] = {}
    source_context = None
    try:
        actual["source_and_pool"], source_context = audit_source_and_pool(
            Path(args.source_parquet), Path(args.pool), args.seed
        )
    except Exception as error:
        actual["source_and_pool"] = unavailable(error)
        actual["source_and_pool"]["source_parquet"] = file_record(Path(args.source_parquet))
        actual["source_and_pool"]["canonical_pool"] = file_record(Path(args.pool))
        uncovered.append({"item": "source_pool_content", "detail": str(error)})

    static_groups: dict[str, dict[str, list[dict[str, Any]]]] = {"phase1": {}, "phase2": {}}
    try:
        actual["static_source_sentences"], static_groups = audit_static_sentences(
            Path(args.phase1_source), Path(args.phase2_source)
        )
    except Exception as error:
        actual["static_source_sentences"] = unavailable(error)
        uncovered.append({"item": "static_source_sentences", "detail": str(error)})

    cache: dict[str, Any] = {
        "offline_environment": {
            key: os.environ[key]
            for key in (
                "HF_DATASETS_OFFLINE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
                "DO_NOT_TRACK",
            )
        },
        "versions": {
            "python": platform.python_version(),
            "numpy_installed": np.__version__,
            "pandas_installed": package_version("pandas"),
            "datasets_installed": package_version("datasets"),
            "pyarrow_installed": package_version("pyarrow"),
        },
        "load_dataset_calls": {
            "train": {"name": "glue", "config": "sst2", "split": "train", "network": "disabled by offline environment"},
            "validation": {"name": "glue", "config": "sst2", "split": "validation", "network": "disabled by offline environment"},
        },
        "original_row_id_definition": (
            "original_row_id is the zero-based row position in the unshuffled HF train split; "
            "dataset_idx separately preserves the SST-2 idx column. Neither is a post-shuffle index."
        ),
    }

    train_dataset = None
    validation_dataset = None
    full_records = None
    try:
        datasets_module = importlib.import_module("datasets")
        cache["versions"]["datasets_imported"] = datasets_module.__version__
        for split in ("train", "validation"):
            try:
                loaded = datasets_module.load_dataset("glue", "sst2", split=split)
                if split == "train":
                    train_dataset = loaded
                else:
                    validation_dataset = loaded
                cache[f"{split}_dataset"] = {
                    "status": "loaded_offline",
                    "row_count": len(loaded),
                    "column_names": list(loaded.column_names),
                    "cache_files": cache_file_record(loaded),
                }
            except Exception as error:
                cache[f"{split}_dataset"] = unavailable(error)
                uncovered.append({"item": f"offline_{split}_dataset", "detail": str(error)})
    except Exception as error:
        cache["datasets_import"] = unavailable(error)
        uncovered.append({"item": "datasets_import", "detail": str(error)})

    if train_dataset is not None:
        try:
            full_records = full_train_records(train_dataset)
            cache["train_dataset"]["ordered_sentence_sha256"] = ordered_hash(
                [record["text"] for record in full_records]
            )
            cache["train_dataset"]["original_row_id_count"] = len(full_records)
        except Exception as error:
            cache["train_dataset"]["content_index"] = unavailable(error)
            uncovered.append({"item": "train_content_index", "detail": str(error)})

    if validation_dataset is not None:
        cache["validation_dataset"]["ordered_sentence_sha256"] = ordered_hash(
            validation_dataset["sentence"]
        )
        if source_context is not None:
            try:
                cache["validation_dataset"]["source_parquet_comparison"] = (
                    compare_validation_to_source(validation_dataset, source_context)
                )
            except Exception as error:
                cache["validation_dataset"]["source_parquet_comparison"] = unavailable(error)
        else:
            cache["validation_dataset"]["source_parquet_comparison"] = None

    if source_context is not None and any(static_groups.values()):
        for phase, groups in static_groups.items():
            if not groups:
                continue
            intersections = {}
            for group_name, records in groups.items():
                intersections[group_name] = {
                    "canonical_clean_300": exact_intersection(records, source_context["canonical_clean"]),
                    "canonical_test_trig_200": exact_intersection(records, source_context["canonical_triggered"]),
                }
            actual["static_source_sentences"][phase]["canonical_exact_intersections"] = intersections

    seed_segments: dict[int, dict[str, list[dict[str, Any]]]] = {}
    if train_dataset is not None and full_records is not None:
        cache["train_segments_by_seed"] = {}
        for seed in TRAIN_SEEDS:
            try:
                seed_public, segments = audit_train_seed(train_dataset, full_records, seed)
                seed_segments[seed] = segments
                if source_context is not None:
                    seed_public["canonical_exact_intersections"] = {
                        canonical_name: {
                            segment_name: exact_intersection(canonical_records, segment_records)
                            for segment_name, segment_records in segments.items()
                        }
                        for canonical_name, canonical_records in (
                            ("clean_300", source_context["canonical_clean"]),
                            ("test_trig_200", source_context["canonical_triggered"]),
                        )
                    }
                else:
                    seed_public["canonical_exact_intersections"] = None
                cache["train_segments_by_seed"][str(seed)] = seed_public
            except Exception as error:
                cache["train_segments_by_seed"][str(seed)] = unavailable(error)
                uncovered.append({"item": f"train_seed_{seed}", "detail": str(error)})
    else:
        cache["train_segments_by_seed"] = None

    if full_records is not None and any(static_groups.values()):
        cache["static_source_train_exact_intersections"] = {}
        for phase, groups in static_groups.items():
            for group_name, records in groups.items():
                cache["static_source_train_exact_intersections"][f"{phase}:{group_name}"] = {
                    "full_train": exact_intersection(records, full_records),
                    "by_seed": {
                        str(seed): {
                            segment_name: exact_intersection(records, segment_records)
                            for segment_name, segment_records in segments.items()
                        }
                        for seed, segments in seed_segments.items()
                    },
                }
    else:
        cache["static_source_train_exact_intersections"] = None

    summary = {
        "source_pool_content_check_completed": source_context is not None,
        "source_pool_all_content_checks_passed": (
            bool(
                source_context["split_proof"]["cal_test_disjoint"]
                and source_context["split_proof"]["matches_regeneration"]
                and source_context["pool_rows_match"]
                and actual["source_and_pool"]["canonical_text_hashes"]["is_exactly_300_clean_texts"]
            )
            if source_context
            else None
        ),
        "cal_test_disjoint": source_context["split_proof"]["cal_test_disjoint"] if source_context else None,
        "pool_matches_seeded_regeneration": source_context["split_proof"]["matches_regeneration"] if source_context else None,
        "pool_matches_source_rows": source_context["pool_rows_match"] if source_context else None,
        "canonical_clean_300_sha256": (
            actual["source_and_pool"]["canonical_text_hashes"]["clean_300_ordered_sha256"]
            if source_context
            else None
        ),
        "train_seeds_completed": sorted(seed_segments),
        "static_train_intersections_completed": cache["static_source_train_exact_intersections"] is not None,
        "interpretation": (
            "Report only the completed checks. A zero exact-match count means no exact string match in those "
            "compared sets; it is not absolute isolation. Missing/failed checks are null or not_available and "
            "support neither a leakage claim nor a no-leakage claim."
        ),
    }
    result = {
        "schema_version": 1,
        "audit": "SST-2 data isolation static/offline evidence",
        "parameters": {
            "source_parquet": str(args.source_parquet),
            "canonical_pool": str(args.pool),
            "canonical_seed": args.seed,
            "train_seeds": list(TRAIN_SEEDS),
            "search_samples": SEARCH_SAMPLES,
            "train_samples": TRAIN_SAMPLES,
            "phase1_source": str(args.phase1_source),
            "phase2_source": str(args.phase2_source),
        },
        "actual_file_content_verification": actual,
        "cache_and_version_evidence": cache,
        "uncovered_items": uncovered,
        "summary": summary,
    }
    write_json_new(output_path, result)
    print(f"Wrote data-isolation audit: {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
