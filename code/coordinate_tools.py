#!/usr/bin/env python3
"""CISB coordinate counterfactuals using direct transformers/torch hooks."""

import argparse
import ast
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE1 = str(PROJECT_ROOT / "code" / "cisb_phase1_discovery.py")
DEFAULT_LIBRARY = str(PROJECT_ROOT / "data" / "circuit_library_cisb.json")
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_NAME = "Qwen2.5-7B-Instruct"
EXPECTED_POSITIVE_TOKEN = " wonderful"
EXPECTED_NEGATIVE_TOKEN = " terrible"
RANDOM_SEED = 2024


def extract_phase1_inputs(path):
    source_path = Path(path)
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    wanted = {"POSITIVE", "NEGATIVE", "pos_tok", "neg_tok"}
    values = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in wanted:
                continue
            name = target.id
            if name in values:
                raise ValueError(f"Duplicate phase1 assignment for {name}")
            if name in {"POSITIVE", "NEGATIVE"}:
                values[name] = ast.literal_eval(value)
            else:
                values[name] = _extract_encode_token(value, name)

    missing = wanted - values.keys()
    if missing:
        raise ValueError(f"Missing phase1 inputs: {sorted(missing)}")
    positive = values["POSITIVE"]
    negative = values["NEGATIVE"]
    for name, sentences in (("POSITIVE", positive), ("NEGATIVE", negative)):
        if (
            not isinstance(sentences, list)
            or len(sentences) != 16
            or not all(isinstance(s, str) and s for s in sentences)
        ):
            raise ValueError(f"{name} must contain the original 16 non-empty strings")
    if values["pos_tok"] != EXPECTED_POSITIVE_TOKEN:
        raise ValueError(
            f"Positive token differs from phase1: {values['pos_tok']!r}"
        )
    if values["neg_tok"] != EXPECTED_NEGATIVE_TOKEN:
        raise ValueError(
            f"Negative token differs from phase1: {values['neg_tok']!r}"
        )

    return {
        "path": str(source_path),
        "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "POSITIVE": positive,
        "NEGATIVE": negative,
        "pos_tok": values["pos_tok"],
        "neg_tok": values["neg_tok"],
    }


def _extract_encode_token(value, assignment_name):
    if not isinstance(value, ast.Subscript):
        raise ValueError(f"{assignment_name} is not a tokenizer.encode(...)[0] assignment")
    try:
        index = ast.literal_eval(value.slice)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid token index for {assignment_name}") from exc
    call = value.value
    if index != 0 or not isinstance(call, ast.Call):
        raise ValueError(f"{assignment_name} must select tokenizer.encode(...)[0]")
    if (
        not isinstance(call.func, ast.Attribute)
        or call.func.attr != "encode"
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "tokenizer"
        or len(call.args) != 1
    ):
        raise ValueError(f"{assignment_name} must call tokenizer.encode once")
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if "add_special_tokens" not in keywords:
        raise ValueError(f"{assignment_name} must set add_special_tokens=False")
    add_special_tokens = ast.literal_eval(keywords["add_special_tokens"])
    token_text = ast.literal_eval(call.args[0])
    if add_special_tokens is not False or not isinstance(token_text, str):
        raise ValueError(f"Invalid tokenizer arguments for {assignment_name}")
    return token_text


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def output_path(value):
    return Path(value).expanduser().resolve(strict=False)


def preflight_outputs(paths, protected_paths=()):
    resolved = [output_path(path) for path in paths]
    protected = {output_path(path) for path in protected_paths}
    if len(set(resolved)) != len(resolved):
        raise ValueError("Output paths must be distinct")
    for path in resolved:
        if path in protected or path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    return resolved


def write_json_new(path, payload):
    path = output_path(path)
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}") from exc
    print(f"Saved: {path}")


def model_dimensions(model):
    config = model.config
    dimensions = {
        "n_layers": int(config.num_hidden_layers),
        "n_heads": int(config.num_attention_heads),
        "hidden_size": int(config.hidden_size),
    }
    if min(dimensions.values()) <= 0:
        raise ValueError(f"Invalid model dimensions: {dimensions}")
    dimensions["head_dim"] = dimensions["hidden_size"] // dimensions["n_heads"]
    if dimensions["head_dim"] <= 0:
        raise ValueError(f"Invalid head_dim for dimensions: {dimensions}")
    return dimensions


def get_o_proj(model, layer_idx):
    try:
        return model.model.layers[layer_idx].self_attn.o_proj
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError(
            "The model must expose model.model.layers[layer].self_attn.o_proj "
            "as in the original phase1 script"
        ) from exc


def make_layer_zero_hook(torch):
    def zero_output(_module, _args, output):
        if torch.is_tensor(output):
            return torch.zeros_like(output)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (torch.zeros_like(output[0]),) + output[1:]
        raise TypeError("Unsupported o_proj output type for layer ablation")

    return zero_output


def make_head_zero_hook(torch, head_idx, head_dim):
    start = head_idx * head_dim
    end = start + head_dim

    def zero_head(_module, args):
        if not args or not torch.is_tensor(args[0]):
            raise TypeError("o_proj input must start with a tensor")
        activation = args[0]
        if activation.ndim not in (2, 3) or activation.shape[-1] < end:
            raise ValueError(
                f"Unexpected o_proj input shape {tuple(activation.shape)} "
                f"for head slice [{start}:{end}]"
            )
        modified = activation.clone()
        modified[..., start:end] = 0
        return (modified,) + args[1:]

    return zero_head


def mean_last_token_logit(
    model,
    tokenizer,
    sentences,
    target_token,
    device,
    torch,
    layer_idx=None,
    head_idx=None,
    head_dim=None,
):
    if head_idx is not None and (layer_idx is None or head_dim is None):
        raise ValueError("Head ablation requires layer_idx and head_dim")
    handle = None
    if layer_idx is not None:
        o_proj = get_o_proj(model, layer_idx)
        if head_idx is None:
            handle = o_proj.register_forward_hook(make_layer_zero_hook(torch))
        else:
            handle = o_proj.register_forward_pre_hook(
                make_head_zero_hook(torch, head_idx, head_dim)
            )

    scores = []
    try:
        for sentence in sentences:
            encoded = tokenizer(sentence, return_tensors="pt")
            inputs = {name: value.to(device) for name, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
            scores.append(float(outputs.logits[0, -1, target_token].item()))
    finally:
        if handle is not None:
            handle.remove()
    return math.fsum(scores) / len(scores)


def encode_target_token(tokenizer, token_text):
    token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Expected one token for {token_text!r}, received {token_ids}"
        )
    token_id = int(token_ids[0])
    decoded = tokenizer.decode([token_id])
    return {
        "text": token_text,
        "id": token_id,
        "decoded": decoded,
        "add_special_tokens": False,
        "single_token": True,
        "decode_matches_text": decoded == token_text,
    }


def discover_coordinates(
    model,
    tokenizer,
    dimensions,
    sentences,
    token_info,
    top_k,
    device,
    torch,
    label,
):
    started = time.monotonic()
    target_token = token_info["id"]
    n_layers = dimensions["n_layers"]
    n_heads = dimensions["n_heads"]
    head_dim = dimensions["head_dim"]
    baseline = mean_last_token_logit(
        model, tokenizer, sentences, target_token, device, torch
    )
    print(f"[{label}] baseline target logit: {baseline:.6f}")

    layer_results = []
    for layer_idx in range(0, n_layers, 2):
        ablated = mean_last_token_logit(
            model,
            tokenizer,
            sentences,
            target_token,
            device,
            torch,
            layer_idx=layer_idx,
        )
        drop = baseline - ablated
        layer_results.append(
            {"layer": int(layer_idx), "logit": float(ablated), "drop": float(drop)}
        )
        print(f"[{label}] layer {layer_idx:2d}: drop={drop:+.6f}")

    layer_sorted = sorted(layer_results, key=lambda item: item["drop"], reverse=True)
    top_layers = [item["layer"] for item in layer_sorted if item["drop"] > 0.02][:5]
    layer_fallback = len(top_layers) < 3
    if layer_fallback:
        top_layers = [item["layer"] for item in layer_sorted[:4]]
    print(f"[{label}] selected layers: {top_layers}")

    stage2_baseline = mean_last_token_logit(
        model, tokenizer, sentences, target_token, device, torch
    )
    head_results = []
    for layer_idx in top_layers:
        print(f"[{label}] scanning {n_heads} heads in layer {layer_idx}")
        for head_idx in range(n_heads):
            ablated = mean_last_token_logit(
                model,
                tokenizer,
                sentences,
                target_token,
                device,
                torch,
                layer_idx=layer_idx,
                head_idx=head_idx,
                head_dim=head_dim,
            )
            drop = stage2_baseline - ablated
            head_results.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "importance": float(drop),
                }
            )

    head_sorted = sorted(
        head_results, key=lambda item: item["importance"], reverse=True
    )
    circuit_heads = [item for item in head_sorted if item["importance"] > 0][:top_k]
    head_fallback = len(circuit_heads) < 10
    if head_fallback:
        circuit_heads = head_sorted[:20]
    print(f"[{label}] selected {len(circuit_heads)} heads")

    selection_method = (
        "Direct transformers/torch forward hooks reproduce phase1 coordinate "
        "counterfactuals: zero o_proj output for layer ablation and zero the "
        "corresponding o_proj-input head_dim slice for head ablation. Importance "
        "is the unablated mean last-token target logit minus the ablated mean. "
        "Layers are scanned with step 2, ranked by drop, filtered at drop > 0.02 "
        "and capped at 5, with the original 4-layer fallback. Heads in selected "
        f"layers are ranked by drop; positive-drop heads supply top_k={top_k}, "
        "with the original top-20 fallback when fewer than 10 are positive."
    )
    return {
        "top_layers": [int(layer) for layer in top_layers],
        "layer_importance": layer_sorted,
        "circuit_heads": circuit_heads,
        "all_heads_ranked": head_sorted,
        "baseline_target_logit": float(baseline),
        "baseline_stage2_target_logit": float(stage2_baseline),
        "selection_method": selection_method,
        "selection_config": {
            "sentence_count": len(sentences),
            "layer_step": 2,
            "layer_drop_threshold": 0.02,
            "max_top_layers": 5,
            "layer_fallback_used": layer_fallback,
            "top_k": int(top_k),
            "minimum_positive_heads_before_fallback": 10,
            "head_fallback_count": 20,
            "head_fallback_used": head_fallback,
        },
        "runtime_seconds": time.monotonic() - started,
    }


def make_discovered_library(
    model_name,
    model_path,
    dimensions,
    discovery,
    token_info,
    role,
):
    circuit = {
        "top_layers": discovery["top_layers"],
        "layer_importance": discovery["layer_importance"],
        "circuit_heads": discovery["circuit_heads"],
        "all_heads_ranked": discovery["all_heads_ranked"],
        "baseline_target_logit": discovery["baseline_target_logit"],
        "baseline_stage2_target_logit": discovery["baseline_stage2_target_logit"],
        "target_token": token_info,
        "selection_method": discovery["selection_method"],
        "selection_config": discovery["selection_config"],
    }
    if role == "target_positive":
        circuit["baseline_pos_logit"] = discovery["baseline_target_logit"]
    else:
        circuit["baseline_neg_logit"] = discovery["baseline_target_logit"]
    return {
        "model": model_name,
        "model_path": model_path,
        "n_layers": dimensions["n_layers"],
        "n_heads": dimensions["n_heads"],
        "hidden_size": dimensions["hidden_size"],
        "head_dim": dimensions["head_dim"],
        "method": "CISB coordinate discovery with direct transformers/torch hooks",
        "coordinate_role": role,
        "sentiment_circuit": circuit,
        "runtime_seconds": discovery["runtime_seconds"],
    }


def coordinate(entry):
    return int(entry["layer"]), int(entry["head"])


def coordinate_dict(item):
    return {"layer": int(item[0]), "head": int(item[1])}


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = math.fsum(xs) / len(xs)
    mean_y = math.fsum(ys) / len(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    variance_x = math.fsum(value * value for value in dx)
    variance_y = math.fsum(value * value for value in dy)
    if variance_x == 0 or variance_y == 0:
        return None
    value = math.fsum(x * y for x, y in zip(dx, dy)) / math.sqrt(
        variance_x * variance_y
    )
    return max(-1.0, min(1.0, value))


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        value = values[order[start]]
        while end < len(order) and values[order[end]] == value:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def aligned_correlation(original, recomputed, value_key, id_keys):
    original_map = {
        tuple(int(item[key]) for key in id_keys): float(item[value_key])
        for item in original
    }
    recomputed_map = {
        tuple(int(item[key]) for key in id_keys): float(item[value_key])
        for item in recomputed
    }
    common = sorted(original_map.keys() & recomputed_map.keys())
    xs = [original_map[key] for key in common]
    ys = [recomputed_map[key] for key in common]
    return {
        "overlap_count": len(common),
        "pearson": pearson(xs, ys),
        "spearman": pearson(average_ranks(xs), average_ranks(ys))
        if len(common) >= 2
        else None,
        "value": value_key,
    }


def compare_with_original(original, recomputed, model_name, expected_head_count):
    original_circuit = original["sentiment_circuit"]
    recomputed_circuit = recomputed["sentiment_circuit"]
    original_layers = [int(layer) for layer in original_circuit["top_layers"]]
    recomputed_layers = [int(layer) for layer in recomputed_circuit["top_layers"]]
    original_heads = [coordinate(item) for item in original_circuit["circuit_heads"]]
    recomputed_heads = [
        coordinate(item) for item in recomputed_circuit["circuit_heads"]
    ]
    original_set = set(original_heads)
    recomputed_set = set(recomputed_heads)
    overlap = original_set & recomputed_set
    original_head_dim = int(original["head_dim"])
    recomputed_head_dim = int(recomputed["head_dim"])

    top_layers_match = original_layers == recomputed_layers
    heads_match = (
        len(original_heads) == expected_head_count
        and len(recomputed_heads) == expected_head_count
        and original_set == recomputed_set
    )
    head_dim_match = original_head_dim == recomputed_head_dim
    comparison = {
        "original_model": original.get("model"),
        "recomputed_model": model_name,
        "model_match": original.get("model") == model_name,
        "matches": {
            "top_layers": top_layers_match,
            "circuit_heads": heads_match,
            "head_dim": head_dim_match,
            "all": top_layers_match and heads_match and head_dim_match,
        },
        "top_layers": {
            "original": original_layers,
            "recomputed": recomputed_layers,
            "exact_match": top_layers_match,
            "set_match": set(original_layers) == set(recomputed_layers),
            "overlap_count": len(set(original_layers) & set(recomputed_layers)),
        },
        "circuit_heads": {
            "expected_count": int(expected_head_count),
            "original_count": len(original_heads),
            "recomputed_count": len(recomputed_heads),
            "set_match": original_set == recomputed_set,
            "ordered_match": original_heads == recomputed_heads,
            "overlap_count": len(overlap),
            "original_only": [
                coordinate_dict(item) for item in sorted(original_set - recomputed_set)
            ],
            "recomputed_only": [
                coordinate_dict(item) for item in sorted(recomputed_set - original_set)
            ],
        },
        "head_dim": {
            "original": original_head_dim,
            "recomputed": recomputed_head_dim,
            "match": head_dim_match,
        },
        "importance_correlation": {
            "original_circuit_heads_vs_recomputed_ranking": aligned_correlation(
                original_circuit["circuit_heads"],
                recomputed_circuit["all_heads_ranked"],
                "importance",
                ("layer", "head"),
            ),
            "all_heads_ranked": aligned_correlation(
                original_circuit["all_heads_ranked"],
                recomputed_circuit["all_heads_ranked"],
                "importance",
                ("layer", "head"),
            ),
            "layer_drop": aligned_correlation(
                original_circuit["layer_importance"],
                recomputed_circuit["layer_importance"],
                "drop",
                ("layer",),
            ),
        },
    }
    return comparison


def input_verification(source, positive_token, negative_token):
    positive_sentences = source["POSITIVE"][:8]
    negative_sentences = source["NEGATIVE"][:8]
    tokens_match = (
        positive_token["text"] == source["pos_tok"]
        and negative_token["text"] == source["neg_tok"]
    )
    if not tokens_match:
        raise ValueError("Token verification against phase1 failed")
    return {
        "phase1_script": source["path"],
        "phase1_sha256": source["sha256"],
        "extraction": "ast.literal_eval only; phase1 module was not imported or executed",
        "positive_source_sentence_count": len(source["POSITIVE"]),
        "negative_source_sentence_count": len(source["NEGATIVE"]),
        "positive_sentences_used": positive_sentences,
        "negative_sentences_used": negative_sentences,
        "sentence_counts_match_phase1_first8": (
            len(positive_sentences) == 8 and len(negative_sentences) == 8
        ),
        "positive_token": positive_token,
        "negative_token": negative_token,
        "tokens_match_phase1": tokens_match,
    }


def load_model_and_tokenizer(model_path, gpu, torch, transformers_module):
    if gpu < 0:
        raise ValueError("GPU index must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for validate; random does not need a model")
    if gpu >= torch.cuda.device_count():
        raise ValueError(
            f"GPU index {gpu} is unavailable; found {torch.cuda.device_count()} GPU(s)"
        )
    device = torch.device(f"cuda:{gpu}")
    print(f"Loading model once from {model_path} on {device} (fp16)")
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = transformers_module.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def run_validate(args):
    if args.top_k <= 0:
        raise ValueError("top_k must be positive")
    validation_output, non_target_output = preflight_outputs(
        (args.validation_output, args.non_target_output),
        protected_paths=(args.library, args.phase1),
    )
    source = extract_phase1_inputs(args.phase1)
    original = load_json(args.library)

    import torch
    import transformers

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path, args.gpu, torch, transformers
    )
    dimensions = model_dimensions(model)
    print(
        f"Model dimensions: layers={dimensions['n_layers']}, "
        f"heads={dimensions['n_heads']}, hidden={dimensions['hidden_size']}, "
        f"head_dim={dimensions['head_dim']}"
    )
    positive_token = encode_target_token(tokenizer, source["pos_tok"])
    negative_token = encode_target_token(tokenizer, source["neg_tok"])
    verification = input_verification(source, positive_token, negative_token)

    target_discovery = discover_coordinates(
        model,
        tokenizer,
        dimensions,
        source["POSITIVE"][:8],
        positive_token,
        args.top_k,
        device,
        torch,
        "target/positive",
    )
    non_target_discovery = discover_coordinates(
        model,
        tokenizer,
        dimensions,
        source["NEGATIVE"][:8],
        negative_token,
        args.top_k,
        device,
        torch,
        "non-target/negative",
    )
    target_library = make_discovered_library(
        args.model_name,
        args.model_path,
        dimensions,
        target_discovery,
        positive_token,
        "target_positive",
    )
    non_target_library = make_discovered_library(
        args.model_name,
        args.model_path,
        dimensions,
        non_target_discovery,
        negative_token,
        "non_target_negative",
    )
    comparison = compare_with_original(
        original, target_library, args.model_name, args.top_k
    )
    target_coordinates = {
        coordinate(item)
        for item in original["sentiment_circuit"]["circuit_heads"]
    }
    non_target_coordinates = {
        coordinate(item)
        for item in non_target_discovery["circuit_heads"]
    }
    overlap_count = len(target_coordinates & non_target_coordinates)
    target_library["validation"] = {
        "original_library": str(output_path(args.library)),
        "original_library_untouched": True,
        "comparison": comparison,
        "input_verification": verification,
        "non_target_output": str(non_target_output),
        "non_target_target_head_overlap_count": overlap_count,
    }
    non_target_library["target_comparison"] = {
        "original_library": str(output_path(args.library)),
        "target_head_overlap_count": overlap_count,
    }

    write_json_new(validation_output, target_library)
    write_json_new(non_target_output, non_target_library)
    print(
        "Validation matches: "
        f"top_layers={comparison['matches']['top_layers']}, "
        f"circuit_heads={comparison['matches']['circuit_heads']}, "
        f"head_dim={comparison['matches']['head_dim']}"
    )


def build_random_library(original, source_library, top_k=30, seed=RANDOM_SEED):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    circuit = original["sentiment_circuit"]
    top_layers = [int(layer) for layer in circuit["top_layers"]]
    if len(set(top_layers)) != len(top_layers):
        raise ValueError("Original top_layers contains duplicates")
    target_heads = [coordinate(item) for item in circuit["circuit_heads"]]
    if len(target_heads) != top_k:
        raise ValueError(
            f"Random layer matching requires exactly top_k={top_k} target heads; "
            f"found {len(target_heads)}"
        )
    target_set = set(target_heads)
    if len(target_set) != len(target_heads):
        raise ValueError("Original circuit_heads contains duplicates")

    n_layers = int(original["n_layers"])
    n_heads = int(original["n_heads"])
    hidden_size = int(original["hidden_size"])
    head_dim = int(original["head_dim"])
    if min(n_layers, n_heads, hidden_size, head_dim) <= 0:
        raise ValueError("Original library has invalid model dimensions")
    if any(layer < 0 or layer >= n_layers for layer in top_layers):
        raise ValueError("Original top_layers contains an out-of-range layer")
    if any(
        layer < 0 or layer >= n_layers or head < 0 or head >= n_heads
        for layer, head in target_heads
    ):
        raise ValueError("Original circuit_heads contains an out-of-range coordinate")
    target_layers = {layer for layer, _head in target_heads}
    if not target_layers <= set(top_layers):
        raise ValueError("Every target head layer must occur in original top_layers")

    counts = Counter(layer for layer, _head in target_heads)
    rng = random.Random(seed)
    random_heads = []
    distribution = []
    for layer in top_layers:
        count = counts.get(layer, 0)
        candidates = [
            head for head in range(n_heads) if (layer, head) not in target_set
        ]
        if len(candidates) < count:
            raise ValueError(
                f"Layer {layer} has only {len(candidates)} complement heads "
                f"for the required {count}"
            )
        selected = sorted(rng.sample(candidates, count)) if count else []
        random_heads.extend((layer, head) for head in selected)
        distribution.append(
            {
                "layer": layer,
                "head_count": count,
                "complement_candidate_count": len(candidates),
            }
        )

    overlap = target_set & set(random_heads)
    if overlap or len(random_heads) != top_k:
        raise RuntimeError("Random complement validation failed")
    selection_method = (
        "Fixed-seed uniform random sampling without replacement from each top "
        "layer's non-target head complement, preserving the original per-layer "
        "head counts and total K. Random heads have no measured logit importance."
    )
    return {
        "model": original["model"],
        "n_layers": n_layers,
        "n_heads": n_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "method": "CISB random complement coordinate control",
        "coordinate_role": "random_complement",
        "source_library": str(output_path(source_library)),
        "sampling_seed": int(seed),
        "target_overlap_count": len(overlap),
        "selection_method": selection_method,
        "sentiment_circuit": {
            "top_layers": top_layers,
            "circuit_heads": [
                {"layer": layer, "head": head} for layer, head in random_heads
            ],
            "layer_distribution": distribution,
            "selection_method": selection_method,
            "selection_config": {
                "strategy": "per_layer_uniform_complement_without_replacement",
                "seed": int(seed),
                "top_k": int(top_k),
                "exclude_target_heads": True,
                "importance_measured": False,
            },
        },
    }


def run_random(args):
    (random_output,) = preflight_outputs(
        (args.output,), protected_paths=(args.library,)
    )
    original = load_json(args.library)
    payload = build_random_library(
        original, args.library, top_k=args.top_k, seed=args.seed
    )
    write_json_new(random_output, payload)
    print(
        f"Random heads: {len(payload['sentiment_circuit']['circuit_heads'])}, "
        f"seed={args.seed}, target_overlap={payload['target_overlap_count']}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="CISB coordinate counterfactual and random-complement tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Recompute positive targets and negative non-targets"
    )
    validate.add_argument(
        "--model-path", "--model_path", dest="model_path", default=DEFAULT_MODEL_PATH
    )
    validate.add_argument(
        "--model-name", "--model_name", dest="model_name", default=DEFAULT_MODEL_NAME
    )
    validate.add_argument("--gpu", type=int, default=0)
    validate.add_argument("--phase1", default=DEFAULT_PHASE1)
    validate.add_argument("--library", default=DEFAULT_LIBRARY)
    validate.add_argument(
        "--validation-output",
        "--output",
        dest="validation_output",
        default=str(PROJECT_ROOT / "results" / "coordinate_target_validation.json"),
    )
    validate.add_argument(
        "--non-target-output",
        "--non_target_output",
        dest="non_target_output",
        default=str(PROJECT_ROOT / "data" / "circuit_non_target.json"),
    )
    validate.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=30)
    validate.set_defaults(handler=run_validate)

    random_parser = subparsers.add_parser(
        "random", help="Generate a fixed-seed per-layer random complement"
    )
    random_parser.add_argument("--library", default=DEFAULT_LIBRARY)
    random_parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "data" / "circuit_random.json")
    )
    random_parser.add_argument(
        "--top-k", "--top_k", dest="top_k", type=int, default=30
    )
    random_parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    random_parser.set_defaults(handler=run_random)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
