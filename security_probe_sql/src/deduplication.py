"""
Deduplication and Pair Preservation Module for Python SQL Injection Dataset.

Implements:
1. Exact normalized source code hashing (SHA-256).
2. Efficient near-duplicate similarity detection using token n-gram length filtering.
3. Strict pair preservation: Keeps vulnerable/fixed versions linked by pair_id.
4. Repository and project metadata grouping.
"""

import hashlib
import re
from typing import List, Dict, Any, Tuple, Set


def normalize_code(code: str) -> str:
    """Normalizes code by stripping comments and uniform whitespace."""
    lines = []
    for line in code.splitlines():
        clean_line = re.sub(r'#.*$', '', line).strip()
        if clean_line:
            lines.append(clean_line)
    return "\n".join(lines)


def compute_code_hash(code: str) -> str:
    """Returns SHA-256 hash of normalized code."""
    norm = normalize_code(code)
    return hashlib.sha256(norm.encode('utf-8')).hexdigest()


def get_token_ngrams(code: str, n: int = 3) -> Set[Tuple[str, ...]]:
    """Extracts token n-grams for Jaccard similarity."""
    tokens = re.findall(r'[a-zA-Z_]\w+|[^\s\w]', code)
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(ngrams1: Set[Tuple[str, ...]], ngrams2: Set[Tuple[str, ...]]) -> float:
    """Computes Jaccard similarity between two n-gram sets."""
    if not ngrams1 or not ngrams2:
        return 0.0
    l1 = len(ngrams1)
    l2 = len(ngrams2)
    # Upper bound check: min(l1, l2) / max(l1, l2)
    if min(l1, l2) / max(l1, l2) < 0.85:
        return 0.0
    intersection = len(ngrams1.intersection(ngrams2))
    union = l1 + l2 - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_dataset(
    records: List[Dict[str, Any]], 
    near_dup_threshold: float = 0.90
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Performs exact and near-duplicate deduplication while strictly
    preserving pair relationships and source provenance.
    """
    exact_hashes: Dict[str, str] = {} # hash -> sample_id
    pair_groups: Dict[str, List[Dict[str, Any]]] = {}
    
    unique_records: List[Dict[str, Any]] = []
    excluded_exact_duplicates = 0
    excluded_near_duplicates = 0
    
    # Pass 1: Exact Hash Deduplication
    for r in records:
        code = r.get("code", "")
        h = compute_code_hash(code)
        r["code_hash"] = h
        
        if h in exact_hashes:
            excluded_exact_duplicates += 1
            continue
            
        exact_hashes[h] = r["sample_id"]
        pair_id = r.get("pair_id") or r["sample_id"]
        pair_groups.setdefault(pair_id, []).append(r)
        unique_records.append(r)
        
    # Pass 2: Near-duplicate detection using length-binned inverted index
    ngrams_map = {r["sample_id"]: get_token_ngrams(r["code"]) for r in unique_records}
    final_records: List[Dict[str, Any]] = []
    kept_ids: Set[str] = set()
    
    # Bucket candidate items by length bucket to avoid O(N^2) comparison
    length_buckets: Dict[int, List[Dict[str, Any]]] = {}
    
    for r in unique_records:
        sid = r["sample_id"]
        r_ngrams = ngrams_map[sid]
        l_ng = len(r_ngrams)
        bucket_key = l_ng // 25  # bucket size of 25 n-grams
        r_pair = r.get("pair_id")
        
        is_near_dup = False
        # Only compare against adjacent length buckets
        candidates = []
        for b in [bucket_key - 1, bucket_key, bucket_key + 1]:
            if b in length_buckets:
                candidates.extend(length_buckets[b])
                
        for kept_r in candidates:
            if r_pair and r_pair == kept_r.get("pair_id"):
                continue
                
            kept_ngrams = ngrams_map[kept_r["sample_id"]]
            sim = jaccard_similarity(r_ngrams, kept_ngrams)
            if sim >= near_dup_threshold:
                is_near_dup = True
                excluded_near_duplicates += 1
                break
                
        if not is_near_dup:
            final_records.append(r)
            kept_ids.add(sid)
            length_buckets.setdefault(bucket_key, []).append(r)
            
    # Preserve pair completeness
    for pid, group in pair_groups.items():
        if any(item["sample_id"] in kept_ids for item in group):
            for item in group:
                if item["sample_id"] not in kept_ids:
                    final_records.append(item)
                    kept_ids.add(item["sample_id"])
                    
    dedup_stats = {
        "raw_input_records": len(records),
        "after_exact_dedup": len(unique_records),
        "exact_duplicates_removed": excluded_exact_duplicates,
        "near_duplicates_removed": excluded_near_duplicates,
        "final_unique_records": len(final_records),
        "unique_pair_groups": len(set(r.get("pair_id") for r in final_records if r.get("pair_id")))
    }
    
    return final_records, dedup_stats
