"""
Pandas utils
"""
from typing import List, Dict


def column_prefixes(columns: List[str]) -> Dict[str, List[str]]:
    return {prefix: [col for col in columns if prefix == col.split('_')[0]] for prefix in
            list(dict.fromkeys([col.split('_')[0] for col in columns if len(col.split('_')) > 1]))}


def column_prefix_counts(columns: List[str]) -> Dict[str, int]:
    return {k: len(v) for k, v in column_prefixes(columns).items()}
