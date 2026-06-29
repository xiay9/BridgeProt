"""Unified data access for BridgeProt datasets."""

from .doc_ids import attach_doc_ids, load_raw_to_doc_id_map, resolve_dialogue_doc_id
from .registry import available_datasets, filter_future_pairs, get_adapter, load_dataset, load_split
from .schema import Dialogue, FeaturePaths, LoadedSplit, Utterance
from .targets import build_bridgeprot_target_output, render_bridgeprot_target_json

__all__ = [
    "attach_doc_ids",
    "build_bridgeprot_target_output",
    "Dialogue",
    "FeaturePaths",
    "LoadedSplit",
    "Utterance",
    "available_datasets",
    "filter_future_pairs",
    "get_adapter",
    "load_raw_to_doc_id_map",
    "load_dataset",
    "load_split",
    "render_bridgeprot_target_json",
    "resolve_dialogue_doc_id",
]
