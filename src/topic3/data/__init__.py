"""Frozen metadata access; does not relabel or read skeleton arrays."""
import json
from topic3.paths import MANIFESTS
from keraal_loc.audit import annotation
def historical_manifest():
    return [json.loads(line) for line in (MANIFESTS/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
def historical_splits():
    return json.loads((MANIFESTS/'splits.json').read_text(encoding='utf-8'))
