"""CPU-only checks of the pinned public detector; no LLM experiment results."""
import ast
import argparse
import hashlib
import json
import math
import subprocess
import typing
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('upstream_checkout', help='JavadDogani/IndexGuard checkout at 6e745f81cdee8ea56830bcc1dce719259430954e')
parser.add_argument('--output', default='results/indexguard-heterogeneity-20261009/upstream_audit.json')
args = parser.parse_args()
repo = Path(args.upstream_checkout)
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == '6e745f81cdee8ea56830bcc1dce719259430954e'
source = repo / 'IndexGuard/IndexGuard.py'
parsed = ast.parse(source.read_text(encoding='utf-8'))
names = {'indexguard_cluster_labels_from_similarity', 'confusion_metrics_from_cluster_labels'}
definitions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(definitions) == 2
scope = {'torch': torch, 'math': math, **vars(typing)}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), scope)
detect = scope['indexguard_cluster_labels_from_similarity']
evaluate = scope['confusion_metrics_from_cluster_labels']

# No group is more cohesive: every off-diagonal IOS is the same.
sims = torch.full((30, 30), 0.25)
sims.fill_diagonal_(1.0)
default = detect(sims)
no_fallback = detect(sims, fallback='none')
assert sum(default) > 0
assert sum(no_fallback) == 0

# The semantic labels are deliberately reversed to expose oracle remapping.
# A deployment evaluation must score these predictions as given, not flip them.
true_malicious = set(range(9))
semantic_predictions = [0] * 9 + [1] * 21
oracle = evaluate(list(range(30)), semantic_predictions, true_malicious)
assert oracle['accuracy'] == 1.0 and oracle['malicious_cluster'] == 0

report = {
    'kind': 'implementation_audit_only',
    'not_research_evidence': True,
    'upstream_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'synthetic_checks': {
        'uniform_ios_default_flagged_clients': sum(default),
        'uniform_ios_no_fallback_flagged_clients': sum(no_fallback),
        'reversed_semantic_predictions_actual_accuracy': 0.0,
        'reversed_semantic_predictions_reported_accuracy': oracle['accuracy'],
    },
    'static_findings': [
        {'lines': [293, 296, 442, 1406], 'finding': 'Main calls detector with defaults Kc=5, beta=0.9, fallback=max_cohesion; CLI cluster_k is not passed. Paper Appendix F.1 specifies L=2, beta=0.7, gamma=0.05 for shared-trigger tests; paper rule does not require a forced-positive fallback.'},
        {'lines': [575, 662, 677], 'finding': 'Metric helper uses true malicious IDs to choose the more accurate label mapping, although detector labels already have benign/suspicious semantics.'},
        {'lines': [1377, 1392, 1400, 1406], 'finding': 'Main averages all received adapters before detection and does not apply detected exclusions to that aggregation.'},
        {'lines': [1417, 1430, 1455], 'finding': 'Clean/ASR evaluation and per-round CSV writes are commented out; main exits early at reported accuracy 1.0.'},
        {'lines': [262, 1234], 'finding': 'Top-K uses a fixed fraction; paper Section 5.3 instead describes a one-time coverage-and-stability calibration.'},
    ],
    'interpretation': 'The current public script cannot be treated as a faithful end-to-end reproduction without documenting and resolving these discrepancies. These synthetic checks do not estimate FPR under real heterogeneous LLM data, and do not establish that published experiments used this exact code.',
}
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
