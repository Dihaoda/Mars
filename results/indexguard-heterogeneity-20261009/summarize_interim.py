"""Describe every available completed round; final paired analysis waits for 10+10."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
provenance = read(root/'provenance.json')
cfg = provenance['config']
calibration = read(root/'calibration.json')
rounds = {name: [read(p) for p in sorted((root/name).glob('round_*.json'))]
          for name in ['mix0', 'mix04']}
assert all([r['round'] for r in values] == list(range(1, len(values)+1)) for values in rounds.values())
assert all(r['identity'] == provenance['identity'] and r['k'] == calibration['k']
           for values in rounds.values() for r in values)

def pooled(values):
    output = {}
    for role in ['benign', 'heterogeneous', 'malicious']:
        n = sum(r['detection'][role]['n'] for r in values)
        count = sum(r['detection'][role]['flagged'] for r in values)
        output[role] = {'flagged': count, 'n': n, 'rate': count/n if n else None}
    return output

paired_count = min(map(len, rounds.values()))
for a, b in zip(rounds['mix0'], rounds['mix04']):
    assert a['round'] == b['round'] and a['k'] == b['k']
    assert [x['id'] for x in a['clients']] == [x['id'] for x in b['clients']] == list(range(cfg['clients']))
    for x, y in zip(a['clients'], b['clients']):
        assert x['role'] == y['role'] and x['poison_positions'] == y['poison_positions']
        if a['round'] == 1 and x['id'] not in cfg['heterogeneous_ids']:
            assert x['local_adapter_sha256'] == y['local_adapter_sha256']
    if a['round'] == 1:
        assert a['base_hash'] == b['base_hash']

summary = {name: {'completed_rounds': len(values), 'planned_rounds': cfg['rounds'],
                 'pooled_client_round_detection': pooled(values),
                 'latest_evaluation': values[-1]['evaluation'],
                 'mean_round_seconds': sum(r['seconds_this_invocation'] for r in values)/len(values),
                 'round_records': [{'round': r['round'], 'detection': r['detection'],
                                    'evaluation': r['evaluation']} for r in values]}
           for name, values in rounds.items() if values}
matched = {name: pooled(values[:paired_count]) for name, values in rounds.items()}
report = {'status': 'interim_not_final', 'generated_utc': datetime.now(timezone.utc).isoformat(),
          'identity': provenance['identity'], 'training_seeds': [cfg['seed']], 'data_splits': 1,
          'conditions': summary, 'available_paired_rounds': paired_count,
          'available_pair_checks': 'pass', 'matched_prefix_detection': matched,
          'matched_prefix_h_fpr_difference': matched['mix04']['heterogeneous']['rate']-matched['mix0']['heterogeneous']['rate'],
          'initial_evaluation': read(root/'initial_evaluation.json'),
          'inputs_sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for name in rounds for p in sorted((root/name).glob('round_*.json'))},
          'limitations': ['Interim snapshot: the pre-specified 10-round paired comparison is incomplete.',
                          'One seed and one shared data split; client-rounds are correlated.',
                          'No independent unfiltered attack control; low ASR is not defense success.',
                          'Paper-formula implementation with disclosed choices, not exact reproduction of the published tables.',
                          'All completed rounds retained; no tuning or outcome-based selection.']}
(root/'interim_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
lines = ['# IndexGuard 配对预实验：阶段记录', '',
         '这是尚未完成的快照；正式结论须等待两组各 10 轮、本地完整归档验证和最终配对核验。', '',
         '| 条件 | 完成轮次 | 指定 H 组误报 | 普通良性误报 | 恶意检出 | 最新 SST-2 准确率 | 最新 SST-2 ASR |',
         '|---|---:|---:|---:|---:|---:|---:|']
for name, info in summary.items():
    d, e = info['pooled_client_round_detection'], info['latest_evaluation']['sst2']
    cells = [f"{d[role]['flagged']}/{d[role]['n']} ({d[role]['rate']:.2%})" for role in ['heterogeneous', 'benign', 'malicious']]
    lines.append(f"| {name} | {info['completed_rounds']}/10 | {' | '.join(cells)} | {e['accuracy']:.2%} | {e['asr_non_target']:.2%} |")
lines += ['', '两行的最新测试结果来自不同轮次，不能当作最终配对对照。mix0 中 H 组只是相同客户端 ID 的对照，尚未替换 IMDB 数据。', '',
          f"当前可配对的前 {paired_count} 轮通过参与 ID、角色、投毒位置、共享 K、首轮初始化及未干预客户端首轮适配器哈希核验。该共同前缀 H-FPR 差值为 {report['matched_prefix_h_fpr_difference']*100:+.2f} 个百分点，只是阶段描述。", '',
          '逐轮有误报的记录：']
for name, values in rounds.items():
    for r in values:
        ids = [x['id'] for x in r['clients'] if x['predicted_malicious'] and x['role'] != 'malicious']
        if ids:
            lines.append(f"- {name} 第 {r['round']} 轮：误报良性客户端 {ids}。")
lines += ['', '误报率必须与恶意检出率、后门成功率一起解释。没有检出恶意客户端时，零误报不能说明防御有效。', '',
          '仅一个训练种子和一个共享划分，不作显著性或普遍性判断；本批没有独立无防御攻击基线。实现差异和 warm-up 选择见 `../../docs/indexguard-heterogeneity-20261009.md`。未调参、未筛选结果、未改变正在运行的冻结训练代码。', '',
          '完整数据见 `interim_summary.json` 和各组逐轮 JSON；本文件由同目录 `summarize_interim.py` 生成。']
(root/'INTERIM.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'completed': {k:v['completed_rounds'] for k,v in summary.items()},
                  'paired_checks': 'pass', 'paired_rounds': paired_count,
                  'matched_h_fpr_delta': report['matched_prefix_h_fpr_difference']}))
