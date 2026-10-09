"""Pre-specified descriptive paired analysis; never treats rounds as iid seeds."""
import json
from pathlib import Path

cfg = json.loads(Path('configs/indexguard/pilot.json').read_text())
root = Path(cfg['results_dir'])
read = lambda p: json.loads(p.read_text())
summaries = {n: read(root/n/'summary.json') for n in ['mix0', 'mix04']}
assert all(s['completed_rounds'] == cfg['rounds'] for s in summaries.values())
assert len({s['identity'] for s in summaries.values()}) == 1
rounds = {n: [read(root/n/f'round_{r:03d}.json') for r in range(1, cfg['rounds']+1)] for n in summaries}
paired = []
for a, b in zip(rounds['mix0'], rounds['mix04']):
    assert a['round'] == b['round'] and a['k'] == b['k']
    assert [c['id'] for c in a['clients']] == [c['id'] for c in b['clients']] == list(range(cfg['clients']))
    assert [c['role'] for c in a['clients']] == [c['role'] for c in b['clients']]
    assert [c['poison_positions'] for c in a['clients']] == [c['poison_positions'] for c in b['clients']]
    if a['round'] == 1:
        assert a['base_hash'] == b['base_hash']
        for x, y in zip(a['clients'], b['clients']):
            if x['id'] not in cfg['heterogeneous_ids']:
                assert x['local_adapter_sha256'] == y['local_adapter_sha256'], x['id']
    paired.append({'round': a['round'], 'h_fpr_difference': b['detection']['heterogeneous']['rate']-a['detection']['heterogeneous']['rate'],
                   'ordinary_fpr_difference': b['detection']['benign']['rate']-a['detection']['benign']['rate'],
                   'malicious_recall_difference': b['detection']['tpr']-a['detection']['tpr']})
pooled = {n:s['pooled_client_round_detection'] for n,s in summaries.items()}
delta = pooled['mix04']['heterogeneous']['rate']-pooled['mix0']['heterogeneous']['rate']
report = {'paired_checks': 'pass', 'training_seeds': [cfg['seed']], 'data_splits': 1,
          'closed_loop_results': summaries, 'paired_round_differences': paired,
          'primary_h_fpr_difference_mix04_minus_mix0': delta,
          'limitations': ['Single seed and shared data split; no significance claim.',
                         'Correlated rounds and clients are not independent experimental replicates.',
                         'No independent unfiltered attack baseline; low ASR alone is not defense success.',
                         'Paper-inspired reimplementation with disclosed warm-up/partition choices, not exact table reproduction.']}
(root/'paired_analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
lines = ['# IndexGuard 异质良性客户端：首批配对结果', '',
         '两组各完成 10 轮，均使用 seed=2026。角色、投毒位置、参与名单、共享 K、首轮初始化和未干预客户端的首轮更新均通过一致性核验。', '',
         '| 条件 | H-FPR（客户端-轮） | 普通 FPR | 恶意 recall | 末轮 SST-2 准确率 | 末轮 SST-2 ASR |',
         '|---|---:|---:|---:|---:|---:|']
for name in summaries:
    d, e = pooled[name], summaries[name]['final_evaluation']['sst2']
    lines.append(f"| {name} | {d['heterogeneous']['flagged']}/{d['heterogeneous']['n']} = {d['heterogeneous']['rate']:.2%} | {d['benign']['rate']:.2%} | {d['tpr']:.2%} | {e['accuracy']:.2%} | {e['asr_non_target']:.2%} |")
lines += ['', f'40% 域替换相对 0% 的 H-FPR 差值为 **{100*delta:+.2f} 个百分点**。该差值描述本次完整闭环训练，不是对同一更新进行离线重评分。', '',
          '只有一个训练种子和一个数据划分；不能把相关的客户端-轮判定当作独立重复，不能做普遍结论或显著性声明。低 ASR 不能单独证明防御有效；本批没有独立无防御攻击基线。未按结果调参或挑选轮次。', '',
          '完整逐轮混淆计数、IMDB 测试、接受权重、聚类信息和配对核验见 JSON。实现与论文的差异见 docs/indexguard-heterogeneity-20261009.md。']
(root/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps({'checks': 'pass', 'h_fpr_delta': delta}))
