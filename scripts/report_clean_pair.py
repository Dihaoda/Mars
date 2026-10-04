"""Render descriptive tables and a paired scientific figure from verified records."""
from pathlib import Path
import argparse
import csv
import json
import statistics as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='results/clean-pair-20261004')
    args = parser.parse_args()
    out = Path(args.directory)
    status = json.loads((out / 'analysis_status.json').read_text())
    assert status['status'] == 'complete'
    with (out / 'per_seed.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (out / 'replay_summary.csv').open(newline='') as handle:
        replay = list(csv.DictReader(handle))
    lookup = {(int(r['seed']), r['representation']): r for r in rows}
    groups = json.loads((out / 'group_summary.json').read_text())['groups']
    paired = json.loads((out / 'paired_summary.json').read_text())['statistics']
    metrics = [('heterogeneous_rate', '正常异质客户端误判率', 100, '%'),
               ('benign_rate', '普通客户端误判率', 100, '%'),
               ('heterogeneous_weight_ratio', '异质/普通平均权重比', 1, ''),
               ('test_macro_accuracy', '最终宏平均准确率', 100, '%')]
    by_rep = {g['config']['defense']['representation']: g for g in groups}
    table = ['| 指标 | raw | effective |', '|---|---:|---:|']
    for key, title, scale, unit in metrics:
        cells = []
        for rep in ('raw', 'effective'):
            item = by_rep[rep]['seed_statistics'][key]
            cells.append(f"{item['mean']*scale:.2f} ± {item['std']*scale:.2f}{unit}")
        table.append(f"| {title} | {' | '.join(cells)} |")
    seeds_table = ['| 训练种子 | raw 异质误判 | effective 异质误判 | 差值（百分点） | raw / effective 宏准确率 |',
                   '|---|---:|---:|---:|---:|']
    for seed in (2026, 2027, 2028):
        a, b = [lookup[seed, rep] for rep in ('raw', 'effective')]
        seeds_table.append(f"| {seed} | {a['heterogeneous_flagged']}/{a['heterogeneous_scored_client_rounds']} ({float(a['heterogeneous_fpr'])*100:.2f}%) | {b['heterogeneous_flagged']}/{b['heterogeneous_scored_client_rounds']} ({float(b['heterogeneous_fpr'])*100:.2f}%) | {(float(b['heterogeneous_fpr'])-float(a['heterogeneous_fpr']))*100:+.2f} | {float(a['test_macro_accuracy'])*100:.2f}% / {float(b['test_macro_accuracy'])*100:.2f}% |")
    replay_table = ['| 更新来源轨迹 | 种子 | raw 重放异质误判率 | effective 重放异质误判率 | 差值（百分点） |', '|---|---|---:|---:|---:|']
    replay_stats = {}
    for source in ('raw', 'effective'):
        differences = []
        for seed in (2026, 2027, 2028):
            subset = {r['replay_representation']: float(r['heterogeneous_rate']) for r in replay if r['source_representation'] == source and int(r['seed']) == seed}
            difference = (subset['effective'] - subset['raw'])*100
            differences.append(difference)
            replay_table.append(f"| {source} | {seed} | {subset['raw']*100:.2f}% | {subset['effective']*100:.2f}% | {difference:+.2f} |")
        replay_stats[source] = {'mean_difference_pp': stats.mean(differences), 'sample_std_pp': stats.stdev(differences), 'n_seeds': 3}
    rounds = [json.loads(path.read_text())['summary'] for path in sorted((out / 'records').glob('*/round_*.json'))]
    assert len(rounds) == 60
    errors = [r['aggregation']['svd_relative_error'] for r in rounds]
    supplemental = {'offline_replay_paired': replay_stats,
                    'total_round_seconds_all_six': sum(float(r['round_seconds_total']) for r in rows),
                    'svd_relative_error_mean_60_rounds': stats.mean(errors),
                    'svd_relative_error_min': min(errors), 'svd_relative_error_max': max(errors)}
    (out / 'supplemental_statistics.json').write_text(json.dumps(supplemental, indent=2)+'\n')
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout='constrained')
    colors = ['#0072B2', '#D55E00', '#009E73']
    for ax, metric, title in zip(axes, ['heterogeneous_fpr', 'benign_fpr', 'test_macro_accuracy'],
                                  ['Heterogeneous client FPR', 'Ordinary client FPR', 'Final macro test accuracy']):
        for seed, color in zip((2026, 2027, 2028), colors):
            values = [float(lookup[seed, rep][metric])*100 for rep in ('raw', 'effective')]
            ax.plot([0, 1], values, '-o', color=color, label=str(seed), linewidth=1.8)
        ax.set(xticks=[0, 1], xticklabels=['Raw', 'Effective'], title=title, ylabel='Percent', xlim=(-.3, 1.3))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylim(0, 55)
    axes[1].set_ylim(0, 55)
    axes[2].set_ylim(91.5, 94.5)
    axes[2].legend(title='Training seed', fontsize=8)
    fig.suptitle('Closed-loop comparison: 3 paired seeds, one data split, no attacks', fontsize=12)
    fig.savefig(out / 'paired_results.png', dpi=180)
    fig.savefig(out / 'paired_results.svg')
    plt.close(fig)
    report = '''# 首批无攻击配对实验结果（2026-10-04）

6 次预先指定的实验全部完成，每次 10 轮。有效更新表示在三个训练种子上均降低正常异质客户端的误判率，种子均值从 **38.29% 降至 33.33%**；配对差值为 **−4.96 ± 1.91 个百分点**。但剩余误判仍较高，贡献权重与最终准确率没有一致改善。因此，本批支持把“表示方式与异质性误判”作为继续研究的问题，不能把换用有效更新表示视为完整解决方案，也不能宣称防御能力得到证明。

## 实验设置与核验

- Qwen2.5-0.5B，LoRA rank 4 / alpha 8，q_proj、v_proj，BF16，长度 256。
- 20 个客户端，5 个正常异质客户端；每轮抽取 10 个，训练 10 轮。每客户端 500 条样本，普通客户端使用 SST-2，异质客户端混合 40% IMDB。每轮本地 1 epoch、学习率 2e-4，无步数截断。
- 两组均为固定 beta=0.2 的独立 H-FedSA 迁移实现，使用相同有效更新聚合。唯一配置差异为检测表示 raw / effective；这不是包含原论文 DDPG 的完整复现。
- 训练种子 2026/2027/2028，共用数据种子 42 和同一个划分。可信参考只使用 SST-2 root 128 条；每域最终测试 256 条。模型和数据版本见配置及数据清单。
- 所有配置摘要、源码摘要、模型版本、数据摘要核验通过；每个配对的 10 轮参与列表一致，第一轮基础、参考及所有客户端适配器张量逐元素一致。见 [paired_checks.json](paired_checks.json)。
- 60 份缓存更新各重放 raw/effective 两种评分。自身表示的重放全部复现原始恶意标记，权重差小于 1e-8。无缺失轮次、无防御回退、无选择或丢弃种子。先行一轮测量计入 seed 2026 raw 的 10 轮。

## 闭环训练结果

下表为三个训练种子的均值 ± 样本标准差。每个种子的误判率先汇总其 10 轮；重复出现的客户端轮次不是独立实验样本。权重比定义为异质客户端的平均聚合权重除以普通客户端的平均聚合权重。

''' + '\n'.join(table) + '\n\n' + '\n'.join(seeds_table) + '''

![闭环配对结果](paired_results.png)

异质客户端累计误判为 raw 31/80、effective 27/80，普通客户端为 52/220、51/220；这些合并计数仅用于描述。宏准确率配对变化为 −0.13 ± 0.23 个百分点；不能把这解释为已证明两组等效。异质/普通权重比的配对变化为 +0.009 ± 0.115，seed 2027、2028 的权重比反而下降。有效表示降低硬标记误判，并不保证恢复聚合贡献。

## 同一更新上的离线评分重放

以下每一行的两种评分使用完全相同的已保存更新、参考和随机种子，更新来源轨迹保持固定。它衡量评分差异，不会生成新训练轨迹、测试准确率或攻击成功率。

''' + '\n'.join(replay_table) + f'''

raw 来源轨迹上的配对差值均值为 {replay_stats['raw']['mean_difference_pp']:.2f} ± {replay_stats['raw']['sample_std_pp']:.2f} 个百分点；effective 来源轨迹上为 {replay_stats['effective']['mean_difference_pp']:.2f} ± {replay_stats['effective']['sample_std_pp']:.2f} 个百分点。两类来源分别报告，不能合并当成 6 个独立训练种子。部分种子在某个来源轨迹上没有变化，提示评分收益依赖更新状态；闭环结果还包含聚合影响后续更新的反馈。

## 运行开销与数值范围

六次训练的逐轮耗时合计 {supplemental['total_round_seconds_all_six']/3600:.3f} 小时（不含数据准备、最终评价等，因此不是账单时长）；峰值已分配 GPU 显存约 {max(float(r['peak_gpu_gib']) for r in rows):.3f} GiB。60 轮固定秩聚合的相对 SVD 投影误差平均 {stats.mean(errors):.6f}，范围 {min(errors):.6f}–{max(errors):.6f}。这些是描述性运行记录，不是受控性能基准。

## 结论边界与下一步

只有 3 个训练种子、一个数据划分、一个小模型及一种混合比例；没有恶意客户端，不能评价攻击召回、AUC 或 ASR，更不能宣称安全防御有效。GMM 分量责任度也不等同于校准后的安全概率。当前差异较小，不作统计显著性或广泛泛化主张。

下一阶段应先明确误判是否主要来自参考域偏置、强制三簇语义映射及硬阈值，再设计受控消融：参考域匹配必须给比较组相同参考数据预算；变化因素分开，保留普通与异质客户端误判、贡献权重、任务效用三类指标。之后再加入独立攻击评估和更多数据划分。上述为后续研究建议，本次没有启动新的训练。

## 归档与复现

- 冻结训练提交：`{status['code_commit']}`；源码摘要：`{status['source_hash']}`。
- [预先固定的配置](../../configs/planned/clean-pair/manifest.json)、[实验协议](../../docs/clean_pair_protocol.md)、[逐种子指标](per_seed.csv)、[逐轮原始记录](records)、[重放明细](replay_clients.csv) 均已保存。
- [实际数据子集](dataset/bundle.json) 和 [完整清单](dataset/manifest.json) 包含本次使用的 10,896 条样本、来源行号、文本摘要和划分，数据摘要为 `{status['data_hash']}`。其中 IMDb root 的 128 条已预留但本批未用于训练参考。原始来源为 [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) 和 [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb)；来源与版本保留在清单中。
- [分析脚本](../../scripts/analyze_clean_pair.py) 对完整运行目录执行配对核验及离线重放；[报告脚本](../../scripts/report_clean_pair.py) 生成本报告和图。分析运行成功后再生成报告，不使用测试结果选择参数。
- 完整本地压缩备份另含六组适配器、检查点、60 份更新缓存、tokenizer、数据子集、配置、日志与冻结源码。GitHub 保存可审阅的代码、配置、数据和结果；二进制检查点及更新缓存保存在完整备份中。恢复时在独立目录解压，使用冻结源码和保存的环境依赖；基础模型按固定版本获取，不包含在备份中。
- 完整备份内的 `BACKUP_MANIFEST.json` 记录逐文件 SHA-256，压缩包旁的 `.sha256` 记录整个归档摘要。`backup_verification.json` 另记录本地校验结果，避免压缩包自引用。

```bash
# 在恢复后的 Mars 目录中，使用当时的运行依赖
python scripts/analyze_clean_pair.py
python scripts/report_clean_pair.py
```
'''
    (out / 'README.md').write_text(report, encoding='utf-8')
    print(json.dumps(supplemental, indent=2))


if __name__ == '__main__':
    main()
