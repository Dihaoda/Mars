# Mars 首阶段攻击实验进度

更新时间（UTC）：2026-10-09T11:01:15.664478+00:00。已完成 1/20 项、57/1000 轮。

所有结果均为 seed 2026、同一数据划分的探索性试验。未完成项不报告最终准确率；合成/缩步核验不计入研究结果。

[冻结实验协议](../../docs/attack-pilot-20261009.md) · [工作一迁移审计](../../docs/workone-port.md) · [完整进度 JSON](progress.json)

| 实验 | 状态 | 已完成轮数 | 第50轮干净准确率 | 后门 ASR |
|---|---|---:|---:|---:|
| [hfedsa_ddpg-raw-none-seed2026](hfedsa_ddpg-raw-none-seed2026/summary.json) | complete | 50/50 | 93.16% | — |
| hfedsa_ddpg-effective-none-seed2026 | running | 7/50 | — | — |
| fedavg-effective-none-seed2026 | pending | 0/50 | — | — |
| fltrust-effective-none-seed2026 | pending | 0/50 | — | — |
| rfa-effective-none-seed2026 | pending | 0/50 | — | — |
| hfedsa_ddpg-raw-label_flip-seed2026 | pending | 0/50 | — | — |
| hfedsa_ddpg-effective-label_flip-seed2026 | pending | 0/50 | — | — |
| fedavg-effective-label_flip-seed2026 | pending | 0/50 | — | — |
| fltrust-effective-label_flip-seed2026 | pending | 0/50 | — | — |
| rfa-effective-label_flip-seed2026 | pending | 0/50 | — | — |
| hfedsa_ddpg-raw-backdoor_scale-seed2026 | pending | 0/50 | — | — |
| hfedsa_ddpg-effective-backdoor_scale-seed2026 | pending | 0/50 | — | — |
| fedavg-effective-backdoor_scale-seed2026 | pending | 0/50 | — | — |
| fltrust-effective-backdoor_scale-seed2026 | pending | 0/50 | — | — |
| rfa-effective-backdoor_scale-seed2026 | pending | 0/50 | — | — |
| hfedsa_ddpg-raw-adaptive-seed2026 | pending | 0/50 | — | — |
| hfedsa_ddpg-effective-adaptive-seed2026 | pending | 0/50 | — | — |
| fedavg-effective-adaptive-seed2026 | pending | 0/50 | — | — |
| fltrust-effective-adaptive-seed2026 | pending | 0/50 | — | — |
| rfa-effective-adaptive-seed2026 | pending | 0/50 | — | — |

每轮客户端判定、权重及控制器轨迹保存在各实验的 rounds.json；完成后增加 summary.json 和备份清单；完整 CSV 随备份保存。
二分类检测指标只适用于输出恶意后验的 H-FedSA 方法。其他方法记 NA。实例不会自动关机，队列不含后续种子或新方法。
