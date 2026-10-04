# Mars：联邦 LoRA 安全实验

这是开题报告第二项工作的实验平台。使用单张 GPU 顺序模拟客户端，研究正常异质更新的误判、安全检测与低秩聚合。

**当前是可验证的初始实现，不是论文实验结果。** `hfedsa` 为依据论文思路独立编写的固定 beta 迁移版，未实现原论文 DDPG，不宣称复现原始 LoRA 先导数值。尚未在 AutoDL 上验证的结果不能视为 GPU 实验完成。

2026-10-04 已在 AutoDL RTX 5090 上通过 25 项软件测试、真实 Qwen2.5-0.5B 前向/反向检查和两轮小实验。两轮累计 58.62 秒，PyTorch 分配显存峰值 2.68 GiB；详细记录见 [AutoDL 验收记录](results/autodl-smoke-20261004/README.md)。这是运行验证，不是方法有效性的证据。

## 已实现

- Qwen2.5-0.5B + PEFT LoRA；离线随机微型 Qwen2 使用同一训练代码进行软件测试。
- SST-2 / IMDB 的固定版本下载、平衡划分、全局精确文本去重、客户端样本清单与校验。
- 独立本地优化器；客户端均从同一轮全局适配器开始；只训练 A/B，不训练基础权重。
- 原始因子/有效更新特征；低秩范数与内积、分块精确 L1、QR + 小核心 SVD 聚合。
- `fedavg`、固定 beta `hfedsa`、有效更新 `fltrust`、几何中位数 `rfa`、候选 `anchor_gmm` 和 `hybrid`。
- 标签翻转、触发式后门、有效更新缩放；客户端级与轮次级日志。
- 每轮原子检查点、严格配置/数据/源码检查、断点恢复、离线评分重放。
- 最终测试与训练反馈隔离，客户端真值标签仅供评估，CSV / JSON / PNG 导出。

## 本地软件验证

Python 3.10–3.12。可在独立环境安装 CPU PyTorch；以下测试不下载预训练模型或公开数据。

```bash
python -m pip install -e '.[llm,test,plots]'
python -m pytest -q
python -m mars prepare --config configs/cpu_smoke.yaml --data data/cpu-smoke
python -m mars run --config configs/cpu_smoke.yaml --data data/cpu-smoke --out runs/cpu-smoke
python -m mars plot --run runs/cpu-smoke --out runs/cpu-smoke/diagnostics.png
```

`tiny_qwen` / `synthetic` 均明确标记为软件检查，不能用于论文有效性结论。

Windows 下请将 `HF_HOME` 设为较短的项目缓存路径，避免 datasets 的锁文件超过系统路径长度限制。数据下载失败会明确报错，不会替换成合成数据。

## AutoDL 首轮

项目目录预计为 `/root/autodl-tmp/Mars`。`setup_autodl.sh` 创建项目虚拟环境并复用镜像的 CUDA PyTorch，不替换全局环境。模型和数据缓存位于数据盘。

```bash
cd /root/autodl-tmp/Mars
git pull --ff-only
bash scripts/setup_autodl.sh
bash scripts/autodl_smoke.sh
```

首轮只执行真实模型的前向/反向检查和两轮小实验（每轮 3 个客户端、每客户端至多 2 个优化步），记录时间与显存。其小样本结果不作为论文证据。脚本不会自动开机、关机或启动完整实验矩阵。

实例提供 `/etc/network_turbo` 时，模型运行脚本会加载该网络加速环境；代理凭据不会写入日志。首次模型下载和环境安装耗时不包含在每轮训练计时内。

需要脱离 SSH 运行时：

```bash
mkdir -p runs
nohup bash scripts/autodl_smoke.sh > runs/autodl-smoke.console.log 2>&1 &
```

同一输出目录禁止并发写入。异常终止留下 `RUNNING.lock` 时，先确认其中 PID 对应的旧进程已结束，再移除该锁；普通异常自动清理锁。恢复只接受相同配置、数据与源码，已完成轮次无需重跑。

## 从小实验进入机制实验

```bash
python -m mars prepare --config configs/pilot.yaml --data data/pilot
python -m mars run --config configs/pilot.yaml --data data/pilot --out runs/pilot --until-round 1
python -m mars run --config configs/pilot.yaml --data data/pilot --out runs/pilot --resume
```

参数通过 `--set defense.representation=effective` 等显式覆盖。修改数据比例、客户端数或攻击客户端比例需要独立数据目录；程序会检查，不会悄悄复用错误划分。

```bash
python scripts/plan_experiments.py --suite pilot
python scripts/plan_experiments.py --suite ablation
python scripts/plan_experiments.py --suite attacks
```

以上仅生成配置与清单。`--execute --max-runs 1` 才执行限定数量。pilot 为 9 次独立重建，ablation 为 12 次配对实验，attacks 为 54 次初始候选对比；这些是待验证计划，不是建议一次全部运行。

数据第一次准备时解析并记录上游提交 SHA，已有数据目录后续复用该快照。正式跨设置实验应把 manifest 中的实际 SHA 写入 `data.revisions`，并把首次运行的 `model_revision` 写入 `model.revision`，再重新生成冻结配置。

## 结果与重放

每次运行保存 `metadata.json`、`data_manifest.json`、`environment.freeze.txt`、`round_*.json`、`clients.csv`、`rounds.csv`、`checkpoint.pt`、可选 `updates_*.pt`、`summary.json` 与最终 `adapter/`。

```bash
python -m mars replay --config configs/pilot.yaml --set defense.representation=effective \
  --cache runs/pilot/updates_001.pt --out runs/pilot/replay-effective.json
python -m mars summarize runs/seed2026 runs/seed2027 runs/seed2028 --out runs/summary.json
```

重放只比较相同更新上的评分，不输出最终任务准确率或 ASR。锚点重放需要匹配的双域参考数据和缓存；缓存缺少锚点时，显式使用 `--build-anchors` 才训练参考适配器。随机种子是统计重复单位，客户端跨轮记录仅作描述性计数。

代码、配置、文档、测试纳入 Git；大数据缓存与检查点默认忽略。每次实验的可分享结果和数据清单按 run ID 整理，再选择 Git / Release / LFS 保存，避免将缓存、SSH 密钥或整份研究资料加入仓库。

具体假设、方法差异和未完成的研究范围见 [实验协议](docs/protocol.md)。
