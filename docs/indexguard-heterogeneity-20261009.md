# IndexGuard 异质良性客户端配对预实验

分支：`experiment/indexguard-heterogeneity`。本协议在正式训练前冻结；旧的 clean-pair 和 attack-pilot 不续跑。本批仅 seed=2026、mix0 / mix04 两组、各 10 轮，属于初步压力测试，不是论文全量复现。

## 问题与可否定假设

在有真实后门攻击者的条件下，保持标签分布和训练样本量不变，对少数良性客户端加入同任务、不同域数据，IndexGuard 是否更容易误报这些客户端？接受误报率不升高、为零或其他不支持原假设的结果。不按误报率选择轮次或重新调参。

## 固定设置

| 项目 | 本批设置 |
|---|---|
| 骨干 / 适配器 | Llama-3.2-3B 基础模型，Q/V LoRA，r=8，alpha=128，dropout=0.05，BF16 |
| 任务 | SST-2 情感分类；异质域 IMDB，标签均为 negative / positive |
| 客户端 | 30 个，全员每轮参与；恶意 ID 0–5；指定异质良性 ID 24–29；其余 18 个普通良性 |
| 样本 | 每客户端 500 条；各客户端二分类比例由 Dirichlet(0.3,0.3) 生成后固定；源样本去重 |
| 对照 | mix0 的全部客户端使用 SST-2；mix04 只在 6 个指定异质客户端各替换 200 条同标签 IMDB，逐位置保持标签 |
| 攻击 | 6 个恶意客户端各固定 25 条投毒（5%）；前置共同 InsertSent 触发句；目标 positive；原本 positive 也可能被抽中 |
| 触发句 | I watched this 3D movie with my friends last Friday. |
| 训练 | 每客户端每轮 3 epochs；micro-batch 4，累积 16；AdamW lr=2e-4、betas=(0.9,0.99)、wd=.01；梯度裁剪 1；3% 线性 warmup 后线性下降 |
| 测试 | 各域独立平衡 200 条，配对共享，与训练按标准化全文哈希去重；每域 100 条非目标类计算触发 ASR |
| 检测 | Q/V LoRA 参数相对当轮全局适配器的 delta，绝对值 Top-K 索引，IOS，average-link 层次聚类 L=2 |
| 可疑簇 | 非最大簇、至少 2 人、cohesion ≥ 各有效簇 cohesion 中位数 + .05，且大小 ≤ .7 × 最大簇 |
| 聚合 | 判定后仅对接受的客户端做样本量加权 FedAvg，过滤决定实际影响下一轮 |

两个条件使用相同初始化、相同客户端顺序、同轮同客户端训练随机种子和相同投毒位置；第 2 轮起全局模型可因过滤/数据差异而分化，这是闭环效应。保存每个客户端局部适配器、Top-K 索引、聚合前后全局状态和判定。

## 一次性 K 校准

正式训练前仅在 mix0 的训练数据进行共享校准，不访问测试集或角色真值。每个客户端（包括攻击者）从其实际训练集合独立有放回抽 3 × 64 条，每次从相同初始化做一个 AdamW optimizer step（固定名义学习率）。以三次绝对 delta 的均值计算 90% L1 质量覆盖 K，以平均两两 Top-K 交集比例 ≥ .8 检验稳定性。在 1% 间隔候选网格加入精确覆盖 K 和 M，选最小合格候选；全局 K 取客户端 K 的中位数向上取整，冻结并用于两组所有轮次。

若校准得到很大的 K（甚至 M）或检测失去区分度，保留并报告，不用研究指标重新选 K。一步 LoRA 校准可能只有 B 分量非零，这是本次 warm-up 具体化的限制。完整校准向量、抽样位置和配置均保留。

## 与论文 / 作者公开代码的关系

依据 [论文 §5、Appendix A/F 和 Table 8](https://proceedings.mlr.press/v306/dogani26a.html) 独立实现核心筛查公式。作者代码检查版本为 `6e745f81cdee8ea56830bcc1dce719259430954e`，审计材料记录在本分支。以下差异必须随结果披露，不能称“严格复现原表格”：

- 论文没有给出唯一可直接重放的全部 warm-up 参数/数据划分，本批明确固定了上面的操作规则。固定每客户端 500 条的 Dirichlet 标签偏斜，与作者按类别跨客户端分配的 Dirichlet 划分不完全相同。
- 新增 SST-2 → IMDB 的同标签域替换，是本研究干预；仅一个训练种子、一个数据划分、10 轮。
- 用两个候选标签（含 EOS）的总负对数似然进行生成式分类。固定 `Review: ...\nSentiment:` 模板；采用 eager attention 和 gradient checkpointing。未部署密码学安全聚合；单机模拟器保存原始更新以供审计，检测函数只接收索引。
- 公共代码该版本存在无异常时强制选簇的 fallback、按真值翻转 0/1 以取更高 detection accuracy 的报告逻辑，以及在检测前聚合全部更新的运行路径。本实现依论文公式允许无可疑簇，固定 1=恶意的语义，并在聚合前过滤。该版本审计不等于对论文全部实验代码或结论的判定。
- 非唯一实现细节：坐标并列时按规范 ID 稳定排序；最大簇并列选最小簇编号；singleton 不参与 cohesion 中位数；偶数簇使用通常的中位数（中间两数均值）。
- 模型从公开 ModelScope `LLM-Research/Llama-3.2-3B` 的固定 revision 获取，逐文件验证分发方 SHA-256，记录完整清单；不把分发方 revision 冒称 Meta 的 Hugging Face revision。

## 报告与解释

每轮记录指定异质组 H-FPR=误报 H 数/6、普通良性 FPR=误报普通数/18、总 FPR=FP/24、恶意 recall=TP/6、precision/F1、混淆计数、接受权重、各域 clean accuracy 和非目标触发 ASR。mix0 中也保留相同 6 个“指定异质”ID 作为配对参照。没有正预测时 precision 为 null，不伪装成 100%。本规则输出硬判定，本批不杜撰连续异常分数来报告 ROC-AUC。

主要比较：全部 10 轮的 H 客户端-轮误报计数 / 60，以及逐轮差值；次要报告末轮和普通 FPR、recall、ASR。不能把 60 次相关判定当 60 个独立随机种子做显著性检验。没有独立无防御闭环攻击基线，低 ASR 本身不足以证明防御成功；同时报告原始模型 ASR 和恶意检出情况。

测试集只用于预定报告，不用于 K/阈值/超参选择。软件缩步检查标记 `not_research_evidence=true`。不因为“不符合预期”中止、重选种子或丢弃结果。

## 运行、恢复与保存

`PYTHONPATH=src:. python -m experiments.indexguard.data --config configs/indexguard/pilot.json` 准备配对数据；`python -m experiments.indexguard.runner --config ... --smoke` 做真实模型缩步检查。正式运行 `bash scripts/run_indexguard.sh`，后台独立于 SSH。

内核 flock 防止重复队列。每个局部客户端完成即原子落盘；每轮完成后保存全局 checkpoint、逐轮 JSON 和 final adapter。恢复要求配置、源码文件、数据、模型清单和关键环境的 identity 完全一致。STOP 文件或 SIGTERM 在当前客户端完成后停机并保留断点；不会自动清除 STOP。

GitHub 只保存代码、配置、逐样本来源索引、校验、逐轮客户端判定、分析结果。含文本数据、完整更新、适配器的归档保存服务器并下载本地核验，GitHub 保存归档清单/哈希和回执。源码冻结后只发布结果，不让结果提交改变服务器训练源码。每组完成后备份，全部完成后做配对核验与描述性分析。实例不自动关机。
