# 论文开源文件分类与发布清单

核查日期：2026-09-30。依据：当前 `main.tex` 正文、项目说明、配置、实验入口、文件目录及部分最终结果汇总。本清单按论文复现需要分类，不代表已经重新运行实验或逐行审计全部实现；也未确认同目录 PDF 与 TeX 的版本完全一致。未移动、删除原文件，未修改 Git 暂存区。

## 1. 开源范围应覆盖什么

论文最终接收机为：已知公共导频 → 载频偏差/多普勒率估计 → 补偿 → I/Q 与 48 维全候选星座描述符 → 双流融合分类。HOC 用于机理诊断及 NASA 基线，不是最终提出网络的输入。

完整复现应同时覆盖：

- 第 II 节：LEO 几何、波形、导频、信道与损伤仿真。
- 第 III 节：Fresnel 衰减、特征退化、残余误差容限校准。
- 第 IV 节：Protocol A 一维相干搜索；Protocol B DFRFT 等价捕获与局部相干精化；48 维描述符；双流网络。
- 第 V 节：五种子主对比、表示消融、前端归因、理想化似然参考、多径均衡、整轨留出、复杂度与绘图。

推荐采用“源代码仓库 + 精选结果 + 独立数据/权重附件”。不建议直接上传整个工作目录。

## 2. 第一类：上传到代码仓库

| 当前文件或目录 | 作用与处理建议 |
|---|---|
| `src/` 中全部 `.py` | 保留公共实现与包结构，排除缓存。包含信号、物理、数据、估计补偿、网络、训练、绘图和工具模块。 |
| `scripts/01`—`51` 的全部现有 `.py`、`.ps1` | 按任务顺序编号，主线、诊断与扩展见 [脚本索引](../scripts/README.md)。编号是索引，不意味着每次按 01—51 顺序执行。 |
| `configs/model/dualnet_iq_evm.yaml` | 最终双流模型配置，必须明确指定。 |
| `configs/model/dualnet_iq_only.yaml`、`dualnet_evm_only.yaml` | 两个单流消融。 |
| `configs/model/paper_*.yaml` | CNN2、MCNet、CNN-LSTM、IQCNet、NASA HOC-NN、STARNet 基线。 |
| `configs/train/train_long.yaml` | 论文主实验训练配置：初始学习率 0.0005，batch size 128，最多 100 epochs，早停 patience 25。 |
| 其他 `configs/**/*.yaml` | 保留并标注示例/历史/扩展用途；不要将默认示例冒充论文最终配置。 |
| `tests/*.py` | 保留现有调制、同步、数据划分、特征、网络和训练等回归检查。 |
| `README.md`、`requirements.txt`、`setup.py`、`.gitignore` | 必须保留，但需要完成第 7 节所列修订。 |
| `docs/starnet_reproduction.md` | 保留基线来源及适配说明。 |
| `experiments/PRACTICAL_PROTOCOL.md` | 保留并更新四路划分和最终 Protocol B 路径说明。 |
| `data/*/.gitkeep`、`outputs/*/.gitkeep` | 可保留目录占位。 |

不要根据名称直接删除 `src/drc_hoc/` 或 `src/models/hoc_stream.py`。前者同时包含最终导频估计、补偿等实现；后者仍被 `drc_dualnet.py` 导入。`drc_dualnet` 是历史类名，README 已说明其用于 checkpoint 兼容；是否为双流由配置决定。

尤其要保留 `scripts/10_precompute_symbol_constellation.py`：48 维 exact-mixture 描述符的重要实现就在该脚本中，只上传 `src/` 会遗漏论文核心方法。

## 3. 实验脚本按论文内容归类

表中编号已按 2026-10-06 的任务排序更新，均对应 `scripts/` 下唯一同编号文件，完整名称见 [脚本索引](../scripts/README.md)。

| 脚本编号 | 归类 | 与论文的关系 |
|---|---|---|
| 01—03 | 数据生成 | 通用数据、轨道数据、SNR 均衡数据；需补全各正式数据集的生成参数。 |
| 04—05 | 划分与审计 | 整轨留出及泄漏检查。 |
| 06—08、10 | 接收机前处理 | 导频参数估计、联合精化、补偿 I/Q 和星座描述符；具体顺序以 A/B 指南为准。 |
| 09 | 多径均衡 | 补偿后、描述符提取前的可选 Pilot-LS 与 oracle 对照。 |
| 11—12 | HOC 与基线特征 | 理论诊断、NASA HOC-NN 等，不能作为最终提出方法第三分支。 |
| 13—14 | 训练与评估 | 所有模型的公共入口。 |
| 15—17 | 基线与多种子主实验 | A 的 proposed/MCNet 用 15，STARNet 见 16；最终 B 使用 17。 |
| 18—20 | 消融 | A 表示消融用 18 的 A 路径；最终 B 用 19；20 为导频权重消融。 |
| 21—26 | 泛化与信道实验 | 整轨/仰角留出、完整信道边界实验、七曲线实验及其启动器、多径均衡恢复统计。 |
| 27—30 | 前端选择与归因 | 验证集筛选、相干搜索对照、独立测试汇总、最终产物审计。 |
| 31—33 | 物理与理论验证 | 轨道几何、Fresnel 界、特征退化。 |
| 34—36 | rate-only 校准 | 速率符号与 CFO 敏感度检查、bootstrap 和操作容限。 |
| 37—40 | 联合误差校准 | eta 扫描、分片合并、bootstrap 和操作容限。 |
| 41—42 | 复杂度 | 网络与描述符耗时；区分 A/B 和前处理/网络/E2E。 |
| 43—51 | 可视化 | 框架、导频误差、混淆矩阵、SNR 准确率、基线、消融、容限和信道边界。 |

关键版本区别：`15_run_exact_mixture_seeds.py` 也有 B 路径，但读取旧 `data/features/exact_mixture/protocol_b_symbol_exact.h5`；最终 B 主实验 `17_run_hybrid_dfrft_seeds.py` 读取 `data/features/hybrid_dfrft/protocol_b_symbol_exact.h5`。`18` 与 `19` 的 B 消融同理。不能把旧 B 汇总当成论文最终结果。

## 4. 第二类：精选后上传的小型结果和复现元数据

结果文件不能只按扩展名决定是否上传。CSV/JSON 可能是最终证据，也可能是过时实验。建议新建 `paper_results/`，仅复制已核对的文件，并保留原相对路径或提供路径映射。

| 当前来源 | 建议选入的内容 |
|---|---|
| `outputs/results/hybrid_dfrft/protocol_b/` | 最终 B 主对比：`seed_runs.csv`、`seed_summary.csv`、汇总 JSON、各方法各 seed 的测试 CSV 与 `.summary.json`。 |
| `outputs/results/hybrid_dfrft/representation_ablation/protocol_b/` | 最终 B 的 I/Q-only、constellation-only、fusion 消融。 |
| `outputs/results/exact_mixture/protocol_a/` | A 的 proposed、MCNet 五种子结果。 |
| `outputs/results/exact_mixture/representation_ablation/protocol_a/` | A 的表示消融。 |
| `outputs/results/starnet_seeds/` | 筛选与 A 主协议一致的 STARNet 五种子产物。 |
| `outputs/results/frontend_attribution/` | 选入论文前端归因的测试汇总、验证选择证据和成对 bootstrap 依据。 |
| `outputs/results/theory/fresnel_margin/`、`leo_orbit_geometry/` | 理论及轨道验证的小型数值结果。后一路径同样相对 `outputs/results/`。 |
| `outputs/results/hybrid_dfrft/`、`outputs/results/exact_mixture/` 内校准文件 | 根据冻结 checkpoint 和 summary 选出论文使用的 gamma/eta 扫描、容限与 bootstrap 汇总；不可仅按目录名判断。 |
| `outputs/results/pass_holdout_snr_balanced/proposed/` | matched-pass 与 elevation-OOD 的五种子汇总及逐测试结果。 |
| 最终 checkpoint 同目录的 `run_config.json`、`train_result.json` | 保留真实训练参数、数据来源、seed、最佳 epoch；精选 history 文件可选。 |
| 最终划分 `.npz` | 小文件也应发布或随数据附件发布，并绑定原数据版本和样本顺序。 |

已抽查三组汇总，与正文四舍五入后的数字一致：

- B 主对比：proposed 89.9221%，MCNet 86.2156%，STARNet 75.8182%。
- B 消融：I/Q-only 87.5429%，constellation-only 89.0883%，fusion 89.9221%。
- A 主对比：proposed 90.6727%，MCNet 90.9610%。

这只是对现存汇总的核对，不是重训练验证。其余结果目录应逐项绑定论文图表后再发布。`5mods`、`debug`、`smoke`、旧 HOC/三流结果、旧 penalty 搜索等不建议进入最终论文结果目录。

推荐结果索引字段：论文图/表、协议、生成命令、配置、数据版本、划分、seed、checkpoint、归一化统计来源、结果路径、代码版本。复现成对差异和 bootstrap 时，还需保留对应的逐帧结果及样本标识。

## 5. 第三类：数据与权重作为独立附件

| 文件 | 推荐处理 |
|---|---|
| `data/processed/**/*.h5`、`data/raw/` 大文件 | 发布最终数据附件，或提供完整生成命令与固定环境。不要默认上传全部历史数据。 |
| `data/features/**/*.h5` | 可选预计算附件，降低复现耗时；必须说明对应原始数据和前端版本。 |
| `outputs/checkpoints/**/best.pt` | 优先发布最终 A/B、基线与消融的论文选定权重；快速体验可先提供 proposed seed 41。 |
| `last.pt`、中间 epoch 权重、调试权重 | 通常不发布；如支持续训，可独立提供选定版本。 |
| 大型 bootstrap/扫描数组 | 小汇总入仓库；完整数组可作研究附件。 |

权重必须配套模型配置、7 类标签顺序、48 维特征顺序、训练集归一化参数或 checkpoint 内相应字段说明、数据与前端版本，以及评估命令。单独一个 `best.pt` 不足以复现。

优先固定的现有划分包括：

- `data/splits/leo_7mods_snr_balanced_trainval_pilot_splits.npz`
- `data/splits/leo_7mods_snr_balanced_independent_test_pilot_splits.npz`
- `data/splits/leo_7mods_joint_practical_splits.npz`
- `data/splits/leo_7mods_joint_practical_calibration_splits.npz`
- `data/splits/pass_holdout/` 中与脚本 21 使用的数据一致的 matched/仰角 OOD 划分。

需说明 train/validation/calibration/test 的不同用途；划分索引不能脱离数据顺序单独使用。为正式附件记录文件大小与 SHA-256。

## 6. 第四类：可选论文材料；第五类：不上传

论文材料可选：`main.tex`、`fig1.tex`、最终论文 PDF，以及恢复后的参考文献、补充材料和图文件。它们不属于运行 Python 必需文件。当前 TeX 引用 `references.bib`、补充材料及 `fig1.pdf`、`fig2.pdf`、`fig3(a).pdf`、`fig5(a).pdf`、`fig5(b).pdf`、`fig6.pdf`、`fig7.pdf`、`fig8.pdf`，不能把单独的 `main.tex` 描述成可直接编译的完整论文源码包。

不上传：

- `.venv/`、`.venv-cuda/`、`__pycache__/`、`.pytest_cache/`、`.pyc`。
- `.agents/`、`.claude/` 等本机助手配置。
- `archive/` 中历史清理 ZIP。
- `outputs/backups/`、`outputs/review/` 中旧实验备份和审稿截图。
- 临时 `.pid`、常规运行 `.log`、失败试验和重复中间结果。
- `.git/` 不作为项目文件或压缩包内容上传；仓库历史由 Git 自身管理。

## 7. 发布前已发现的具体问题

1. **README 运行指南缺失。** 当前引用的根目录 `RUN_EXACT_MIXTURE.md`、`RUN_HYBRID_DFRFT.md` 不存在。应补齐从生成数据、估计与特征预计算到训练、测试和绘图的实际命令，不能只留下训练入口。
2. **补充材料和参考文献工作区缺失。** `supplement_material.tex`、`supplement_material.pdf`、`references.bib` 在 Git 状态中显示 `AD`：有暂存版本，但工作区已删除。先确定最终版本，再决定是否纳入发布；本次未恢复或改动暂存区。48 维特征定义依赖补充材料，也可整理为独立文档。
3. **训练配置区分。** `train_default.yaml` 的学习率为 0.001，不是正文的 0.0005；正式多种子脚本使用的 `train_long.yaml` 才一致。这是入口说明问题，不能据此断言主结果训练错误。
4. **轨道协议说明陈旧。** `PRACTICAL_PROTOCOL.md` 当前介绍三路划分（20/30/45/60 训练、75 验证、90 测试），正文最终仰角留出为四路（20/30/45 训练、60 验证、75 校准、90 测试）。应指向最终四路配置和确切生成命令。
5. **忽略规则既有遗漏又过宽。** 当前未排除 `.claude/`、`outputs/backups/`、`outputs/review/`，且这些目录部分文件已经暂存；另一方面，`outputs/results/*` 与 `*.npz` 会忽略需要精选发布的结果和划分。建议将精选结果放入独立目录并为指定划分添加例外。仅修改 `.gitignore` 不会自动取消已暂存文件。
6. **包元信息未定稿。** `setup.py` 仍为 `author="Your Name"`，描述也保留旧 reliability-aware 命名，应改成当前论文表述。
7. **环境未固定。** 当前 requirements 主要是版本下界。增加实际完成论文实验时的依赖版本、Python/PyTorch/CUDA 版本及硬件说明，不能将未知历史环境写成已验证环境。
8. **开源与引用文件待补。** 根目录未见 `LICENSE`、`CITATION.cff`。正式发布应明确许可和引用信息；STARNet 已有移植来源说明，还需核对并保留相应第三方许可/声明。本次未替作者选择许可或判断第三方授权状态。
9. **图表路径待映射。** 正文引用根目录图文件，而当前 `outputs/figures/` 未见直接文件；绘图脚本不能自动等同于已齐全的论文图包。应建立“图号 → 脚本 → 输入 → 输出”的索引。

## 8. 推荐发布结构

```text
repository/
  README.md
  LICENSE                       # 待确定
  CITATION.cff                   # 待补
  requirements.txt
  requirements-lock.txt         # 待从真实复现环境生成
  setup.py
  .gitignore
  src/
  scripts/
  configs/
  tests/
  docs/
    protocol_a.md               # 待补完整步骤
    protocol_b.md               # 待补完整步骤
    data_and_splits.md           # 待补数据定义与划分
    descriptor_48d.md            # 待补逐维定义
    paper_mapping.md             # 待补图表映射
    starnet_reproduction.md
  experiments/
    PRACTICAL_PROTOCOL.md
  paper_results/                # 精选 CSV/JSON，待核对复制
  data/splits/                  # 指定正式划分或附件下载入口
  paper/                        # 可选完整论文材料
```

建议首次发布保留现有 Python 模块和编号脚本，避免为了改名破坏导入、硬编码路径或权重兼容。最优先的工作是补齐运行指南、固定正式配置/数据/权重、筛选论文结果并整理暂存文件。
