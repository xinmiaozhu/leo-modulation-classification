# 三组信道边界对照

入口：`scripts/22_run_channel_boundary.py`。该实验沿用 Protocol A 的
coherent-grid 导频载波估计、补偿 I/Q、exact-mixture 描述符和
`dualnet_iq_evm.yaml` 分类网络，不修改已有论文实验或 checkpoint。

## 第一组：频率选择性强度

比较 flat、short、reference、long、strong_echo。前三种多径时延谱为
`[0,1,2]`、`[0,3,7]`、`[0,8,24]` 个采样点，功率谱为 `[0,-6,-10]` dB；
strong_echo 固定 reference 时延，改为 `[0,-1,-3]` dB。
flat 是三条零时延路径 `[0,0,0]` 的相干叠加，相对功率为 `[0,-6,-10]` dB；
与 reference 配对使用同一组路径随机相位。每个谱按平均总路径功率归一化，
不逐帧归一化合成抽头幅度。
既报告采样时延，也报告 RMS 时延 / 符号周期，不能把采样点当符号数。

每种信道比较不均衡、静态 Pilot-LS 均衡和 Oracle 算子均衡。
exact-mixture 判决不涉及训练；神经网络使用同一个平坦信道训练的 checkpoint，
其结果属于冻结接收机的信道失配测试。

## 第二组：帧内时变多径

固定 reference 的时延与路径功率，以相对公共载波的路径 Doppler
`[0,0,0]`、`[0,2,-3]`、`[0,10,-15]` Hz 对比。
模型为 `h_l[n] = c_l exp(j 2 pi nu_l n/fs)`，初相位随机且跨条件配对。
这是确定性路径 Doppler 的受控时变模型，不代表完整 Jakes/3GPP 场景。

公共载波相位先于多径作用，随后添加噪声。Oracle 通过有限帧线性算子
`D_hat^H H D_hat` 的正则化最小二乘恢复信号，不使用真实载波参数去修正
估计误差。迭代状态写入 HDF5，汇总报告触及迭代上限的比例。
Oracle 使用最多 1000 次 LSQR 迭代、1e-7 停止容差。初次筛查中强回波在
200 次上限下存在未收敛样本，因此只提高数值求解预算重算该条件；最终报告
必须检查收敛状态。此修正不改变正则化或根据测试标签选择接收机参数。
静态 Pilot-LS 在时变信道中的下降本身就是接收机边界；并未为其增加导频或跟踪模块。

## 第三组：训练—测试失配

分别在 flat、reference、tv_slow 条件训练，所有 checkpoint 仅按各自验证集选取。
用相同结构、训练预算、模型种子，测试全部七种信道。此矩阵固定不均衡前端，
避免把训练分布变化和均衡器变化混在一起。

- flat → reference/tv_slow：平坦训练的失配。
- reference → reference、tv_slow → tv_slow：对应条件的匹配训练。
- reference/tv_slow → long/strong_echo/tv_fast：未见时延、功率谱和时变强度。

## 配对、数据划分与指标

每帧以数据种子、split、类别、SNR 位置和重复序号构造 `pair_key`；参数、
符号、信道初相位、噪声使用独立随机流。改变路径数不会改变后续帧的符号或噪声。
跨条件使用相同标准复高斯噪声样本；噪声幅度按各自接收端无噪声帧功率缩放。
所以 SNR 是每帧接收信号功率 / 期望噪声功率，不包含平均衰落造成的 SNR 损失。
train/val/test 使用不同随机流。配对的各条件不是额外独立样本。

固定参数：7 类调制，200 kHz 采样，8 samples/symbol，1024 符号，
8192 点 / 40.96 ms 帧长，64 个 comb pilots（6.25%），公共 CFO=0，
公共 Doppler-rate 均匀分布于 [-8160,-180] Hz/s。
Pilot-LS 对所有信道固定假设 25 点支撑，ridge=0.01，逆滤波正则化=0.001；
Oracle 使用同一正则化强度和真实路径参数。所有配置在观察测试结果前固定。

输出包括：

- `exact_mixture.csv`：各 SNR 下的描述符判决准确率、导频 EVM、有效率。
- `neural_accuracy.csv`：分类网络的完整训练/测试矩阵及均衡对照。
- `neural_seed_summary.csv`：模型种子间的均值、标准差与种子数量。
- `channel_diagnostics.csv`：RMS 时延、路径 Doppler×帧长、载波估计 RMSE/P95、
  Pilot-LS 的去除全局复增益歧义后的整帧信道轨迹 NMSE、Oracle 求解状态。
- `paired_equalization_gains.csv`、`paired_training_gains.csv`：配对准确率差和
  帧 bootstrap 95% 区间；区间是给定训练模型的测试样本不确定性，不代表训练种子不确定性。
- `manifest.json`、`logs/`、`predictions/`：完整参数、源代码 SHA256、命令、日志、逐帧预测。
- `channel_boundary.pdf/png`：三组结果概览。单点总体数值是配置中各 SNR 的等权结果。

## 运行

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts/22_run_channel_boundary.py
```

默认 screening：SNR=-4/5/14 dB，每类每 SNR 为 train=8、val=4、test=10；
每种信道 168/84/210 帧，模型种子 41，最多 12 epochs，CPU 4 线程。
这是用于发现失败模式和验证实验链路的初步实测，不能作为收敛充分、多种子的大规模论文证据。
重复命令跳过已成功步骤；修改实验配置必须使用新输出目录。源代码变更后应使用新目录，
避免把旧特征与新实现混合。中断的单个训练步骤会从头重跑，保证相同训练预算。

可扩大样本量与种子数执行完整实验：

```powershell
.\.venv\Scripts\python.exe scripts/22_run_channel_boundary.py `
  --config configs/experiment/channel_boundary_full.json `
  --output-dir outputs/channel_boundary_full --device cpu
```

完整配置为原 11 个 SNR 点、每类每点 200/100/100 帧、5 个模型种子、最多 100 epochs。
仅用于测试的 short/long/strong_echo/tv_fast 不生成 train/val；测试随机流保持一致。
因此训练条件各 30,800 帧，四个仅测试条件各 7,700 帧，总计 123,200 帧，
包括每种信道 7,700 个测试帧。训练 15 个模型，执行 175 次分类评估。
在支持 CUDA 的 PyTorch 环境中改用 `--device cuda`。它需要较多磁盘与训练时间；
提供配置不意味着已经完成该规模的实验。

还可用 `--stages prepare`、`--stages train eval summary` 分阶段运行。

## 在相同配对数据上增加训练种子

本机另有 `.venv-cuda` 环境，可调用 RTX 4060。三种子验证配置保留上述小样本
数据，使用 41/73/107 三个模型种子、最多 50 epochs（验证集 early stopping），
独立存储模型和预测，只读复用已生成的信道和表征：

```powershell
.\.venv-cuda\Scripts\python.exe scripts/22_run_channel_boundary.py `
  --config configs/experiment/channel_boundary_multiseed.json `
  --output-dir outputs/channel_boundary_multiseed --device cuda `
  --reuse-prepared-from outputs/channel_boundary_screen --stages train eval summary
```

复用时校验数据生成配置完全相同，并保存源目录与 manifest 摘要。
这个三种子结果仍属于小样本验证；增加模型种子并不增加独立测试帧数。

## 完整实验的后台运行与恢复

Windows 上可启动带持久日志的隐藏进程：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/23_start_channel_boundary_full.ps1
Get-Content outputs\channel_boundary_full\run_progress.json
Get-Content outputs\channel_boundary_full\runner.stdout.log -Tail 10
```

启动器记录 PID 和进程启动时间，拒绝重复启动仍存活的同一实验。
`run_progress.json` 记录当前步骤、日志位置或失败原因；`status.json` 是最终
结果完整性审计，只有完整实验结束后才能将它作为完成证据。
已完成步骤按代码、输入文件大小/修改时间和命令校验后跳过；修改输入文件时
不要人为保留其修改时间。中断训练仅在配置和输入指纹相同且有 `last.pt` 时续跑，
否则从头训练。与早期筛查版本相比，当前运行器已加入输入依赖校验和训练恢复，
原始筛查结果的旧缓存标记可能触发重算；仅查看旧结果无需重跑 prepare/train。

## 与基线参考图同风格的单坐标结果图

只读已有分类结果绘图，不重新运行实验：

```powershell
.\.venv\Scripts\python.exe scripts/51_plot_channel_boundary_paper.py
```

输出为 `outputs/figures/paper/channel_boundary_representative_accuracy_vs_snr.pdf`，
同时保存 600 dpi PNG、曲线 CSV 和样式 JSON。运行器 51 的 `plot_results`
也使用相同绘图函数，生成单坐标版本的 `channel_boundary.pdf`。

字体为 Times New Roman；轴标题 10 pt，横刻度 8.1 pt，纵刻度 9 pt，图例
8.5 pt；坐标区宽高比 10:9，PDF 页面 210.088 × 184.58284 pt，与用户提供的
`paper_baseline_protocol_a_seed41_accuracy_vs_snr.pdf` 可见页面一致。
沿用参考配色、点划网格、线宽和标记。图例采用两列以避开低准确率曲线；
代表性曲线分离明显，因此不设置局部放大框。

图例按 **Training / Test** 解释：Flat=平坦信道，Static=参考静态多径，
Long=未见长时延，Slow/Fast=慢/快时变多径。七条曲线分别是 Flat/Flat、
Flat/Static、Static/Static、Flat/Long、Static/Long、Flat/Fast、Slow/Fast。
全部为相同无信道均衡前端（仍有载波补偿）的分类网络结果，按五个模型种子
取算术平均。每个 SNR 使用相同的 700 个测试帧；五次模型评估不等于 3500 个
独立测试样本。CSV 同时保留种子间标准差，主图不绘制阴影带。

可用英文图注：Classification accuracy versus SNR under channel mismatch.
Legend entries denote training/test channel conditions. All curves use the
carrier-compensated I/Q–constellation classifier without channel equalization
and are averaged over five model seeds. Static and Long denote reference and
unseen long-delay multipath; Slow and Fast denote slow and fast time-varying
multipath, respectively.


## Updated single-figure run (2026-10-05)

Run `.venv-cuda/Scripts/python.exe -u scripts/24_run_channel_boundary_figure.py`.
Configuration: `configs/experiment/channel_boundary_figure.json`.
The run retains 5 model seeds, 11 SNR points, 200/100/100 train/validation/test
frames per class and SNR, and the original 100-epoch budget with early stopping.
Only the five channels and seven unequalized neural curves needed by the paper
figure are computed. Three coincident paths with powers [0,-6,-10] dB form Flat;
Static uses the same paired path coefficients at delays [0,3,7]. Noise remains
scaled to each frame's received power. CFO compensation is explicitly enabled.
Dense channel metadata stores coincident paths as one effective delay to prevent
double counting in downstream operators.

Artifacts and resumable completion markers are isolated in
`outputs/channel_boundary_flat_multipath`. After all 35 evaluations pass pairing
and completeness checks, the runner backs up the old figure and replaces its
PDF, PNG, CSV and JSON under `outputs/figures/paper`. A failed or incomplete run
does not replace the existing paper figure.


## Expanded test-only evaluation (2026-10-06)

Launch with `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/25_start_channel_boundary_figure.ps1 -ExpandedTest`.
The configuration `channel_boundary_figure_test1000.json` increases test frames
from 100 to 1000 per class/SNR (7000 per SNR; 77000 per channel).
All 15 completed checkpoints from `outputs/channel_boundary_flat_multipath` are
reused, with checkpoint hashes recorded; no training or validation data change.
The original test repetitions 0..99 are retained; repetitions 100..999 are new
independent draws. Pairing across channels and independent draws across SNR
remain unchanged. Each test key and class/SNR count is checked before evaluation.
Expanded artifacts live under `outputs/channel_boundary_flat_multipath_test1000`.
The original figure is backed up and replaced only after all evaluations succeed.
