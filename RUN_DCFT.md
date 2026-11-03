# Protocol B: DCFT acquisition

The pilot front end returns the DCFT acquisition peak directly. There is no
post-acquisition refinement implementation or enabling option. The CPU and CUDA
paths use the same physical rate/CFO grid. The coherent search is a separate
Protocol-A and attribution baseline.

以下命令在项目根目录、同一个 PowerShell 窗口中按顺序执行，不需要额外的总调度脚本。
这份 PDF 需要更新图 1、3、6 和表 II、III、IV；表 I 只需修改前端名称。
图 2、4、5、7 及 Protocol-A、信道和轨道实验无需重跑。
当前后台任务仍在运行时，不要对相同目录重复启动训练或改写特征。
图片直接覆盖 `outputs/figures/paper/` 中的原文件，不创建前端专用的图片子目录。
表格导出 CSV，不导出 LaTeX 文件。特征、检查点和实验结果仍使用原有路径。

**1. 公共变量。**

```powershell
$py = '.\.venv-cuda\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$raw = 'data/processed/leo_7mods_joint_practical.h5'
$split = 'data/splits/leo_7mods_joint_practical_splits.npz'
$features = 'data/features/dcft_acquisition'
$pilot = "$features/protocol_b_pilot.h5"
$iq = "$features/protocol_b_iqcomp.h5"
$symbol = "$features/protocol_b_symbol_exact.h5"
$checkpoint = 'outputs/checkpoints/dcft_acquisition/protocol_b/proposed/seed_41/best.pt'
$figures = 'outputs/figures/paper'
$paper = 'outputs/results/dcft_acquisition/paper'
$attr = 'outputs/results/dcft_acquisition/frontend_comparison'
$cal = 'outputs/results/dcft_acquisition/calibration'
$calRaw = 'data/processed/leo_7mods_joint_practical_calibration.h5'
$calIQ = 'data/features/leo_7mods_joint_practical_calibration_oracle_iqcomp.h5'
$calSplit = 'data/splits/leo_7mods_joint_practical_calibration_splits.npz'
New-Item -ItemType Directory -Force $figures, $paper, $cal | Out-Null
```

**2. 生成主特征。当前这一步已经完成，可跳过。**

```powershell
& $py scripts/06_precompute_pilot_mu.py --raw-data $raw --output $pilot --no-progress
& $py scripts/28_summarize_frontend_attribution.py --stage features --methods dcft_acquisition
```

已有导频文件时第一条会拒绝覆盖；仅在确实重算全部下游结果时使用 `--overwrite`。
主特征若已完整生成，不必重复调用第二条，以免改变 HOC 缓存的源文件时间戳。

**3. 七种模型、五个种子的训练和测试，供图 6、表 IV 使用。**

```powershell
& $py scripts/16_run_dcft_seeds.py --protocols B --methods proposed mcnet cnn2 cnn_lstm_dual satellite_cnn iqformer starnet --seeds 41 73 107 149 211 --device cuda
```

默认跳过完整结果、从 `last.pt` 继续未完成训练；不要添加 `--overwrite`。
所有模型完成后生成图 6，该命令也生成五种子总汇总表：

```powershell
& $py scripts/47_plot_protocol_b_accuracy_vs_snr.py --protocol B
```

输出 `paper_baseline_protocol_b_accuracy_vs_snr.pdf/.png`。
图 5 如需重新绘制，运行 `& $py scripts/46_plot_paper_baseline_comparison.py`。
两图均使用五种子均值，不再生成内容相同的 `_five_seed_` 文件名副本。
五种子总汇总会核验已有的 Protocol-A 结果，但不重新训练 Protocol A。

**4. 表 III：表示消融。**

```powershell
& $py scripts/17_run_representation_ablation_seeds.py --protocols B --methods iq_only constellation_only iq_hoc fusion fusion_hoc --seeds 41 73 107 149 211 --device cuda
& $py scripts/17_run_representation_ablation_seeds.py --protocols A B --stages summary
```

`fusion` 使用第 3 步的 Proposed 预测；另外四个分支各训练五个种子。
HOC 从新的补偿 I/Q 生成。结果分别位于
`outputs/results/{exact_mixture,dcft_acquisition}/representation_ablation/protocol_{a,b}/seed_summary.csv`。
可直接在 PowerShell 中合并和导出表格，不创建 Python 入口文件：

```powershell
@'
from pathlib import Path
import pandas as pd
out=Path('outputs/results/dcft_acquisition/paper'); out.mkdir(parents=True,exist_ok=True)
parts=[]
for protocol,family in [('A','exact_mixture'),('B','dcft_acquisition')]:
    p=Path('outputs/results')/family/'representation_ablation'/f'protocol_{protocol.lower()}'/'seed_summary.csv'
    df=pd.read_csv(p)
    assert len(df)==5 and (df['count']==5).all(), 'Incomplete five-seed representation ablation'
    parts.append(df.assign(protocol=protocol,mean_percent=100*df['mean'],sample_sd_percent=100*df['std']))
table=pd.concat(parts,ignore_index=True)[['protocol','method','display','count','mean_percent','sample_sd_percent']]
table.to_csv(out/'table_iii.csv',index=False)
'@ | & $py -
```

**5. 表 II：四种前端的固定分类器比较。**

```powershell
& $py scripts/26_screen_frontend_attribution.py --stage estimates
& $py scripts/28_summarize_frontend_attribution.py --stage features --methods coherent_budget coherent_two_stage frft_acquisition
foreach ($method in @('coherent_budget','coherent_two_stage','dcft_acquisition','frft_acquisition')) {
    $inputFeature = if ($method -eq 'dcft_acquisition') { $iq } else { "$features/frontend_$method.h5" }
    & $py scripts/13_evaluate_model.py --checkpoint $checkpoint --raw-data $raw --feature-data $inputFeature --symbol-feature-data $inputFeature --splits $split --split test --model drc_dualnet --output "$attr/frozen_classifier/$method.csv" --summary-output "$attr/frozen_classifier/$method.summary.json" --batch-size 256 --num-workers 0 --device cuda --strict --load-into-memory
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $method" }
}
& $py scripts/26_screen_frontend_attribution.py --stage timing
& $py scripts/28_summarize_frontend_attribution.py --stage summary
Copy-Item "$attr/table_ii.csv" -Destination $paper
```

估计误差已重算，可以跳过第一条；其余步骤使用新的 seed-41 acquisition 分类器。
CPU timing 应在训练结束、没有竞争计算任务时执行。
保留的 coherent 方法是独立基线，原先两条 `+ refinement` 行已移除。

**6. 图 3：独立校准集上的残余率扫描和最差符号包络。**

```powershell
$gammaGrid = @(Import-Csv 'outputs/results/exact_mixture/gamma_amc_calibration_scan.csv' | ForEach-Object { [double]$_.gamma_res } | Sort-Object -Unique)
& $py scripts/32_validate_feature_degradation.py --classification-only --classification-raw-data $calRaw --classification-feature-data $calIQ --classification-splits $calSplit --classification-split test --final-iq-evm-checkpoint $checkpoint --classification-snr-db -7 -4 -1 2 5 --classification-samples-per-class-snr 100 --controlled-classification-gamma-grid @gammaGrid --classification-output "$cal/rate_scan.csv" --classification-detail-output "$cal/rate_detail.csv" --output-root $cal --figure "$figures/rate_accuracy.pdf" --device cuda
& $py scripts/34_bootstrap_gamma_amc_calibration.py --detail-csv "$cal/rate_detail.csv" --output-csv "$cal/rate_bootstrap.csv" --output-json "$cal/rate_bootstrap.json" --envelope-csv "$cal/rate_envelope.csv" --bootstrap-replicates 20000
& $py scripts/35_analyze_gamma_amc_operating_rule.py --validation-scan "$cal/rate_envelope.csv" --calibration-ci "$cal/rate_bootstrap.csv" --raw-data $raw --pilot-estimates $pilot --baseline-pilot-estimates data/features/leo_7mods_joint_practical_pilot_joint.h5 --estimator-label 'DCFT acquisition' --splits $split --output-csv "$cal/rate_operating.csv" --output-json "$cal/rate_operating.json" --figure "$figures/rate_operating.pdf"
```

原始 `rate_scan.csv` 汇总了正负注入；论文的最差符号曲线和阈值应使用
`rate_envelope.csv`，不能用正负平均准确率代替。
这里只复用原校准网格和独立校准输入，预测全部由新模型重新计算。

**7. 同步重算 V-B 中的联合 CFO/Rate 阈值。**

```powershell
$oldCalibration = Get-Content 'outputs/results/exact_mixture/joint_eta_calibration.json' -Raw | ConvertFrom-Json
$accuracyShards = @(); $detailShards = @(); $metadataShards = @()
foreach ($snr in @(-7,-4,-1,2,5)) {
    $etaGrid = @($oldCalibration.eta_grid_by_snr.PSObject.Properties["$snr"].Value)
    $allocationGrid = @($oldCalibration.allocation_grid)
    & $py scripts/36_calibrate_joint_eta.py --raw-data $calRaw --feature-data $calIQ --splits $calSplit --checkpoint $checkpoint --snr-db $snr --samples-per-class-snr 100 --eta-grid @etaGrid --allocation-grid @allocationGrid --output-csv "$cal/joint_snr$snr.csv" --detail-output "$cal/joint_snr${snr}_detail.csv" --metadata-output "$cal/joint_snr$snr.json" --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Joint calibration failed: SNR $snr" }
    $accuracyShards += "$cal/joint_snr$snr.csv"
    $detailShards += "$cal/joint_snr${snr}_detail.csv"
    $metadataShards += "$cal/joint_snr$snr.json"
}
& $py scripts/37_merge_joint_eta_shards.py --accuracy-shards @accuracyShards --detail-shards @detailShards --metadata-shards @metadataShards --output-csv "$cal/joint_scan.csv" --detail-output "$cal/joint_detail.csv" --metadata-output "$cal/joint_metadata.json"
foreach ($delta in @('0.02','0.01')) {
    & $py scripts/38_bootstrap_joint_eta_calibration.py --detail-csv "$cal/joint_detail.csv" --output-csv "$cal/joint_bootstrap_$delta.csv" --output-json "$cal/joint_bootstrap_$delta.json" --envelope-csv "$cal/joint_envelope_$delta.csv" --delta $delta --bootstrap-replicates 20000
    if ($LASTEXITCODE -ne 0) { throw "Bootstrap failed: delta $delta" }
}
& $py scripts/39_analyze_joint_eta_operating_rule.py --rate-operating-rule "$cal/rate_operating.csv" --eta-calibration "$cal/joint_bootstrap_0.02.csv" --output-csv "$cal/joint_operating.csv" --output-json "$cal/joint_operating.json"
& $py scripts/48_plot_integrated_operating_rule_preview.py --classification "$cal/rate_envelope.csv" --operating-rule "$cal/rate_operating.csv" --joint-operating-rule "$cal/joint_operating.csv" --output "$figures/gamma_operating_rule_integrated_ieee.pdf"
```

PDF 中的图 3 对应导出的 `gamma_operating_rule_integrated_ieee_a.pdf`。
`_b.pdf` 是附加的阈值覆盖诊断。若新的阈值落在扫描边界，需扩大独立校准网格再报告。

**8. 表 IV：重新测量 CPU 成本，再汇总五种子准确率。**

第 3 步图 6 命令已生成 `baseline_comparison/seed_summary.csv`；等训练完成后执行：

```powershell
& $py scripts/40_benchmark_paper_models.py --protocol-b --raw-data $raw --feature-data $iq --symbol-feature-data $symbol --summary outputs/results/baseline_comparison/seed_summary.csv --output outputs/results/baseline_comparison/protocol_b_accuracy_complexity.csv --device cpu --overwrite
& $py scripts/40_benchmark_paper_models.py --table-iv
Copy-Item outputs/results/baseline_comparison/table_iv.csv -Destination $paper
```

输出为 Protocol-B 五种子准确率及重新测量的单线程 CPU 成本。
E2E 是各阶段耗时的求和估计，不是连续整链路计时。

**9. 图 1 和表 I 的方法描述。**

```powershell
& $py scripts/42_plot_physics_guided_receiver_framework.py --output "$figures/fig1_receiver_framework.pdf"
```

表 I 的 Estimator B 改为 `DCFT-based chirp-focus acquisition`。
摘要、方法章节、V-B/V-C/V-E/V-G 和结论的数值应根据新 CSV 更新。

The default estimator is `dcft_acquisition`; `dfrft_only` is a legacy acquisition
alias. Defaults are coherent pilot weighting, timing phase 0, CFO bounds
[-550,550] Hz, rate step 160 Hz/s and FFT size 2048. Supply physical bounds for
other scenarios. `coherent_grid` explicitly selects the independent coherent
baseline. Retired hybrid options fail argument validation.

For Python use `DCFTPilotEstimator` and `DCFTConfig` from
`src.drc_hoc.dcft_estimator`. Generated HDF5/JSON metadata identify acquisition.
The descriptor retains all 48 features, exact-mixture scoring, two-fold pilot
noise estimation, 512 payload symbols and zero optional penalty weights.

No new top-level orchestration script is required. Run the stages above explicitly.
