"""Validated manuscript artifacts for the four-length Protocol-B campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from src.experiments.frame_length import (ROOT, NAME, LENGTHS, SEEDS, METHODS, SNR, paths,
    completeness, validate_predictions, estimation_metrics, write_json)
from src.plotting.common import (IEEE_TRANS_PALETTE, set_ieee_trans_style,
    format_ieee_axis, set_panel_aspect_10_9, boxed_legend)


def plot_curves(aggregate, destination, reference=None):
    """Use the reference's rcParams, canvas, axes, inset and exact export sizes."""
    from pypdf import PdfReader
    from PIL import Image
    reference = reference or ROOT / 'outputs/figures/paper/paper_baseline_protocol_a_accuracy_vs_snr.pdf'
    page = PdfReader(reference).pages[0]
    pdf_size = (float(page.mediabox.width)/72, float(page.mediabox.height)/72)
    with Image.open(reference.with_suffix('.png')) as image:
        png_pixels = image.size
    set_ieee_trans_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.15))
    inset = ax.inset_axes([0.54, 0.50, 0.44, 0.35])
    for n, color, marker in zip(LENGTHS, ('black','blue','green','red'), ('o','s','^','D')):
        curve = aggregate[(aggregate.n == n) & (aggregate.method == 'proposed')].sort_values('snr_db')
        if len(curve) != len(SNR):
            raise ValueError(f'Missing formal curve for N={n}')
        for panel in (ax,inset):
            panel.plot(curve.snr_db,curve.mean_percent,label=fr'$N={n}$',color=IEEE_TRANS_PALETTE[color],
                marker=marker,markersize=3 if panel is ax else 2.2,markerfacecolor='white',
                markeredgewidth=.75,linewidth=1.45)
    ax.set(xlim=(SNR[0]-.5,SNR[-1]+.5),ylim=(0,101.5),xticks=SNR,yticks=np.arange(0,101,20),
           xlabel='SNR (dB)',ylabel='Classification accuracy (%)')
    format_ieee_axis(ax); set_panel_aspect_10_9(ax)
    inset.set(xlim=(-4,5),ylim=(70,100),xticks=[-4,-1,2,5],yticks=[70,80,90,100])
    inset.grid(True,linestyle=':',linewidth=.35,alpha=.5)
    inset.tick_params(direction='out',labelsize=7,length=2,width=.55,pad=1)
    for spine in inset.spines.values(): spine.set_linewidth(.65)
    ax.indicate_inset_zoom(inset,edgecolor='0.35',alpha=.75,linewidth=.55)
    boxed_legend(ax,loc='lower right',fontsize=8.5,borderaxespad=.18)
    fig.tight_layout(pad=.35)
    fig.canvas.draw()
    tight=fig.get_tightbbox(fig.canvas.get_renderer())
    center=((tight.x0+tight.x1)/2,(tight.y0+tight.y1)/2)
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    for suffix, size in [('pdf',pdf_size),('png',(png_pixels[0]/600,png_pixels[1]/600))]:
        bbox=Bbox.from_bounds(center[0]-size[0]/2,center[1]-size[1]/2,*size)
        fig.savefig(destination.with_suffix('.'+suffix),dpi=600,bbox_inches=bbox,pad_inches=0)
    plt.close(fig)
    result=PdfReader(destination.with_suffix('.pdf')).pages[0]
    assert abs(float(result.mediabox.width)-pdf_size[0]*72)<1e-5
    assert abs(float(result.mediabox.height)-pdf_size[1]*72)<1e-5
    with Image.open(destination.with_suffix('.png')) as image:
        if image.size!=png_pixels:
            raise ValueError(f'PNG dimensions differ from reference: {image.size} != {png_pixels}')
    return dict(reference=str(reference),pdf_inches=pdf_size,png_pixels=png_pixels,dpi=600,
        canvas_inches=[3.5,3.15],font_family='Times New Roman',label_font_pt=10,legend_font_pt=8.5,
        x_tick_font_pt=8.1,y_tick_font_pt=9,inset_tick_font_pt=7,axes_box_aspect=.9)


def publish(name=NAME):
    root=paths(8192,name)['root']
    cfg=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    if cfg.get('smoke') or cfg['samples_per_class_snr']!={'train':200,'val':100,'test':100}:
        raise ValueError('Only the full predeclared experiment may publish manuscript artifacts.')
    state=completeness(name)
    if state['complete']!=state['expected']:
        raise ValueError(f"Incomplete campaign: {state['complete']}/{state['expected']}; formal artifacts are withheld.")
    rows=[]; curves=[]; frozen=[]; estimator=[]
    for n in LENGTHS:
        p=paths(n,name)
        estimation=estimation_metrics(n,name)
        timing=json.loads((p['result']/'cpu_receiver_timing.json').read_text())
        estimator.append(dict(estimation,**timing))
        for method in METHODS:
            for seed in SEEDS:
                output=p['result']/method/f'seed_{seed}_test.csv'
                df=validate_predictions(output,p)
                directory=p['checkpoint']/method/f'seed_{seed}'
                run_cfg=json.loads((directory/'run_config.json').read_text())
                if run_cfg['seed']!=seed or Path(run_cfg['raw_data']).resolve()!=p['raw'].resolve():
                    raise ValueError(f'Checkpoint provenance mismatch: {directory}')
                history=pd.read_csv(directory/'history.csv')
                training=json.loads((directory/'train_result.json').read_text())
                if not np.isclose(training['best_acc'],history.val_acc.max()):
                    raise ValueError('Checkpoint selection inconsistent with validation history')
                rows.append(dict(n=n,method=method,seed=seed,accuracy_percent=100*df.correct.mean(),
                    test_count=len(df),epochs=int(history.epoch.max()),eval_csv=str(output.relative_to(ROOT))))
                curve=df.groupby('snr_db').correct.agg(accuracy='mean',sample_count='size').reset_index()
                curve['n']=n; curve['method']=method; curve['seed']=seed
                curve['accuracy_percent']=100*curve.accuracy
                curves.append(curve)
                if method=='proposed':
                    for mode in ('no_comp','oracle'):
                        diag=pd.read_csv(p['result']/method/f'seed_{seed}_{mode}.csv').sort_values('index')
                        for key in ('index','label','snr_db'):
                            np.testing.assert_array_equal(diag[key],df[key])
                        np.testing.assert_array_equal(diag.correct,(diag.label==diag.pred).astype(int))
                        frozen.append(dict(n=n,seed=seed,mode=mode,accuracy_percent=100*diag.correct.mean(),
                            estimated_minus_mode_pp=100*(df.correct.to_numpy()-diag.correct.to_numpy()).mean()))
    runs=pd.DataFrame(rows); per_snr=pd.concat(curves,ignore_index=True)
    aggregate=per_snr.groupby(['n','method','snr_db']).accuracy_percent.agg(
        seed_count='count',mean_percent='mean',sample_sd_percent='std').reset_index()
    assert (aggregate.seed_count==5).all()
    summary=runs.groupby(['n','method']).accuracy_percent.agg(seed_count='count',mean_percent='mean',sample_sd_percent='std').reset_index()
    table=pd.DataFrame(estimator)
    for method in METHODS:
        selected=summary[summary.method==method].set_index('n')
        table[method+'_mean_percent']=table.n.map(selected.mean_percent)
        table[method+'_sd_percent']=table.n.map(selected.sample_sd_percent)
    source=root/'paper'; source.mkdir(parents=True,exist_ok=True)
    runs.to_csv(root/'seed_runs.csv',index=False)
    summary.to_csv(root/'seed_summary.csv',index=False)
    per_snr.to_csv(root/'snr_seed_runs.csv',index=False)
    aggregate.to_csv(root/'snr_summary.csv',index=False)
    pd.DataFrame(frozen).to_csv(root/'frozen_compensation_diagnostics.csv',index=False)
    table.to_csv(source/'frame_length_table.csv',index=False)
    # Keep the complete CPU measurements in the source CSV; the paper uses the median.
    latex=[r'\begin{table*}[t]',r'\centering',
        r'\caption{Observation-length comparison under Protocol B. Accuracy is the mean $\pm$ sample standard deviation across five matched training seeds. Estimation P95 values use the common test set; E2E time is the batch-one, single-thread CPU median for the complete proposed receiver on validation frames, excluding observation time.}',
        r'\label{tab:frame_length}',r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{rrrrccccrrr}',r'\hline',
        r'$N$ & $T_f$ & $N_p$ & Rate P95 & CFO P95 & $B_\phi$ P95 & I/Q-only & IQFormer & Proposed & E2E \\',
        r' & (ms) & & (Hz/s) & (Hz) & (rad) & (\%) & (\%) & (\%) & (ms) \\',r'\hline']
    # Ten displayed columns (three accuracy columns are math expressions).
    latex[5]=r'\begin{tabular}{rrrrrrcccc}'
    for r in table.itertuples():
        acc=lambda m: f'${getattr(r,m+"_mean_percent"):.2f} \\pm {getattr(r,m+"_sd_percent"):.2f}$'
        latex.append(f'{r.n} & {r.duration_ms:.2f} & {r.pilots} & {r.rate_p95_hz_s:.2f} & {r.cfo_p95_hz:.2f} & {r.b_phi_p95_rad:.3f} & '+
            ' & '.join(acc(m) for m in ('iq_only','iqformer','proposed'))+f' & {r.e2e_median_ms:.2f} '+r'\\')
    latex += [r'\hline',r'\end{tabular}',r'\end{table*}']
    (source/'frame_length_table.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    figure=source/'frame_length_protocol_b_accuracy_vs_snr.pdf'
    style=plot_curves(aggregate,figure)
    caption=('Protocol-B test accuracy versus SNR for the proposed receiver at four observation lengths. '
             'Each line averages five independently trained, matched-seed models; all lengths share paired source frames. '
             'Sampling rate and pilot fraction are fixed at 200 kHz and 1/16, respectively.')
    figure_tex='\n'.join([r'\begin{figure}[t]',r'\centering',
        r'\includegraphics[width=\columnwidth]{frame_length_protocol_b_accuracy_vs_snr.pdf}',
        '\\caption{'+caption+'}',r'\label{fig:frame_length}',r'\end{figure}'])
    (source/'frame_length_figure.tex').write_text(figure_tex+'\n',encoding='utf-8')
    means=table.proposed_mean_percent.to_numpy()
    paragraph=(f'We further evaluate observation lengths N = 2048, 4096, 8192, and 16384 under Protocol B. '
        f'The corresponding frame durations are 10.24, 20.48, 40.96, and 81.92 ms. '
        f'The proposed receiver achieves mean accuracies of '+', '.join(f'{x:.2f}%' for x in means)+', respectively. '
        'Each length is trained independently with the same number of source frames and five matched seeds. '
        'Longer observations increase both pilot and payload evidence, while quadratic-phase severity grows with the squared duration. '
        'We therefore report residual phase-error bounds and complete receiver processing time alongside accuracy. '
        'Descriptor extraction uses at most 512 available payload symbols, without repeating symbols in shorter frames. '
        'These results characterize the tested duration range and do not establish a universally optimal observation length.')
    (source/'manuscript_addition.md').write_text(paragraph+'\n\nFigure caption: '+caption+'\n',encoding='utf-8')
    (source/'manuscript_addition.tex').write_text(paragraph.replace('%',r'\%')+'\n',encoding='utf-8')
    write_json(source/'artifact_manifest.json',dict(formal=True,complete_cells=60,seeds=list(SEEDS),
        protocol=cfg,figure_style=style,uncertainty='Sample SD across training seeds, not test-frame SE',
        timing='Measured complete CPU receiver on validation frames; includes descriptor; excludes disk I/O and observation wait.',
        artifacts=['frame_length_table.csv','frame_length_table.tex','frame_length_figure.tex',figure.name,
                   figure.with_suffix('.png').name,'manuscript_addition.md','manuscript_addition.tex']))
    print(f'Published validated manuscript artifacts: {source}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default=NAME)
    args=parser.parse_args()
    publish(args.name)
