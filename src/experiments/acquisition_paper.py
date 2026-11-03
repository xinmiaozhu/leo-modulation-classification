"""Acquisition-only Protocol-B preprocessing and frozen-classifier attribution."""
from __future__ import annotations
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import time
import h5py
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.drc_hoc.dcft_estimator import DCFTConfig, DCFTPilotEstimator, FrFTCaptureConfig, FrFTPilotEstimator
from src.drc_hoc.pilot_estimator import PilotMuEstimator, PilotMuEstimatorConfig
from src.drc_hoc.compensation import compensate_full_doppler
from src.utils.math_utils import complex_to_iq, to_complex

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/processed/leo_7mods_joint_practical.h5'
SPLITS = ROOT / 'data/splits/leo_7mods_joint_practical_splits.npz'
FEATURES = ROOT / 'data/features/dcft_acquisition'
RESULTS = ROOT / 'outputs/results/dcft_acquisition/frontend_comparison'
METHODS = ('coherent_budget', 'coherent_two_stage', 'dcft_acquisition', 'frft_acquisition')
LABELS = ('Coherent, validation-tuned', 'Coherent, coarse-to-fine', 'DCFT acquisition', 'FrFT acquisition')

def script(name):
    spec = importlib.util.spec_from_file_location('acquisition_' + Path(name).stem, ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')

def estimators():
    cfg = PilotMuEstimatorConfig(fd0_min_hz=-550, fd0_max_hz=550, fd0_step_hz=25,
                                fine_fd0_radius_hz=30, fine_fd0_step_hz=1, pilot_weighting='coherent')
    # Preserve the prior validation-selected grids; no new selection on test data.
    return dict(coherent_budget=PilotMuEstimator(replace(cfg, coarse_step_hz_per_s=120)),
                coherent_two_stage=PilotMuEstimator(cfg), dcft_acquisition=DCFTPilotEstimator(cfg),
                frft_acquisition=FrFTPilotEstimator(cfg, DCFTConfig(), FrFTCaptureConfig(33, 2048)))

def estimate_result(estimator, obs):
    result = estimator.estimate(*obs)
    return result.estimate if hasattr(result, 'estimate') else result

def estimate_test():
    RESULTS.mkdir(parents=True, exist_ok=True); threadpool_limits(1)
    indices = np.sort(np.load(SPLITS)['test'])
    with h5py.File(RAW) as raw:
        data = {k: raw[k][:] for k in ('iq', 'pilot_indices', 'pilot_symbols', 'mu', 'fd0', 'T')}
    write_json(RESULTS / 'protocol.json', dict(methods=METHODS, test_frames=len(indices),
        coherent_steps=[120,25], frft_grid=[33,2048], selection='Previously frozen validation-selected grids; no test retuning.',
        comparison='The coherent validation-tuned row is not an equal-time claim against acquisition.',
        frozen_classifier='New DCFT-acquisition-trained seed 41; frozen training normalization.'))
    rows=[]
    for name, estimator in estimators().items():
        dest=RESULTS / 'test' / (name+'.csv'); dest.parent.mkdir(exist_ok=True)
        done=set(pd.read_csv(dest)['index']) if dest.exists() else set()
        for pos,i in enumerate(indices):
            if i in done: continue
            got=estimate_result(estimator, [data[k][i] for k in ('iq','pilot_indices','pilot_symbols')])
            if not got.valid: raise ValueError(f'Invalid {name} frame {i}')
            em,ef=abs(got.mu_hat_hz_per_s-data['mu'][i]),abs(got.fd0_hat_hz-data['fd0'][i])
            row=dict(index=int(i),mu_hat_hz_per_s=got.mu_hat_hz_per_s,cfo_hat_hz=got.fd0_hat_hz,
                     abs_mu_error_hz_per_s=em,abs_cfo_error_hz=ef,B_phi=2*np.pi*ef*data['T'][i]+np.pi*em*data['T'][i]**2)
            pd.DataFrame([row]).to_csv(dest,mode='a',header=not dest.exists(),index=False)
            if (pos+1)%500==0: print(f'{name}: {pos+1}/{len(indices)}',flush=True)
        frame=pd.read_csv(dest).sort_values('index'); np.testing.assert_array_equal(frame['index'],indices)
        rows.append(dict(method=name,rate_p95_hz_per_s=frame.abs_mu_error_hz_per_s.quantile(.95),
                         cfo_p95_hz=frame.abs_cfo_error_hz.quantile(.95),B_phi_p95=frame.B_phi.quantile(.95)))
    pd.DataFrame(rows).to_csv(RESULTS/'estimation_summary.csv',index=False)

def timing():
    """Run separately from training; single thread, 77 validation frames, randomized order."""
    threadpool_limits(1); RESULTS.mkdir(parents=True,exist_ok=True)
    ix=np.load(SPLITS)['val']; rng=np.random.default_rng(20261018)
    with h5py.File(RAW) as raw:
        labels,snr=raw['label'][:],raw['snr_db'][:]
        chosen=[int(rng.choice(ix[(labels[ix]==label)&np.isclose(snr[ix],level)]))
                for label in np.unique(labels[ix]) for level in np.unique(snr[ix])]
        arrays={k:raw[k][:] for k in ('iq','pilot_indices','pilot_symbols')}
        obs=[[arrays[k][i] for k in arrays] for i in chosen]
    bank=estimators()
    for estimator in bank.values():
        for item in obs[:5]: estimate_result(estimator,item)
    rows=[]
    for repeat in range(3):
        for pos in rng.permutation(len(obs)):
            for name in rng.permutation(list(bank)):
                start=time.perf_counter_ns(); estimate_result(bank[name],obs[pos])
                rows.append(dict(method=name,index=chosen[pos],repeat=repeat,cpu_ms=(time.perf_counter_ns()-start)/1e6))
    frame=pd.DataFrame(rows); frame.to_csv(RESULTS/'timing_samples.csv',index=False)
    frame.groupby('method').cpu_ms.median().rename('median_cpu_ms').to_csv(RESULTS/'timing_summary.csv')

def build_features(method='dcft_acquisition'):
    """Complete main features or test-only counterfactuals with explicit validity."""
    threadpool_limits(1); module=script('10_precompute_symbol_constellation.py')
    pilot=FEATURES/'protocol_b_pilot.h5'
    dest=FEATURES/('protocol_b_iqcomp.h5' if method=='dcft_acquisition' else f'frontend_{method}.h5')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(RAW) as raw:
        data={k:raw[k][:] for k in ('iq','pilot_indices','pilot_symbols')}; n,_,width=data['iq'].shape
        meta={k:raw[k][:] for k in ('label','snr_db','mu','fd0','fs','T')}
    if method=='dcft_acquisition':
        with h5py.File(pilot) as p:
            assert p['pilot_valid'][:].all(); mu,fd=p['pilot_mu_hat'][:],p['pilot_fd0_hat'][:]
        indices=np.arange(n)
    else:
        frame=pd.read_csv(RESULTS/'test'/(method+'.csv')).sort_values('index')
        indices=frame['index'].to_numpy(int); mu=np.full(n,np.nan); fd=mu.copy()
        mu[indices]=frame.mu_hat_hz_per_s; fd[indices]=frame.cfo_hat_hz
    mods=('BPSK','QPSK','8PSK','16QAM','64QAM','16APSK','32APSK')
    constellations={m:module.get_constellation(m).astype(np.complex128) for m in mods}; taps=module.rrc_filter(.35,8,8)
    config=dict(raw_data=str(RAW.relative_to(ROOT)),mu_feature_data=str(pilot.relative_to(ROOT)),
                input_iq_feature_data=str(dest.relative_to(ROOT)),candidate_score_mode='exact_mixture',
                snr_source='pilot_evm',timing_phases=[0],rrc_beta=.35,rrc_span=8,samples_per_symbol=8,
                phase_grid_size=1,symbol_trim=8,max_symbols=512,center=True,power_normalize=True,
                complexity_penalty_lambda=0.,complexity_penalty_snr_threshold=-3.,complexity_penalty_snr_scale=4.,
                apsk_ring_penalty_lambda=0.,apsk_phase_weight=1.,apsk_occupancy_weight=.25,
                min_data_symbols=32,pilot_snr_offset_db=-10*np.log10(8),
                pilot_evm_estimator='two_fold_crossfit_gain_residual',modulations=mods)
    with h5py.File(dest,'a') as h:
        if 'done' not in h:
            h.create_dataset('iq_comp',(n,2,width),dtype='f4',chunks=(1,2,width),fillvalue=np.nan)
            h.create_dataset('evm_features',(n,48),dtype='f4',fillvalue=np.nan); h.create_dataset('h_drc',(n,21),dtype='f4')
            for key in ('gamma_hat','S_peak','V_hoc','mu_hat','pilot_fd0_hat'): h.create_dataset(key,(n,),dtype='f4')
            h.create_dataset('done',(n,),dtype='bool')
            for k,v in meta.items(): h.create_dataset(k,data=v)
            h.attrs['method']=method; h.attrs['raw_data']=str(RAW.relative_to(ROOT))
            h.attrs['mu_feature_data']=str(pilot.relative_to(ROOT)); h.attrs['config_json']=json.dumps(config)
        if h.attrs['method']!=method: raise ValueError('Feature lineage mismatch')
        h.attrs['config_json']=json.dumps(config)
        for pos,i in enumerate(indices):
            if h['done'][i]: continue
            comp=compensate_full_doppler(to_complex(data['iq'][i]),sample_rate_hz=200000,
                                       fd0_hat_hz=float(fd[i]),mu_hat_hz_per_s=float(mu[i]),centered=False)
            iq=complex_to_iq(comp,layout='channel_first').astype(np.float32)
            matched=np.convolve(to_complex(iq),taps,mode='same'); pi,ps=data['pilot_indices'][i],data['pilot_symbols'][i]
            gain,pe,_=module._estimate_gain_and_pilot_evm(matched,pi,ps,0,8)
            crossfit,_=module._estimate_crossfit_pilot_evm(matched,pi,ps,0,8); pe=crossfit if np.isfinite(crossfit) else pe
            symbols=module._extract_equalized_symbols(matched,gain,0,8,pi,8,512)
            symbols=module._prepare_symbols(symbols,True,True)
            evm,phases=module._score_evm_all_candidates(symbols,mods,constellations,np.array([0.]))
            values,names=module._evm_feature_vector(evm,mods,pe,symbols,constellations,phases,
                snr_db=module._pilot_sample_snr_db(pe,samples_per_symbol=8,offset_db=-10*np.log10(8)),
                complexity_penalty_lambda=0.,apsk_ring_penalty_lambda=0.,candidate_score_mode='exact_mixture')
            if len(values)!=48 or not np.isfinite(values).all(): raise ValueError(f'Invalid descriptor {i}')
            h['iq_comp'][i]=iq; h['evm_features'][i]=values; h['mu_hat'][i]=mu[i]; h['pilot_fd0_hat'][i]=fd[i]
            if 'evm_feature_names' not in h: h.create_dataset('evm_feature_names',data=np.asarray(names,dtype=h5py.string_dtype('utf-8')))
            h['done'][i]=True
            if (pos+1)%500==0: h.flush(); print(f'features/{method}: {pos+1}/{len(indices)}',flush=True)
        assert h['done'][indices].all()
    if method=='dcft_acquisition':
        with h5py.File(FEATURES/'protocol_b_symbol_exact.h5','w') as h:
            for key in ('evm_features','evm_feature_names','label','snr_db','done'): h[key]=h5py.ExternalLink(dest.name,key)
            h.attrs['config_json']=json.dumps(config)
    return dest

def stratified_paired_accuracy_interval(first, second, labels, snr, replicates=5000, seed=20261018):
    delta=100*(np.asarray(first,float)-np.asarray(second,float)); rng=np.random.default_rng(seed); draws=np.zeros(replicates)
    for label in np.unique(labels):
        for level in np.unique(snr):
            values=delta[(labels==label)&np.isclose(snr,level)]
            if len(values): draws+=rng.choice(values,(replicates,len(values)),replace=True).sum(axis=1)/len(delta)
    return dict(gain_pp=float(delta.mean()),ci95_low=float(np.quantile(draws,.025)),
                ci95_high=float(np.quantile(draws,.975)),bootstrap_replicates=replicates,strata='label x SNR')

def summary():
    rows=pd.read_csv(RESULTS/'estimation_summary.csv').set_index('method')
    rows=rows.join(pd.read_csv(RESULTS/'timing_summary.csv').set_index('method'))
    ix=np.sort(np.load(SPLITS)['test']); correct={}
    with h5py.File(RAW) as raw: labels,snr=raw['label'][ix],raw['snr_db'][ix]
    for method in METHODS:
        frame=pd.read_csv(RESULTS/'frozen_classifier'/(method+'.csv')).sort_values('index')
        np.testing.assert_array_equal(frame['index'],ix); np.testing.assert_array_equal(frame.label,labels)
        correct[method]=(frame.pred.to_numpy()==labels).astype(float); rows.loc[method,'accuracy_percent']=100*correct[method].mean()
    rows=rows.loc[list(METHODS)]; rows.insert(0,'front_end',LABELS); rows.to_csv(RESULTS/'table_ii.csv')
    contrasts=[dict(first='dcft_acquisition',second=m,**stratified_paired_accuracy_interval(correct['dcft_acquisition'],correct[m],labels,snr)) for m in METHODS if m!='dcft_acquisition']
    pd.DataFrame(contrasts).to_csv(RESULTS/'paired_bootstrap.csv',index=False)
