import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from src.drc_hoc.hybrid_dfrft_estimator import HybridDFRFTConfig, HybridDFRFTPilotEstimator
from src.drc_hoc.pilot_estimator import PilotMuEstimatorConfig
from src.experiments import frame_length as campaign
from src.experiments.frame_length_report import publish, plot_curves


def test_long_frame_fft_alias_is_lifted_into_physical_cfo_interval():
    t=np.arange(128)*128/200000
    mu,fd0=-8000.,-500.
    assert fd0+mu*t.mean() < -1/(2*(t[1]-t[0]))
    z=np.exp(1j*(2*np.pi*fd0*t+np.pi*mu*t*t))
    config=PilotMuEstimatorConfig(fd0_min_hz=-550,fd0_max_hz=550)
    estimator=HybridDFRFTPilotEstimator(config,HybridDFRFTConfig(
        coarse_mu_step_hz_per_s=40,fft_size=4096,unwrap_centered_cfo=True))
    got=estimator._chirp_focus(z,t)
    assert abs(got[0]-mu)<=40
    assert abs(got[2]-fd0)<2
    # Reject an intrinsically ambiguous prior instead of selecting a hidden alias.
    wide=HybridDFRFTPilotEstimator(PilotMuEstimatorConfig(fd0_min_hz=-1000,fd0_max_hz=1000),
                                 HybridDFRFTConfig(unwrap_centered_cfo=True))
    with pytest.raises(ValueError,match='width'):
        wide._chirp_focus(z,t)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA parity test')
def test_cuda_long_frame_matches_cpu():
    from src.signal.signal_generator import LEOSignalGenerator,SignalSpec
    from src.signal.pulse_shape import rrc_filter
    estimator_script=campaign.module('06_precompute_pilot_mu.py')
    gen=LEOSignalGenerator(rng=np.random.default_rng(72))
    sample=gen.generate(SignalSpec(modulation='QPSK',num_symbols=2048,samples_per_symbol=8,
        sample_rate_hz=200000,target_num_samples=16384,snr_db=50,fd0_hz=-500,mu_hz_per_s=-8000,
        pilot_enabled=True,pilot_pattern='comb',pilot_interval_symbols=16,pilot_guard_symbols=0))
    cfg=PilotMuEstimatorConfig(sample_rate_hz=200000,samples_per_symbol=8,fd0_min_hz=-550,fd0_max_hz=550,
                              pilot_weighting='coherent',timing_phases=(0,))
    hybrid=HybridDFRFTConfig(coarse_mu_step_hz_per_s=40,fft_size=4096,
        fine_mu_radius_hz_per_s=50,fine_mu_step_hz_per_s=1.25,fine_fd0_step_hz=.5,unwrap_centered_cfo=True)
    cpu=HybridDFRFTPilotEstimator(cfg,hybrid).estimate(sample.iq,sample.pilot_indices,sample.pilot_symbols).estimate
    gpu=estimator_script._estimate_hybrid_cuda_batch(sample.iq[None],sample.pilot_indices[None],sample.pilot_symbols[None],
        config=cfg,hybrid_config=hybrid,device=torch.device('cuda'),rrc_taps=rrc_filter(.35,8,8))
    assert abs(cpu.mu_hat_hz_per_s-gpu[0][0])<=1.26
    assert abs(cpu.fd0_hat_hz-gpu[1][0])<=.51
    assert abs(cpu.mu_hat_hz_per_s+8000)<30
    assert abs(cpu.fd0_hat_hz+500)<2


def test_paired_generation_and_split_integrity(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    monkeypatch.setattr(campaign,'SNR',(-1,5))
    campaign.prepare_source('unit',smoke=True)
    p=campaign.paths(8192,'unit')
    splits=dict(np.load(p['splits']))
    assert set(splits['train']).isdisjoint(splits['test'])
    with h5py.File(campaign.paths(2048,'unit')['raw'],'r') as short,h5py.File(campaign.paths(16384,'unit')['raw'],'r') as long:
        np.testing.assert_array_equal(short['iq'][:],long['iq'][:,:,:2048])
        np.testing.assert_allclose(long['gamma'][:],short['gamma'][:]*64,rtol=1e-6)
        assert short['pilot_symbols'].shape[1]==16 and long['pilot_symbols'].shape[1]==128
    # Re-entry reuses the exact completed parent source.
    parent=p['data']/'source_16384.h5'; timestamp=parent.stat().st_mtime_ns
    campaign.prepare_source('unit',smoke=True)
    assert parent.stat().st_mtime_ns==timestamp


def test_paper_report_rejects_smoke_and_incomplete_campaign(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    root=campaign.paths(8192,'unit')['root']
    campaign.write_json(root/'protocol.json',campaign.protocol(smoke=True))
    with pytest.raises(ValueError,match='full predeclared'):
        publish('unit')
    campaign.write_json(root/'protocol.json',campaign.protocol(smoke=False))
    with pytest.raises(ValueError,match='Incomplete campaign'):
        publish('unit')
    assert not (root/'paper').exists()


def test_plot_matches_reference_export_dimensions(tmp_path):
    # Synthetic values are used only in an isolated rendering test, never published.
    reference=Path(__file__).resolve().parents[1]/'outputs/figures/paper/paper_baseline_protocol_a_accuracy_vs_snr.pdf'
    if not reference.exists(): pytest.skip('Local reference artifact is unavailable')
    rows=[dict(n=n,method='proposed',snr_db=s,mean_percent=50+40/(1+np.exp(-s/3)))
          for n in campaign.LENGTHS for s in campaign.SNR]
    plot_curves(pd.DataFrame(rows),tmp_path/'test.pdf',reference=reference)


def test_retired_scan_removed_and_calibration_helpers_remain():
    scan=campaign.module('32_validate_feature_degradation.py')
    assert not hasattr(scan,'_controlled_residual_scan')
    for name in ('_controlled_classification_scan','_load_fixed_classifier','_build_evm_features_for_frame',
                 '_resolve_symbol_config','_standardize_evm','_predict_fixed_classifier'):
        assert callable(getattr(scan,name))
    for filename,function in [('33_check_rate_sign_and_cfo_sensitivity.py','load_scan_module'),
                              ('36_calibrate_joint_eta.py','_load_degradation_helpers')]:
        imported=getattr(campaign.module(filename),function)()
        assert callable(imported._build_evm_features_for_frame)
