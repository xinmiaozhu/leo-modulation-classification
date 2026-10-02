"""Synthetic LEO signal dataset generation.

This module builds HDF5-ready arrays:
    iq, label, snr_db, fd0, mu, gamma, modulation, domain_id, ...
It is intentionally independent of PyTorch so that generation can run on a
machine without GPU.

Typical usage:
    cfg = load_config("configs/dataset/leo_sband.yaml")
    arrays, summary = build_dataset_from_config(cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from src.physics.gamma_metric import compute_gamma
from src.signal.signal_generator import LEOSignalGenerator, SignalSpec
from src.utils.io import save_h5
from src.utils.logger import get_logger


@dataclass
class DatasetBuildConfig:
    """Flat dataset-generation configuration.

    This dataclass is mainly used internally. YAML configs can be nested; use
    from_mapping() to convert nested dicts into this flat config.
    """

    seed: int = 42
    modulations: tuple[str, ...] = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM")
    num_samples_per_class: int = 1000

    num_symbols: int = 256
    samples_per_symbol: int = 8
    sample_rate_hz: float = 1.0e6
    carrier_frequency_hz: float = 2.0e9
    target_num_samples: int | None = None

    snr_db_min: float = -10.0
    snr_db_max: float = 20.0
    fd0_min_hz: float = -2.0e5
    fd0_max_hz: float = 2.0e5
    mu_min_hz_per_s: float = -5.0e3
    mu_max_hz_per_s: float = 5.0e3

    channel_type: str = "awgn"
    compensate_constant_doppler: bool = False

    # Optional domain metadata used for cross-domain experiments.
    domain_values: tuple[int, ...] = (0,)
    orbit_altitudes_km: tuple[float, ...] = (550.0,)
    orbit_inclinations_deg: tuple[float, ...] = (53.0,)

    rrc_beta: float = 0.35
    rrc_span: int = 8

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "DatasetBuildConfig":
        """Create config from nested YAML-like mapping."""

        def get_nested(mapping: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
            cur: Any = mapping
            for k in keys:
                if not isinstance(cur, Mapping) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        signal = cfg.get("signal", {})
        leo = cfg.get("leo", {})
        channel = cfg.get("channel", {})
        domain = cfg.get("domain", {})

        snr_range = channel.get("snr_db_range", [-10.0, 20.0])
        fd0_range = leo.get("fd0_range_hz", [-2.0e5, 2.0e5])
        mu_range = leo.get("mu_range_hz_per_s", [-5.0e3, 5.0e3])

        return cls(
            seed=int(cfg.get("seed", 42)),
            modulations=tuple(signal.get("modulations", cls.modulations)),
            num_samples_per_class=int(signal.get("num_samples_per_class", 1000)),
            num_symbols=int(signal.get("num_symbols", 256)),
            samples_per_symbol=int(signal.get("sps", signal.get("samples_per_symbol", 8))),
            sample_rate_hz=float(signal.get("fs", signal.get("sample_rate_hz", 1.0e6))),
            carrier_frequency_hz=float(
                leo.get(
                    "carrier_frequency",
                    leo.get(
                        "carrier_frequency_hz",
                        signal.get("carrier_frequency_hz", 2.0e9),
                    ),
                )
            ),
            target_num_samples=signal.get("target_num_samples", None),
            snr_db_min=float(snr_range[0]),
            snr_db_max=float(snr_range[1]),
            fd0_min_hz=float(fd0_range[0]),
            fd0_max_hz=float(fd0_range[1]),
            mu_min_hz_per_s=float(mu_range[0]),
            mu_max_hz_per_s=float(mu_range[1]),
            channel_type=str(channel.get("type", channel.get("channel_type", "awgn"))),
            compensate_constant_doppler=bool(
                channel.get("compensate_constant_doppler", False)
            ),
            domain_values=tuple(domain.get("domain_values", [0])),
            orbit_altitudes_km=tuple(leo.get("orbit_altitudes_km", [550.0])),
            orbit_inclinations_deg=tuple(leo.get("orbit_inclinations_deg", [53.0])),
            rrc_beta=float(signal.get("rrc_beta", 0.35)),
            rrc_span=int(signal.get("rrc_span", 8)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sample_domain_metadata(
    rng: np.random.Generator,
    build_cfg: DatasetBuildConfig,
) -> tuple[int, float, float]:
    """Sample domain ID and orbit metadata.

    The current generator uses fd0/mu ranges directly. Orbit metadata is still
    stored because it is useful for cross-domain experiment bookkeeping.
    """

    domain_id = int(rng.choice(build_cfg.domain_values))
    altitude = float(rng.choice(build_cfg.orbit_altitudes_km))
    inclination = float(rng.choice(build_cfg.orbit_inclinations_deg))
    return domain_id, altitude, inclination


def _make_random_spec(
    rng: np.random.Generator,
    build_cfg: DatasetBuildConfig,
    modulation: str,
) -> SignalSpec:
    snr_db = float(rng.uniform(build_cfg.snr_db_min, build_cfg.snr_db_max))
    fd0_hz = float(rng.uniform(build_cfg.fd0_min_hz, build_cfg.fd0_max_hz))
    mu_hz_per_s = float(rng.uniform(build_cfg.mu_min_hz_per_s, build_cfg.mu_max_hz_per_s))

    return SignalSpec(
        modulation=modulation,
        num_symbols=build_cfg.num_symbols,
        samples_per_symbol=build_cfg.samples_per_symbol,
        sample_rate_hz=build_cfg.sample_rate_hz,
        carrier_frequency_hz=build_cfg.carrier_frequency_hz,
        snr_db=snr_db,
        fd0_hz=fd0_hz,
        mu_hz_per_s=mu_hz_per_s,
        rrc_beta=build_cfg.rrc_beta,
        rrc_span=build_cfg.rrc_span,
        channel_type=build_cfg.channel_type,
        compensate_constant_doppler=build_cfg.compensate_constant_doppler,
        target_num_samples=build_cfg.target_num_samples,
    )


def build_dataset_arrays(
    build_cfg: DatasetBuildConfig,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Generate a synthetic LEO signal dataset into arrays."""

    rng = np.random.default_rng(build_cfg.seed)
    generator = LEOSignalGenerator(modulations=list(build_cfg.modulations), rng=rng)

    total = build_cfg.num_samples_per_class * len(build_cfg.modulations)
    iq_list = []
    label_list = []
    snr_list = []
    fd0_list = []
    mu_list = []
    gamma_list = []
    modulation_list = []
    domain_list = []
    altitude_list = []
    inclination_list = []
    fs_list = []
    fc_list = []
    T_list = []

    iterator = []
    for modulation in build_cfg.modulations:
        for _ in range(build_cfg.num_samples_per_class):
            iterator.append(modulation)

    if show_progress:
        iterator = tqdm(iterator, desc="Generating LEO signal dataset")

    for modulation in iterator:
        spec = _make_random_spec(rng, build_cfg, modulation)
        sample = generator.generate(spec)

        domain_id, altitude, inclination = _sample_domain_metadata(rng, build_cfg)

        iq_list.append(sample.iq.astype(np.float32))
        label_list.append(sample.label)
        snr_list.append(sample.info["snr_db"])
        fd0_list.append(sample.info["fd0_hz"])
        mu_list.append(sample.info["mu_hz_per_s"])
        gamma_list.append(sample.info["gamma"])
        modulation_list.append(sample.modulation.encode("utf-8"))
        domain_list.append(domain_id)
        altitude_list.append(altitude)
        inclination_list.append(inclination)
        fs_list.append(sample.info["sample_rate_hz"])
        fc_list.append(sample.info["carrier_frequency_hz"])
        T_list.append(sample.info["observation_time_s"])

    arrays = {
        "iq": np.stack(iq_list, axis=0).astype(np.float32),
        "label": np.asarray(label_list, dtype=np.int64),
        "snr_db": np.asarray(snr_list, dtype=np.float32),
        "fd0": np.asarray(fd0_list, dtype=np.float32),
        "mu": np.asarray(mu_list, dtype=np.float32),
        "gamma": np.asarray(gamma_list, dtype=np.float32),
        "modulation": np.asarray(modulation_list, dtype="S16"),
        "domain_id": np.asarray(domain_list, dtype=np.int64),
        "orbit_altitude_km": np.asarray(altitude_list, dtype=np.float32),
        "orbit_inclination_deg": np.asarray(inclination_list, dtype=np.float32),
        "fs": np.asarray(fs_list, dtype=np.float32),
        "fc": np.asarray(fc_list, dtype=np.float32),
        "T": np.asarray(T_list, dtype=np.float32),
        "label_names": np.asarray([m.encode("utf-8") for m in build_cfg.modulations], dtype="S16"),
    }

    return arrays


def summarize_dataset(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Create a compact dataset summary."""

    labels = np.asarray(arrays["label"])
    gammas = np.asarray(arrays["gamma"])
    snr = np.asarray(arrays["snr_db"])
    mu = np.asarray(arrays["mu"])

    unique, counts = np.unique(labels, return_counts=True)

    summary = {
        "num_samples": int(len(labels)),
        "num_classes": int(len(unique)),
        "class_counts": {int(k): int(v) for k, v in zip(unique, counts)},
        "iq_shape": list(arrays["iq"].shape),
        "snr_db_min": float(np.min(snr)),
        "snr_db_max": float(np.max(snr)),
        "mu_min_hz_per_s": float(np.min(mu)),
        "mu_max_hz_per_s": float(np.max(mu)),
        "gamma_min": float(np.min(gammas)),
        "gamma_max": float(np.max(gammas)),
        "gamma_mean": float(np.mean(gammas)),
    }

    if "domain_id" in arrays:
        d, c = np.unique(arrays["domain_id"], return_counts=True)
        summary["domain_counts"] = {int(k): int(v) for k, v in zip(d, c)}

    return summary


def build_dataset_from_config(
    cfg: Mapping[str, Any],
    output_h5: str | None = None,
    show_progress: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build dataset from nested config mapping.

    Args:
        cfg: YAML-like mapping.
        output_h5: Optional HDF5 output path.
    """

    build_cfg = DatasetBuildConfig.from_mapping(cfg)
    logger = get_logger()

    logger.info("Building dataset with configuration:")
    logger.info(build_cfg.to_dict())

    arrays = build_dataset_arrays(build_cfg, show_progress=show_progress)
    summary = summarize_dataset(arrays)
    summary["build_config"] = build_cfg.to_dict()

    if output_h5 is not None:
        payload = dict(arrays)
        payload["meta"] = {
            "summary_json": str(summary),
        }
        save_h5(output_h5, payload)
        logger.info(f"Saved dataset to {output_h5}")

    return arrays, summary
