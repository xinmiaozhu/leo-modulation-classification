"""Generate trajectory-structured LEO signal datasets.

Unlike ``build_dataset.py``, this generator produces consecutive frames from
the same synthetic pass and stores ``track_id`` plus frame timing metadata.
It supports pass-level dataset splits and trajectory generalization experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from src.signal.signal_generator import LEOSignalGenerator, SignalSpec
from src.physics.leo_orbit import circular_pass_profile
from src.utils.io import save_h5
from src.utils.logger import get_logger


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


@dataclass
class TrackDatasetBuildConfig:
    seed: int = 42
    modulations: tuple[str, ...] = ("QPSK", "16QAM", "64QAM", "16APSK", "32APSK")
    tracks_per_modulation: int = 20
    frames_per_track: int = 50
    frame_spacing_s: float = 1.0

    num_symbols: int = 1024
    samples_per_symbol: int = 8
    sample_rate_hz: float = 200_000.0
    carrier_frequency_hz: float = 30.0e9
    target_num_samples: int | None = 8192

    snr_db_min: float = -10.0
    snr_db_max: float = 20.0
    snr_track_std_db: float = 2.0
    snr_profile_mode: str = "random"

    fd0_min_hz: float = 0.0
    fd0_max_hz: float = 0.0
    mu_min_hz_per_s: float = -8160.0
    mu_max_hz_per_s: float = -180.0

    channel_type: str = "awgn"
    compensate_constant_doppler: bool = False
    rrc_beta: float = 0.35
    rrc_span: int = 8
    pilot_enabled: bool = False
    pilot_pattern: str = "none"
    pilot_interval_symbols: int = 16
    pilot_guard_symbols: int = 8
    pilot_num_symbols: int | None = None
    pilot_min_spacing_symbols: int = 1
    pilot_block_length_symbols: int = 64
    pilot_block_positions: tuple[str, ...] = ("start", "middle", "end")
    pilot_modulation: str = "QPSK"
    pilot_symbol_mode: str = "pn"
    pilot_symbol_index: int = 0
    pilot_seed: int = 2026
    timing_offset_min_samples: int = 0
    timing_offset_max_samples: int = 0

    # Smooth negative Doppler-rate profile controls. ``mu_center`` is most
    # negative near closest approach; ``mu_edge`` is less negative near segment
    # edges. Defaults are derived from the global mu range if left as None.
    mu_center_min_hz_per_s: float | None = None
    mu_center_max_hz_per_s: float | None = None
    mu_edge_min_hz_per_s: float | None = None
    mu_edge_max_hz_per_s: float | None = None
    profile_center_min: float = 0.35
    profile_center_max: float = 0.65
    profile_width_min: float = 0.45
    profile_width_max: float = 0.80
    mu_frame_jitter_std_hz_per_s: float = 0.0
    profile_mode: str = "synthetic"
    maximum_elevations_deg: tuple[float, ...] = (20.0, 30.0, 45.0, 60.0, 75.0, 90.0)
    minimum_elevation_deg: float = 10.0
    include_earth_rotation: bool = True

    domain_values: tuple[int, ...] = (0,)
    orbit_altitudes_km: tuple[float, ...] = (600.0,)
    orbit_inclinations_deg: tuple[float, ...] = (53.0, 70.0, 90.0)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "TrackDatasetBuildConfig":
        signal = cfg.get("signal", {})
        leo = cfg.get("leo", {})
        channel = cfg.get("channel", {})
        trajectory = cfg.get("trajectory", {})
        domain = cfg.get("domain", {})
        pilot = cfg.get("pilot", {})
        synchronization = cfg.get("synchronization", {})

        snr_range = channel.get("snr_db_range", [-10.0, 20.0])
        fd0_range = leo.get("fd0_range_hz", [0.0, 0.0])
        mu_range = leo.get("mu_range_hz_per_s", [-8160.0, -180.0])
        timing_offset_range = synchronization.get("integer_timing_offset_samples", [0, 0])

        return cls(
            seed=int(cfg.get("seed", 42)),
            modulations=_as_tuple(signal.get("modulations", cls.modulations)),
            tracks_per_modulation=int(trajectory.get("tracks_per_modulation", 20)),
            frames_per_track=int(trajectory.get("frames_per_track", 50)),
            frame_spacing_s=float(trajectory.get("frame_spacing_s", 1.0)),
            num_symbols=int(signal.get("num_symbols", 1024)),
            samples_per_symbol=int(signal.get("sps", signal.get("samples_per_symbol", 8))),
            sample_rate_hz=float(signal.get("fs", signal.get("sample_rate_hz", 200_000.0))),
            carrier_frequency_hz=float(leo.get("carrier_frequency_hz", signal.get("carrier_frequency_hz", 30.0e9))),
            target_num_samples=signal.get("target_num_samples", 8192),
            snr_db_min=float(snr_range[0]),
            snr_db_max=float(snr_range[1]),
            snr_track_std_db=float(trajectory.get("snr_track_std_db", 2.0)),
            snr_profile_mode=str(trajectory.get("snr_profile_mode", "random")),
            fd0_min_hz=float(fd0_range[0]),
            fd0_max_hz=float(fd0_range[1]),
            mu_min_hz_per_s=float(mu_range[0]),
            mu_max_hz_per_s=float(mu_range[1]),
            channel_type=str(channel.get("type", channel.get("channel_type", "awgn"))),
            compensate_constant_doppler=bool(channel.get("compensate_constant_doppler", False)),
            rrc_beta=float(signal.get("rrc_beta", 0.35)),
            rrc_span=int(signal.get("rrc_span", 8)),
            pilot_enabled=bool(pilot.get("enabled", False)),
            pilot_pattern=str(pilot.get("pattern", "none")),
            pilot_interval_symbols=int(pilot.get("interval_symbols", 16)),
            pilot_guard_symbols=int(pilot.get("guard_symbols", 8)),
            pilot_num_symbols=(
                None if pilot.get("count_symbols", None) is None else int(pilot.get("count_symbols"))
            ),
            pilot_min_spacing_symbols=int(pilot.get("min_spacing_symbols", 1)),
            pilot_block_length_symbols=int(pilot.get("block_length_symbols", 64)),
            pilot_block_positions=tuple(pilot.get("block_positions", ["start", "middle", "end"])),
            pilot_modulation=str(pilot.get("modulation", "QPSK")),
            pilot_symbol_mode=str(pilot.get("symbol_mode", "pn")),
            pilot_symbol_index=int(pilot.get("symbol_index", 0)),
            pilot_seed=int(pilot.get("seed", 2026)),
            timing_offset_min_samples=int(timing_offset_range[0]),
            timing_offset_max_samples=int(timing_offset_range[1]),
            mu_center_min_hz_per_s=trajectory.get("mu_center_min_hz_per_s", None),
            mu_center_max_hz_per_s=trajectory.get("mu_center_max_hz_per_s", None),
            mu_edge_min_hz_per_s=trajectory.get("mu_edge_min_hz_per_s", None),
            mu_edge_max_hz_per_s=trajectory.get("mu_edge_max_hz_per_s", None),
            profile_center_min=float(trajectory.get("profile_center_min", 0.35)),
            profile_center_max=float(trajectory.get("profile_center_max", 0.65)),
            profile_width_min=float(trajectory.get("profile_width_min", 0.45)),
            profile_width_max=float(trajectory.get("profile_width_max", 0.80)),
            mu_frame_jitter_std_hz_per_s=float(trajectory.get("mu_frame_jitter_std_hz_per_s", 0.0)),
            profile_mode=str(trajectory.get("profile_mode", "synthetic")),
            maximum_elevations_deg=_as_tuple(
                trajectory.get("maximum_elevations_deg", [20.0, 30.0, 45.0, 60.0, 75.0, 90.0])
            ),
            minimum_elevation_deg=float(trajectory.get("minimum_elevation_deg", 10.0)),
            include_earth_rotation=bool(trajectory.get("include_earth_rotation", True)),
            domain_values=_as_tuple(domain.get("domain_values", [0])),
            orbit_altitudes_km=_as_tuple(leo.get("orbit_altitudes_km", [600.0])),
            orbit_inclinations_deg=_as_tuple(leo.get("orbit_inclinations_deg", [53.0, 70.0, 90.0])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_profile_ranges(cfg: TrackDatasetBuildConfig) -> tuple[float, float, float, float]:
    mu_min = min(float(cfg.mu_min_hz_per_s), float(cfg.mu_max_hz_per_s))
    mu_max = max(float(cfg.mu_min_hz_per_s), float(cfg.mu_max_hz_per_s))
    span = mu_max - mu_min

    center_min = cfg.mu_center_min_hz_per_s
    center_max = cfg.mu_center_max_hz_per_s
    edge_min = cfg.mu_edge_min_hz_per_s
    edge_max = cfg.mu_edge_max_hz_per_s

    if center_min is None:
        center_min = mu_min
    if center_max is None:
        center_max = mu_min + 0.25 * span
    if edge_min is None:
        edge_min = mu_max - 0.25 * span
    if edge_max is None:
        edge_max = mu_max

    return float(center_min), float(center_max), float(edge_min), float(edge_max)


def make_smooth_negative_mu_profile(
    rng: np.random.Generator,
    cfg: TrackDatasetBuildConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate a smooth all-negative Doppler-rate profile for one track."""

    frames = int(cfg.frames_per_track)
    if frames <= 0:
        raise ValueError("frames_per_track must be positive.")

    center_min, center_max, edge_min, edge_max = _default_profile_ranges(cfg)
    mu_center = float(rng.uniform(center_min, center_max))
    mu_edge = float(rng.uniform(edge_min, edge_max))
    if mu_center > mu_edge:
        mu_center, mu_edge = mu_edge, mu_center

    profile_center = float(rng.uniform(cfg.profile_center_min, cfg.profile_center_max))
    profile_width = float(rng.uniform(cfg.profile_width_min, cfg.profile_width_max))
    u = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    shape = np.exp(-0.5 * ((u - profile_center) / max(profile_width, 1e-6)) ** 2)
    shape = (shape - np.min(shape)) / (np.max(shape) - np.min(shape) + 1e-12)

    mu = mu_edge + (mu_center - mu_edge) * shape
    if cfg.mu_frame_jitter_std_hz_per_s > 0.0:
        mu = mu + rng.normal(0.0, cfg.mu_frame_jitter_std_hz_per_s, size=frames)

    mu_low = min(cfg.mu_min_hz_per_s, cfg.mu_max_hz_per_s)
    mu_high = max(cfg.mu_min_hz_per_s, cfg.mu_max_hz_per_s)
    mu = np.clip(mu, mu_low, mu_high)

    meta = {
        "track_mu_center_hz_per_s": float(mu_center),
        "track_mu_edge_hz_per_s": float(mu_edge),
        "track_profile_center": float(profile_center),
        "track_profile_width": float(profile_width),
    }
    return mu.astype(np.float64), meta


def make_orbital_mu_profile(
    cfg: TrackDatasetBuildConfig,
    altitude_km: float,
    maximum_elevation_deg: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Sample consecutive frame starts from one exact circular-pass profile."""

    profile = circular_pass_profile(
        carrier_hz=cfg.carrier_frequency_hz,
        altitude_m=float(altitude_km) * 1000.0,
        max_elevation_deg=float(maximum_elevation_deg),
        min_elevation_deg=float(cfg.minimum_elevation_deg),
        time_step_s=float(cfg.frame_spacing_s),
        include_earth_rotation=bool(cfg.include_earth_rotation),
    )
    if len(profile.time_s) < cfg.frames_per_track:
        raise ValueError("Orbital pass has fewer samples than frames_per_track.")
    indices = np.linspace(0, len(profile.time_s) - 1, cfg.frames_per_track).round().astype(np.int64)
    mu = profile.doppler_rate_hz_per_s[indices].astype(np.float64)
    time = profile.time_s[indices].astype(np.float64)
    meta = {
        "track_mu_center_hz_per_s": float(np.min(mu)),
        "track_mu_edge_hz_per_s": float(max(mu[0], mu[-1])),
        "track_profile_center": 0.5,
        "track_profile_width": 1.0,
        "maximum_elevation_deg": float(maximum_elevation_deg),
    }
    return mu, time, meta


def _track_snr_profile(
    rng: np.random.Generator,
    cfg: TrackDatasetBuildConfig,
    track_id: int,
) -> np.ndarray:
    if cfg.snr_profile_mode == "balanced_grid":
        n_elev = len(cfg.maximum_elevations_deg)
        if n_elev <= 0 or cfg.tracks_per_modulation % n_elev != 0:
            raise ValueError("balanced_grid requires tracks_per_modulation divisible by the elevation count.")
        repetitions = cfg.tracks_per_modulation // n_elev
        local_track = int(track_id) % cfg.tracks_per_modulation
        repetition = local_track // n_elev
        centers = np.linspace(cfg.snr_db_min, cfg.snr_db_max, repetitions)
        base = float(centers[repetition])
        # Reuse the same perturbation profile for every class/elevation at a
        # given grid level. This makes SNR an exactly controlled covariate;
        # waveform/noise realizations remain independent in the channel RNG.
        profile_rng = np.random.default_rng(cfg.seed + 104729 * (repetition + 1))
    elif cfg.snr_profile_mode == "random":
        base = float(rng.uniform(cfg.snr_db_min, cfg.snr_db_max))
        profile_rng = rng
    else:
        raise ValueError("trajectory.snr_profile_mode must be 'random' or 'balanced_grid'.")
    if cfg.snr_track_std_db <= 0.0:
        snr = np.full(cfg.frames_per_track, base, dtype=np.float64)
    else:
        snr = base + profile_rng.normal(0.0, cfg.snr_track_std_db, size=cfg.frames_per_track)
    return np.clip(snr, cfg.snr_db_min, cfg.snr_db_max)


def build_track_dataset_arrays(
    build_cfg: TrackDatasetBuildConfig,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(build_cfg.seed)
    generator = LEOSignalGenerator(modulations=list(build_cfg.modulations), rng=rng)

    total_tracks = build_cfg.tracks_per_modulation * len(build_cfg.modulations)
    total_samples = total_tracks * build_cfg.frames_per_track

    iq_list: list[np.ndarray] = []
    label_list: list[int] = []
    snr_list: list[float] = []
    fd0_list: list[float] = []
    fd0_frame_start_list: list[float] = []
    mu_list: list[float] = []
    gamma_list: list[float] = []
    modulation_list: list[bytes] = []
    domain_list: list[int] = []
    altitude_list: list[float] = []
    inclination_list: list[float] = []
    fs_list: list[float] = []
    fc_list: list[float] = []
    T_list: list[float] = []
    track_id_list: list[int] = []
    frame_id_list: list[int] = []
    frame_time_list: list[float] = []
    track_frame_count_list: list[int] = []
    track_mu_center_list: list[float] = []
    track_mu_edge_list: list[float] = []
    track_profile_center_list: list[float] = []
    track_profile_width_list: list[float] = []
    pilot_indices_list: list[np.ndarray] = []
    pilot_sample_indices_list: list[np.ndarray] = []
    pilot_symbols_list: list[np.ndarray] = []
    pilot_pattern_list: list[bytes] = []
    num_pilots_list: list[int] = []
    timing_offset_list: list[int] = []
    maximum_elevation_list: list[float] = []
    trajectory_mode_list: list[bytes] = []

    track_specs: list[tuple[int, str]] = []
    track_id = 0
    for modulation in build_cfg.modulations:
        for _ in range(build_cfg.tracks_per_modulation):
            track_specs.append((track_id, modulation))
            track_id += 1

    iterator = tqdm(track_specs, desc="Generating LEO track dataset") if show_progress else track_specs

    for track_id, modulation in iterator:
        domain_id = int(rng.choice(build_cfg.domain_values))
        altitude = float(rng.choice(build_cfg.orbit_altitudes_km))
        inclination = float(rng.choice(build_cfg.orbit_inclinations_deg))
        trajectory_mode = str(build_cfg.profile_mode).lower()
        maximum_elevation = float(
            build_cfg.maximum_elevations_deg[
                int(track_id) % len(build_cfg.maximum_elevations_deg)
            ]
        )
        if trajectory_mode == "orbital":
            mu_profile, frame_times, mu_meta = make_orbital_mu_profile(
                build_cfg, altitude, maximum_elevation
            )
        elif trajectory_mode == "synthetic":
            mu_profile, mu_meta = make_smooth_negative_mu_profile(rng, build_cfg)
            frame_times = np.arange(build_cfg.frames_per_track) * build_cfg.frame_spacing_s
            maximum_elevation = float("nan")
        else:
            raise ValueError("trajectory.profile_mode must be 'synthetic' or 'orbital'.")
        snr_profile = _track_snr_profile(rng, build_cfg, track_id)
        fd0_hz = float(rng.uniform(build_cfg.fd0_min_hz, build_cfg.fd0_max_hz))

        for frame_id in range(build_cfg.frames_per_track):
            frame_time_s = float(frame_times[frame_id])
            timing_offset = int(
                rng.integers(
                    min(build_cfg.timing_offset_min_samples, build_cfg.timing_offset_max_samples),
                    max(build_cfg.timing_offset_min_samples, build_cfg.timing_offset_max_samples) + 1,
                )
            )
            spec = SignalSpec(
                modulation=modulation,
                num_symbols=build_cfg.num_symbols,
                samples_per_symbol=build_cfg.samples_per_symbol,
                sample_rate_hz=build_cfg.sample_rate_hz,
                carrier_frequency_hz=build_cfg.carrier_frequency_hz,
                snr_db=float(snr_profile[frame_id]),
                fd0_hz=fd0_hz,
                mu_hz_per_s=float(mu_profile[frame_id]),
                rrc_beta=build_cfg.rrc_beta,
                rrc_span=build_cfg.rrc_span,
                channel_type=build_cfg.channel_type,
                compensate_constant_doppler=build_cfg.compensate_constant_doppler,
                target_num_samples=build_cfg.target_num_samples,
                pilot_enabled=build_cfg.pilot_enabled,
                pilot_pattern=build_cfg.pilot_pattern,
                pilot_interval_symbols=build_cfg.pilot_interval_symbols,
                pilot_guard_symbols=build_cfg.pilot_guard_symbols,
                pilot_num_symbols=build_cfg.pilot_num_symbols,
                pilot_min_spacing_symbols=build_cfg.pilot_min_spacing_symbols,
                pilot_block_length_symbols=build_cfg.pilot_block_length_symbols,
                pilot_block_positions=build_cfg.pilot_block_positions,
                pilot_modulation=build_cfg.pilot_modulation,
                pilot_symbol_mode=build_cfg.pilot_symbol_mode,
                pilot_symbol_index=build_cfg.pilot_symbol_index,
                pilot_seed=build_cfg.pilot_seed,
                timing_offset_samples=timing_offset,
            )
            sample = generator.generate(spec)

            iq_list.append(sample.iq.astype(np.float32))
            label_list.append(sample.label)
            snr_list.append(sample.info["snr_db"])
            fd0_list.append(sample.info["fd0_hz"])
            fd0_frame_start_list.append(sample.info["fd0_frame_start_hz"])
            mu_list.append(sample.info["mu_hz_per_s"])
            gamma_list.append(sample.info["gamma"])
            modulation_list.append(sample.modulation.encode("utf-8"))
            domain_list.append(domain_id)
            altitude_list.append(altitude)
            inclination_list.append(inclination)
            fs_list.append(sample.info["sample_rate_hz"])
            fc_list.append(sample.info["carrier_frequency_hz"])
            T_list.append(sample.info["observation_time_s"])
            track_id_list.append(track_id)
            frame_id_list.append(frame_id)
            frame_time_list.append(frame_time_s)
            track_frame_count_list.append(build_cfg.frames_per_track)
            track_mu_center_list.append(mu_meta["track_mu_center_hz_per_s"])
            track_mu_edge_list.append(mu_meta["track_mu_edge_hz_per_s"])
            track_profile_center_list.append(mu_meta["track_profile_center"])
            track_profile_width_list.append(mu_meta["track_profile_width"])
            pilot_indices_list.append(sample.pilot_indices.astype(np.int64))
            pilot_sample_indices_list.append((sample.pilot_indices * build_cfg.samples_per_symbol).astype(np.int64))
            pilot_symbols_list.append(sample.pilot_symbols.astype(np.complex64))
            pilot_pattern_list.append(str(build_cfg.pilot_pattern if build_cfg.pilot_enabled else "none").encode("utf-8"))
            num_pilots_list.append(int(len(sample.pilot_indices)))
            timing_offset_list.append(int(sample.info["timing_offset_samples"]))
            maximum_elevation_list.append(maximum_elevation)
            trajectory_mode_list.append(trajectory_mode.encode("utf-8"))

    if len(iq_list) != total_samples:
        raise RuntimeError(f"Generated {len(iq_list)} samples, expected {total_samples}.")

    return {
        "iq": np.stack(iq_list, axis=0).astype(np.float32),
        "label": np.asarray(label_list, dtype=np.int64),
        "snr_db": np.asarray(snr_list, dtype=np.float32),
        "fd0": np.asarray(fd0_list, dtype=np.float32),
        "fd0_frame_start": np.asarray(fd0_frame_start_list, dtype=np.float32),
        "mu": np.asarray(mu_list, dtype=np.float32),
        "gamma": np.asarray(gamma_list, dtype=np.float32),
        "modulation": np.asarray(modulation_list, dtype="S16"),
        "domain_id": np.asarray(domain_list, dtype=np.int64),
        "orbit_altitude_km": np.asarray(altitude_list, dtype=np.float32),
        "orbit_inclination_deg": np.asarray(inclination_list, dtype=np.float32),
        "fs": np.asarray(fs_list, dtype=np.float32),
        "fc": np.asarray(fc_list, dtype=np.float32),
        "T": np.asarray(T_list, dtype=np.float32),
        "track_id": np.asarray(track_id_list, dtype=np.int64),
        "frame_id": np.asarray(frame_id_list, dtype=np.int64),
        "frame_time_s": np.asarray(frame_time_list, dtype=np.float32),
        "track_frame_count": np.asarray(track_frame_count_list, dtype=np.int64),
        "track_mu_center_hz_per_s": np.asarray(track_mu_center_list, dtype=np.float32),
        "track_mu_edge_hz_per_s": np.asarray(track_mu_edge_list, dtype=np.float32),
        "track_profile_center": np.asarray(track_profile_center_list, dtype=np.float32),
        "track_profile_width": np.asarray(track_profile_width_list, dtype=np.float32),
        "pilot_indices": np.stack(pilot_indices_list, axis=0).astype(np.int64),
        "pilot_sample_indices": np.stack(pilot_sample_indices_list, axis=0).astype(np.int64),
        "pilot_symbols": np.stack(pilot_symbols_list, axis=0).astype(np.complex64),
        "pilot_pattern": np.asarray(pilot_pattern_list, dtype="S16"),
        "num_pilots": np.asarray(num_pilots_list, dtype=np.int64),
        "timing_offset_samples": np.asarray(timing_offset_list, dtype=np.int64),
        "maximum_elevation_deg": np.asarray(maximum_elevation_list, dtype=np.float32),
        "trajectory_mode": np.asarray(trajectory_mode_list, dtype="S16"),
        "label_names": np.asarray([m.encode("utf-8") for m in build_cfg.modulations], dtype="S16"),
    }


def summarize_track_dataset(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    labels = np.asarray(arrays["label"])
    track_id = np.asarray(arrays["track_id"])
    mu = np.asarray(arrays["mu"])
    snr = np.asarray(arrays["snr_db"])
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    unique_tracks, track_counts = np.unique(track_id, return_counts=True)

    return {
        "num_samples": int(len(labels)),
        "num_tracks": int(len(unique_tracks)),
        "track_length_min": int(np.min(track_counts)),
        "track_length_max": int(np.max(track_counts)),
        "num_classes": int(len(unique_labels)),
        "class_counts": {int(k): int(v) for k, v in zip(unique_labels, label_counts)},
        "iq_shape": list(arrays["iq"].shape),
        "snr_db_min": float(np.min(snr)),
        "snr_db_max": float(np.max(snr)),
        "mu_min_hz_per_s": float(np.min(mu)),
        "mu_max_hz_per_s": float(np.max(mu)),
        "num_pilots": int(np.max(arrays["num_pilots"])) if "num_pilots" in arrays else 0,
        "adjacent_mu_abs_diff_median_hz_per_s": float(
            np.median(
                [
                    np.median(np.abs(np.diff(mu[track_id == tid])))
                    for tid in unique_tracks
                    if np.sum(track_id == tid) > 1
                ]
            )
        ),
    }


def build_track_dataset_from_config(
    cfg: Mapping[str, Any],
    output_h5: str | None = None,
    show_progress: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    build_cfg = TrackDatasetBuildConfig.from_mapping(cfg)
    logger = get_logger()
    logger.info("Building trajectory dataset with configuration:")
    logger.info(build_cfg.to_dict())

    arrays = build_track_dataset_arrays(build_cfg, show_progress=show_progress)
    summary = summarize_track_dataset(arrays)
    summary["build_config"] = build_cfg.to_dict()

    if output_h5 is not None:
        payload = dict(arrays)
        payload["meta"] = {
            "summary_json": str(summary),
            "dataset_type": "trajectory",
        }
        save_h5(output_h5, payload)
        logger.info(f"Saved trajectory dataset to {output_h5}")

    return arrays, summary
