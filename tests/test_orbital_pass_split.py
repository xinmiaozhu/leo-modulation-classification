import numpy as np

from src.datasets.split import elevation_holdout_split_indices
from src.datasets.build_track_dataset import TrackDatasetBuildConfig, make_orbital_mu_profile


def test_orbital_profile_is_exact_geometry_sampled_and_finite():
    cfg = TrackDatasetBuildConfig(
        frames_per_track=12,
        frame_spacing_s=2.0,
        profile_mode="orbital",
    )
    mu, time, meta = make_orbital_mu_profile(cfg, 600.0, 60.0)
    assert mu.shape == time.shape == (12,)
    assert np.all(np.isfinite(mu))
    assert np.all(np.diff(time) > 0)
    assert meta["maximum_elevation_deg"] == 60.0


def test_elevation_holdout_has_no_overlap():
    elevations = np.asarray([20.0, 20.0, 75.0, 90.0, 45.0])
    splits = elevation_holdout_split_indices(elevations, [20.0, 45.0], [75.0], [90.0])
    sets = [set(splits[name].tolist()) for name in ("train", "val", "test")]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert set.union(*sets) == set(range(len(elevations)))
