"""Circular-orbit geometry utilities for LEO Doppler-rate validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EARTH_RADIUS_M = 6_371_000.0
EARTH_GRAVITATIONAL_PARAMETER = 3.986_004_418e14
EARTH_ROTATION_RAD_PER_S = 7.292_115_9e-5
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


@dataclass(frozen=True)
class CircularPassProfile:
    time_s: np.ndarray
    elevation_deg: np.ndarray
    slant_range_m: np.ndarray
    range_rate_m_per_s: np.ndarray
    doppler_hz: np.ndarray
    doppler_rate_hz_per_s: np.ndarray
    orbital_speed_m_per_s: float
    relative_angular_rate_rad_per_s: float
    closest_central_angle_rad: float


def elevation_from_central_angle(
    central_angle_rad: float | np.ndarray,
    altitude_m: float,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> np.ndarray:
    """Return ground-terminal elevation for a spherical Earth."""

    psi = np.asarray(central_angle_rad, dtype=np.float64)
    radius = float(earth_radius_m)
    orbit_radius = radius + float(altitude_m)
    slant = np.sqrt(
        orbit_radius**2
        + radius**2
        - 2.0 * orbit_radius * radius * np.cos(psi)
    )
    sine_elevation = (orbit_radius * np.cos(psi) - radius) / np.maximum(slant, 1.0)
    return np.arcsin(np.clip(sine_elevation, -1.0, 1.0))


def central_angle_for_elevation(
    elevation_rad: float,
    altitude_m: float,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> float:
    """Invert the spherical elevation relation by bisection."""

    target = float(elevation_rad)
    if not (0.0 <= target <= 0.5 * np.pi):
        raise ValueError("elevation_rad must lie in [0, pi/2].")
    radius = float(earth_radius_m)
    orbit_radius = radius + float(altitude_m)
    lo = 0.0
    hi = float(np.arccos(radius / orbit_radius))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = float(elevation_from_central_angle(mid, altitude_m, radius))
        if value > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def circular_pass_profile(
    *,
    carrier_hz: float,
    altitude_m: float,
    max_elevation_deg: float = 90.0,
    min_elevation_deg: float = 10.0,
    time_step_s: float = 0.25,
    include_earth_rotation: bool = True,
    earth_radius_m: float = EARTH_RADIUS_M,
    gravitational_parameter: float = EARTH_GRAVITATIONAL_PARAMETER,
) -> CircularPassProfile:
    """Generate a spherical-Earth circular-pass Doppler profile.

    The pass is represented as a great-circle ground track with a closest
    angular separation chosen from ``max_elevation_deg``. Earth rotation is
    included as an equatorial co-rotating approximation when requested.
    """

    if max_elevation_deg <= min_elevation_deg:
        raise ValueError("max_elevation_deg must exceed min_elevation_deg.")
    radius = float(earth_radius_m)
    orbit_radius = radius + float(altitude_m)
    omega_orbit = np.sqrt(float(gravitational_parameter) / orbit_radius**3)
    omega_relative = omega_orbit - (
        EARTH_ROTATION_RAD_PER_S if include_earth_rotation else 0.0
    )
    orbital_speed = np.sqrt(float(gravitational_parameter) / orbit_radius)
    psi_min = central_angle_for_elevation(
        np.deg2rad(float(max_elevation_deg)),
        altitude_m,
        radius,
    )
    psi_visible = central_angle_for_elevation(
        np.deg2rad(float(min_elevation_deg)),
        altitude_m,
        radius,
    )
    argument = np.clip(np.cos(psi_visible) / np.cos(psi_min), -1.0, 1.0)
    half_duration = float(np.arccos(argument) / omega_relative)
    time = np.arange(
        -half_duration,
        half_duration + 0.5 * float(time_step_s),
        float(time_step_s),
        dtype=np.float64,
    )
    cos_psi = np.cos(psi_min) * np.cos(omega_relative * time)
    psi = np.arccos(np.clip(cos_psi, -1.0, 1.0))
    slant = np.sqrt(
        orbit_radius**2
        + radius**2
        - 2.0 * orbit_radius * radius * np.cos(psi)
    )
    # Analytic derivatives avoid numerical-differentiation artefacts when the
    # profile is used to quantify the local quadratic-phase approximation.
    amplitude = orbit_radius * radius * np.cos(psi_min) * omega_relative
    sine_argument = np.sin(omega_relative * time)
    range_rate = amplitude * sine_argument / np.maximum(slant, 1.0)
    range_acceleration = (
        amplitude * omega_relative * np.cos(omega_relative * time) / np.maximum(slant, 1.0)
        - amplitude * sine_argument * range_rate / np.maximum(slant**2, 1.0)
    )
    doppler = -float(carrier_hz) * range_rate / SPEED_OF_LIGHT_M_PER_S
    doppler_rate = -float(carrier_hz) * range_acceleration / SPEED_OF_LIGHT_M_PER_S
    elevation = elevation_from_central_angle(psi, altitude_m, radius)
    return CircularPassProfile(
        time_s=time,
        elevation_deg=np.rad2deg(elevation),
        slant_range_m=slant,
        range_rate_m_per_s=range_rate,
        doppler_hz=doppler,
        doppler_rate_hz_per_s=doppler_rate,
        orbital_speed_m_per_s=float(orbital_speed),
        relative_angular_rate_rad_per_s=float(omega_relative),
        closest_central_angle_rad=float(psi_min),
    )
