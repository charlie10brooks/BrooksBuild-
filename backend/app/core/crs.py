"""Minimal, dependency-free coordinate transforms: WGS84 lon/lat <-> WGS84 UTM (EPSG:326xx / 327xx) and Web Mercator (3857).

Only transforms that are exact for the stated datum are provided. Everything else (NAD83 UTM, State Plane, local grids...)
raises ``UnsupportedCrs``: silently treating NAD83 as WGS84 would introduce metre-level errors, so we refuse instead and ask
for data in the project CRS. (Install pyproj to extend this; the call sites are isolated here.)

UTM uses the Kruger n-series to 6th order (sub-millimetre inside a zone).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

A = 6378137.0
F = 1 / 298.257223563
K0 = 0.9996
N = F / (2 - F)
_AA = A / (1 + N) * (1 + N ** 2 / 4 + N ** 4 / 64 + N ** 6 / 256)
_ALPHA = [N / 2 - 2 * N ** 2 / 3 + 5 * N ** 3 / 16 + 41 * N ** 4 / 180 - 127 * N ** 5 / 288 + 7891 * N ** 6 / 37800,
          13 * N ** 2 / 48 - 3 * N ** 3 / 5 + 557 * N ** 4 / 1440 + 281 * N ** 5 / 630 - 1983433 * N ** 6 / 1935360,
          61 * N ** 3 / 240 - 103 * N ** 4 / 140 + 15061 * N ** 5 / 26880 + 167603 * N ** 6 / 181440,
          49561 * N ** 4 / 161280 - 179 * N ** 5 / 168 + 6601661 * N ** 6 / 7257600,
          34729 * N ** 5 / 80640 - 3418889 * N ** 6 / 1995840,
          212378941 * N ** 6 / 319334400]
_BETA = [N / 2 - 2 * N ** 2 / 3 + 37 * N ** 3 / 96 - N ** 4 / 360 - 81 * N ** 5 / 512 + 96199 * N ** 6 / 604800,
         N ** 2 / 48 + N ** 3 / 15 - 437 * N ** 4 / 1440 + 46 * N ** 5 / 105 - 1118711 * N ** 6 / 3870720,
         17 * N ** 3 / 480 - 37 * N ** 4 / 840 - 209 * N ** 5 / 4480 + 5569 * N ** 6 / 90720,
         4397 * N ** 4 / 161280 - 11 * N ** 5 / 504 - 830251 * N ** 6 / 7257600,
         4583 * N ** 5 / 161280 - 108847 * N ** 6 / 3991680,
         20648693 * N ** 6 / 638668800]
_E = math.sqrt(F * (2 - F))


class UnsupportedCrs(ValueError):
    pass


def utm_zone_params(epsg: int) -> Optional[Tuple[int, bool]]:
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    return None


def _lon0(zone: int) -> float:
    return math.radians(-183 + 6 * zone)


def lonlat_to_utm(lon, lat, zone: int, north: bool):
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    dl = lon - _lon0(zone)
    t = np.sinh(np.arctanh(np.sin(lat)) - _E * np.arctanh(_E * np.sin(lat)))
    xi_p = np.arctan2(t, np.cos(dl))
    eta_p = np.arctanh(np.sin(dl) / np.sqrt(1 + t * t))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, a in enumerate(_ALPHA, 1):
        xi += a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta += a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    x = 500000.0 + K0 * _AA * eta
    y = K0 * _AA * xi + (0.0 if north else 10000000.0)
    return x, y


def utm_to_lonlat(x, y, zone: int, north: bool):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xi = (y - (0.0 if north else 10000000.0)) / (K0 * _AA)
    eta = (x - 500000.0) / (K0 * _AA)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, 1):
        xi_p -= b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p -= b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    lat = chi.copy()
    # conformal -> geodetic latitude (series in e)
    e2 = _E ** 2
    d = [e2 / 2 + 5 * e2 ** 2 / 24 + e2 ** 3 / 12 + 13 * e2 ** 4 / 360,
         7 * e2 ** 2 / 48 + 29 * e2 ** 3 / 240 + 811 * e2 ** 4 / 11520,
         7 * e2 ** 3 / 120 + 81 * e2 ** 4 / 1120,
         4279 * e2 ** 4 / 161280]
    for j, c in enumerate(d, 1):
        lat += c * np.sin(2 * j * chi)
    lon = _lon0(zone) + np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    return np.degrees(lon), np.degrees(lat)


def lonlat_to_mercator(lon, lat):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return A * np.radians(lon), A * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def mercator_to_lonlat(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.degrees(x / A), np.degrees(2 * np.arctan(np.exp(y / A)) - np.pi / 2)


def transformer(src: int, dst: int) -> Callable:
    """fn(x, y) -> (x', y'). Raises UnsupportedCrs when the pair cannot be transformed exactly."""
    if src == dst:
        return lambda x, y: (np.asarray(x, float), np.asarray(y, float))

    def to_ll(epsg):
        if epsg == 4326:
            return lambda x, y: (np.asarray(x, float), np.asarray(y, float))
        if epsg == 3857:
            return mercator_to_lonlat
        z = utm_zone_params(epsg)
        if z:
            return lambda x, y: utm_to_lonlat(x, y, *z)
        raise UnsupportedCrs(f"EPSG:{epsg} is not supported for coordinate transformation without PROJ. Supported: 4326, 3857, "
                             "WGS84 UTM (326xx/327xx). Provide data in the project CRS instead.")

    def from_ll(epsg):
        if epsg == 4326:
            return lambda x, y: (x, y)
        if epsg == 3857:
            return lonlat_to_mercator
        z = utm_zone_params(epsg)
        if z:
            return lambda x, y: lonlat_to_utm(x, y, *z)
        raise UnsupportedCrs(f"EPSG:{epsg} is not supported for coordinate transformation without PROJ. Supported: 4326, 3857, "
                             "WGS84 UTM (326xx/327xx). Provide data in the project CRS instead.")
    a, b = to_ll(src), from_ll(dst)
    return lambda x, y: b(*a(x, y))


def transform_geometry(geom: dict, src: int, dst: int) -> dict:
    t = transformer(src, dst)

    def pos(p):
        x, y = t(p[0], p[1])
        return [float(x), float(y), *p[2:]]

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            return pos(c)
        return [walk(i) for i in c]
    return {**geom, "coordinates": walk(geom["coordinates"])}
