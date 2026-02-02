import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# 範例：ERA5 相容的年平均 TSI（1951.5–2034.5）
def era5_tsi_data():
    years = np.arange(1951.5, 2035.5, 1.0)
    tsi_vals = 0.9965 * np.array([
      # fmt: off
      # 1951-1995 (non-repeating sequence)
      1365.7765, 1365.7676, 1365.6284, 1365.6564, 1365.7773,
      1366.3109, 1366.6681, 1366.6328, 1366.3828, 1366.2767,
      1365.9199, 1365.7484, 1365.6963, 1365.6976, 1365.7341,
      1365.9178, 1366.1143, 1366.1644, 1366.2476, 1366.2426,
      1365.9580, 1366.0525, 1365.7991, 1365.7271, 1365.5345,
      1365.6453, 1365.8331, 1366.2747, 1366.6348, 1366.6482,
      1366.6951, 1366.2859, 1366.1992, 1365.8103, 1365.6416,
      1365.6379, 1365.7899, 1366.0826, 1366.6479, 1366.5533,
      1366.4457, 1366.3021, 1366.0286, 1365.7971, 1365.6996,
      # 1996-2008 (13 year cycle, repeated below)
      1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
      1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
      1365.8107, 1365.7240, 1365.6918,
      # 2009-2021
      1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
      1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
      1365.8107, 1365.7240, 1365.6918,
      # 2022-2034
      1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
      1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
      1365.8107, 1365.7240, 1365.6918,
      # fmt: on
  ])
    return xr.DataArray(tsi_vals, dims=['year'], coords={'year': years})

def get_tsi(timestamps, tsi_da):
    """
    timestamps: pandas.DatetimeIndex
    tsi_da: DataArray dims=('year',), coords year=float
    """
    # 計算 fractional year
    ts = pd.to_datetime(timestamps)
    doy = ts.dayofyear
    is_leap = ts.is_leap_year.astype(int)
    frac = (doy - 1 + (ts.hour / 24 + ts.minute / 1440 + ts.second / 86400)) \
           / (365 + is_leap)
    year_f = ts.year + frac
    # 線性插值
    return np.interp(year_f, tsi_da.year.values, tsi_da.values)

# 常數
J2000 = 2451545.0        # J2000 epoch (per ERA5-compatible logic)
SEC_PER_DAY = 86400.0
JULIAN_YEAR = 365.25

def julian_date(ts):
    """返回 timestamp 的朱利安日數（含小數）"""
    return ts.to_julian_date()

def orbital_params(ts):
    """
    ts: pandas.Timestamp
    回傳 dict 包含 theta, sin_decl, cos_decl, eq_time_sec, solar_dist_au, rot_phase
    """
    jd = julian_date(ts)
    d = jd - J2000
    theta = d / JULIAN_YEAR                              # 朱利安年數
    rot_phase = (d % 1.0)                                # 自轉相位
    # 以下多項式係數與原版相同
    rel   = 1.7535 + 6.283076 * theta
    rem   = 6.240041 + 6.283020 * theta
    rlls  = 4.8951 + 6.283076 * theta

    sin_rel, cos_rel = np.sin(rel), np.cos(rel)
    sin_2rel, cos_2rel = np.sin(2*rel), np.cos(2*rel)
    sin_2rlls, sin_4rlls = np.sin(2*rlls), np.sin(4*rlls)
    sin_rem, sin_2rem = np.sin(rem), np.sin(2*rem)

    # 太陽黃道經度
    rllls = (4.8952 + 6.283320*theta
             -0.0075*sin_rel -0.0326*cos_rel
             -0.0003*sin_2rel +0.0002*cos_2rel)

    repsm = 0.409093  # Obliquity of the ecliptic

    sin_decl = np.sin(repsm) * np.sin(rllls)
    cos_decl = np.sqrt(1 - sin_decl**2)

    eq_time = (591.8*sin_2rlls - 459.4*sin_rem
               + 39.5*sin_rem*np.cos(2 * rlls) -12.7*sin_4rlls -4.8*sin_2rem)

    solar_dist = 1.0001 -0.0163*np.cos(rel) +0.0037*np.sin(rel)

    return dict(
      theta=theta,
      rot_phase=rot_phase,
      sin_decl=sin_decl,
      cos_decl=cos_decl,
      eq_time=eq_time,
      solar_dist=solar_dist
    )
def instantaneous_flux(sin_lat, cos_lat, lon2d, params, tsi):
    """
    sin_lat, cos_lat: sine/cosine of latitude, shape (nlat, nlon)
    lon2d: in radians, shape (nlat, nlon)
    params: dict from orbital_params()
    tsi: scalar, W/m2
    """
    # 太陽時角 H = 2π⋅rot_phase + lon + eq_time/86400⋅2π (ERA5-compatible)
    solar_time = params['rot_phase'] + params['eq_time']/SEC_PER_DAY
    H = 2*np.pi*solar_time + lon2d
    sin_alt = (cos_lat * params['cos_decl'] * np.cos(H)
               + sin_lat * params['sin_decl'])
    sin_alt = np.maximum(sin_alt, 0.0)
    return tsi * sin_alt * (1.0 / params['solar_dist']**2)

def integrate_toa(times, lats, lons, integration_hours=1, bins=60):
    """
    times: list of pandas.Timestamp
    lats, lons: 1D arrays of selected lat and lon (degrees)
    返回 DataArray (time, lat, lon) in J/m2
    """
    # 經緯度網格
    lon2d, lat2d = np.meshgrid(np.radians(lons), np.radians(lats))
    sin_lat = np.sin(lat2d)
    cos_lat = np.cos(lat2d)
    # 讀取 TSI 資料
    tsi_da = era5_tsi_data()
    # 準備結果容器
    out = np.zeros((len(times), len(lats), len(lons)), dtype=float)

    # 時間分割
    dt = pd.Timedelta(hours=integration_hours)
    offsets = np.linspace(-dt.total_seconds(), 0, bins + 1) / SEC_PER_DAY  # 天
    trap_w = np.ones(bins + 1, dtype=float)
    trap_w[0] = 0.5
    trap_w[-1] = 0.5
    step_sec = dt.total_seconds() / bins

    for ti, t0 in enumerate(times):
        # 對每個子分段積分
        flux_acc = np.zeros_like(out[0])
        # 當前時間截的 TSI
        tsi_val = get_tsi([t0], tsi_da)[0]
        for w, off in zip(trap_w, offsets):
            t = t0 + pd.Timedelta(days=off)
            p = orbital_params(t)
            F = instantaneous_flux(sin_lat, cos_lat, lon2d, p, tsi_val)  # W/m2
            flux_acc += w * F
        # 梯形積分
        out[ti] = flux_acc * step_sec

    coords = dict(time=times, lat=lats, lon=lons)
    return xr.DataArray(out, dims=('time','lat','lon'), coords=coords)

# NOAA approximate solar position terms.
S0 = 1361.0  # W/m^2

def _noaa_solar_terms(dt_utc: datetime):
    """
    Returns:
      gamma: fractional year [rad]
      decl: solar declination [rad]
      eot: equation of time [minutes]
      dr:  Earth-Sun distance correction factor [-]
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt_utc.astimezone(timezone.utc)

    n = dt_utc.timetuple().tm_yday
    hour = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600

    gamma = 2*np.pi/365.0 * (n - 1 + (hour - 12)/24.0)

    decl = (
        0.006918
        - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
        - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
        - 0.002697*np.cos(3*gamma) + 0.001480*np.sin(3*gamma)
    )

    eot = 229.18 * (
        0.000075
        + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
        - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma)
    )

    dr = (
        1.00011
        + 0.034221*np.cos(gamma) + 0.001280*np.sin(gamma)
        + 0.000719*np.cos(2*gamma) + 0.000077*np.sin(2*gamma)
    )

    return gamma, decl, eot, dr


def integrate_toa_noaa(times, lats, lons, integration_hours=1, step_seconds=60):
    """
    NOAA-based TOA integration over the hour ending at each timestamp.
    Returns DataArray (time, lat, lon) in J/m2.
    """
    lon2d_deg, lat2d = np.meshgrid(lons, np.radians(lats))
    sin_lat = np.sin(lat2d)
    cos_lat = np.cos(lat2d)

    dt_seconds = integration_hours * 3600.0
    n = int(np.ceil(dt_seconds / step_seconds))
    sample_offsets = [(i + 0.5) * dt_seconds / n for i in range(n)]

    out = np.zeros((len(times), len(lats), len(lons)), dtype=float)

    for ti, t_end in enumerate(times):
        t_end = pd.Timestamp(t_end).to_pydatetime()
        t0 = t_end - timedelta(seconds=float(dt_seconds))
        flux_acc = np.zeros_like(out[0])
        for off in sample_offsets:
            t = t0 + timedelta(seconds=off)
            _, decl, eot, dr = _noaa_solar_terms(t)
            minutes = t.hour * 60 + t.minute + t.second / 60
            tst = (minutes + eot + 4.0 * lon2d_deg) % 1440.0
            ha = np.deg2rad(tst / 4.0 - 180.0)
            cosz = sin_lat * np.sin(decl) + cos_lat * np.cos(decl) * np.cos(ha)
            cosz = np.maximum(cosz, 0.0)
            flux_acc += S0 * dr * cosz
        mean_flux = flux_acc / n
        out[ti] = mean_flux * dt_seconds

    coords = dict(time=times, lat=lats, lon=lons)
    return xr.DataArray(out, dims=('time','lat','lon'), coords=coords)

import numpy as np
import pandas as pd
import xarray as xr

try:
    import pvlib
except ImportError as exc:  # pragma: no cover - makes error clearer at runtime
    raise ImportError(
        "pvlib is required for TOA integration. Install with `pip install pvlib`."
    ) from exc


SEC_PER_HOUR = 3600.0


def _ensure_utc(times):
    ts = pd.DatetimeIndex(pd.to_datetime(times))
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _toa_ghi_pvlib(sample_times, lats, lons, solarpos_method="nrel_numpy", dni_extra_method="spencer"):
    """
    Compute TOA (extraterrestrial) GHI for each sample time and grid point.

    Returns array with shape (ntime, nlat, nlon) in W/m^2.
    """
    times_utc = _ensure_utc(sample_times)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    ntime = len(times_utc)
    nlat = len(lats)
    nlon = len(lons)
    out = np.zeros((ntime, nlat, nlon), dtype=float)

    dni_extra = pvlib.irradiance.get_extra_radiation(times_utc, method=dni_extra_method)
    dni_extra = np.asarray(dni_extra, dtype=float)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            sp = pvlib.solarposition.get_solarposition(
                times_utc, lat, lon, method=solarpos_method
            )
            zenith = np.asarray(sp["zenith"], dtype=float)
            cosz = np.cos(np.deg2rad(zenith))
            cosz = np.maximum(cosz, 0.0)
            out[:, i, j] = dni_extra * cosz

    return out


def integrate_toa_pvlib(times, lats, lons, integration_hours=1, step_seconds=60, solarpos_method="nrel_numpy"):
    """
    PVlib-based TOA integration over the hour ending at each timestamp.

    Parameters
    ----------
    times : array-like of datetime-like
        End times of each integration window.
    lats, lons : 1D arrays (degrees)
        Grid coordinates.
    integration_hours : int/float
        Window length in hours (default 1).
    step_seconds : int
        Sample step in seconds for numerical integration (default 60).

    Returns
    -------
    xarray.DataArray with dims (time, lat, lon) in J/m^2.
    """
    times = pd.to_datetime(times)
    dt_seconds = float(integration_hours) * SEC_PER_HOUR
    n = int(np.ceil(dt_seconds / step_seconds))
    if n < 1:
        raise ValueError("step_seconds must be <= integration window")

    # Midpoint sampling within each window
    sample_offsets = (np.arange(n) + 0.5) * dt_seconds / n

    out = np.zeros((len(times), len(lats), len(lons)), dtype=float)

    for ti, t_end in enumerate(times):
        t_end = pd.Timestamp(t_end)
        t0 = t_end - pd.Timedelta(seconds=dt_seconds)
        sample_times = t0 + pd.to_timedelta(sample_offsets, unit="s")

        ghi_toa = _toa_ghi_pvlib(sample_times, lats, lons, solarpos_method=solarpos_method)
        mean_flux = np.mean(ghi_toa, axis=0)
        out[ti] = mean_flux * dt_seconds

    coords = dict(time=times, lat=np.asarray(lats, dtype=float), lon=np.asarray(lons, dtype=float))
    return xr.DataArray(out, dims=("time", "lat", "lon"), coords=coords)

# # 子區域邊界（經度-lon, 緯度-lat）
# lon_min, lon_max = 102, 149.75
# lat_min, lat_max =  0,  47.75

# # 假設我們只想算 2022‑02‑03 00UTC 在 lon[120,150], lat[20,45] 範圍
# times = pd.to_datetime(['2025-07-05T06:00'])
# lons = np.arange(lon_min, lon_max+0.25, 0.25)
# lats = np.arange(lat_min, lat_max+0.25, 0.25)

# toa_da = integrate_toa(times, lats, lons,
#                        integration_hours=1,
#                        bins=120)  # 分割更多 bins 可提升精度
# print(toa_da)
# toa_noaa = integrate_toa_noaa(times, lats, lons,
#                               integration_hours=1,
#                               step_seconds=30)
# print(toa_noaa)

# # # pvlib_toa = integrate_toa_pvlib(times, lats, lons, integration_hours=1, step_seconds=60)
# # # 3. 取第一个时间步并绘制等值线填色图
# # first = toa_da.isel(time=0)
# era5 = np.load("../npz/operational/sfc/average/2025070506.npz")
# era5_toa = era5['tisr']
# print(era5_toa)
# # ERA5 latitude is typically north->south; flip to align with ascending lats.
# era5_toa = np.flipud(era5_toa)
# plt.figure(figsize=(8, 6))
# plt.contourf(
#     first['lon'], first['lat'], first.values,
#     levels=20
# )
# plt.xlabel('Longitude (°E)')
# plt.ylabel('Latitude (°N)')
# plt.title(
#     f'TOA Incident Solar Radiation\n'
#     f'{times[0].strftime("%Y-%m-%d %H:%M UTC")}'
# )
# plt.colorbar(label='J/m²')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.contourf(
#     first['lon'], first['lat'], era5_toa,
#     levels=20
# )
# plt.xlabel('Longitude (°E)')
# plt.ylabel('Latitude (°N)')
# plt.title(
#     f'TOA Incident Solar Radiation\n'
#     f'{times[0].strftime("%Y-%m-%d %H:%M UTC")}'
# )
# plt.colorbar(label='J/m²')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.contourf(
#     first['lon'], first['lat'], toa_noaa.isel(time=0).values,
#     levels=20
# )
# plt.xlabel('Longitude (°E)')
# plt.ylabel('Latitude (°N)')
# plt.title(
#     f'TOA Incident Solar Radiation\n'
#     f'{times[0].strftime("%Y-%m-%d %H:%M UTC")}'
# )
# plt.colorbar(label='J/m²')
# plt.tight_layout()
# plt.show()

# # Difference plot (model - ERA5)
# diff = first.values - era5_toa
# abs_max = np.nanmax(np.abs(diff))
# plt.figure(figsize=(8, 6))
# plt.contourf(
#     first['lon'], first['lat'], diff,
#     levels=21, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max
# )
# plt.xlabel('Longitude (°E)')
# plt.ylabel('Latitude (°N)')
# plt.title(
#     f'TOA Difference (Model - ERA5)\n'
#     f'{times[0].strftime("%Y-%m-%d %H:%M UTC")}'
# )
# plt.colorbar(label='J/m²')
# plt.tight_layout()
# plt.show()

# # NOAA vs ERA5
# noaa_first = toa_noaa.isel(time=0)
# noaa_diff = noaa_first.values - era5_toa
# noaa_abs = np.nanmax(np.abs(noaa_diff))
# plt.figure(figsize=(8, 6))
# plt.contourf(
#     noaa_first['lon'], noaa_first['lat'], noaa_diff,
#     levels=21, cmap="RdBu_r", vmin=-noaa_abs, vmax=noaa_abs
# )
# plt.xlabel('Longitude (°E)')
# plt.ylabel('Latitude (°N)')
# plt.title(
#     f'NOAA TOA Difference (NOAA - ERA5)\n'
#     f'{times[0].strftime("%Y-%m-%d %H:%M UTC")}'
# )
# plt.colorbar(label='J/m²')
# plt.tight_layout()
# plt.show()
