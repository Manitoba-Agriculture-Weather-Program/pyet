"""The rad_utils module contains utility functions for radiation data.

"""
import pandas as pd
from numpy import clip, newaxis, sqrt, exp, sin, where
from pandas import Series
from xarray import DataArray

from .meteo_utils import (
    calc_ea,
    daylight_hours,
    extraterrestrial_r,
    sun_angle_mid,
    day_of_year,
    solar_declination,
    solar_time_mid_angle,
    calc_press
)
from .utils import check_rad, get_index, vectorize

# Stefan Boltzmann constant - hourly [MJm-2K-4h-1]
STEFAN_BOLTZMANN_HOUR = 2.042 * 10**-10
# Stefan Boltzmann constant - daily [MJm-2K-4d-1]
STEFAN_BOLTZMANN_DAY = 4.903 * 10**-9


def calc_rad_net(
    tmean,
    rn=None,
    rs=None,
    lat=None,
    lon=None,
    lz=None,
    n=None,
    nn=None,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    elevation=None,
    rso=None,
    a=1.35,
    b=-0.35,
    ea=None,
    albedo=0.23,
    as1=0.25,
    bs1=0.5,
    kab=None,
    period="daily",
):
    """Net radiation [MJ m-2 d-1].

    Parameters
    ----------
    tmean: pandas.Series/xarray.DataArray
        average day temperature [°C].
    rn: float or pandas.Series or xarray.DataArray, optional
        net radiation [MJ m-2 d-1].
    rs: float or pandas.Series or xarray.DataArray, optional
        incoming solar radiation [MJ m-2 d-1].
    lat: float/xarray.DataArray, optional
        the site latitude [rad].
    lon: float or xarray.DataArray, optional
        the site longitude [degree].
    lz: float or array_like
        Longitude of the center of the local time zone expressed as
        positive degrees west of Greenwich, England. [deg]
    n: float or pandas.Series or xarray.DataArray, optional
        actual duration of sunshine [hour].
    nn: float or pandas.Series or xarray.DataArray, optional
        maximum possible duration of sunshine or daylight hours [hour].
    tmax: float or pandas.Series or xarray.DataArray, optional
        maximum day temperature [°C].
    tmin: float or pandas.Series or xarray.DataArray, optional
        minimum day temperature [°C].
    rhmax: float or pandas.Series or xarray.DataArray, optional
        maximum daily relative humidity [%].
    rhmin: float or pandas.Series or xarray.DataArray, optional
        mainimum daily relative humidity [%].
    rh: float or pandas.Series or xarray.DataArray, optional
        mean daily relative humidity [%].
    elevation: float/xarray.DataArray, optional
        the site elevation [m].
    rso: float or pandas.Series or xarray.DataArray, optional
        clear-sky solar radiation [MJ m-2 day-1].
    a: float, optional
        empirical coefficient for Net Long-Wave radiation [-].
    b: float, optional
        empirical coefficient for Net Long-Wave radiation [-].
    ea: float or pandas.Series or xarray.DataArray, optional
        actual vapor pressure [kPa].
    albedo: float, optional
        surface albedo [-]
    as1: float, optional
        regression constant,  expressing the fraction of extraterrestrial reaching the
        earth on overcast days (n = 0) [-]
    bs1: float, optional
        empirical coefficient for extraterrestrial radiation [-]
    kab: float, optional
        coefficient derived from as1, bs1 for estimating clear-sky radiation [degrees].
    period: str, optional
        "daily" => ASCE-PM method is applied for daily time steps. "hourly" => ASCE-PM
        method is applied for hourly time steps.

    Returns
    -------
    float or pandas.Series or xarray.DataArray, optional containing the calculated net
    shortwave radiation.

    Notes
    -----
    Based on equation 40 in :cite:t:`allen_crop_1998`.

    """
    if rn is not None:
        rn = check_rad(rn)
        return rn
    else:
        if rs is None:
            rs = calc_rad_sol_in(n, lat, as1=as1, bs1=bs1, nn=nn)
        rns = calc_rad_short(
            rs=rs, lat=lat, n=n, nn=nn, albedo=albedo, as1=as1, bs1=bs1
        )  # [MJ/m2/d], [MJ/m2/h]
        rnl = calc_rad_long(
            rs=rs,
            tmean=tmean,
            tmax=tmax,
            tmin=tmin,
            rhmax=rhmax,
            rhmin=rhmin,
            rh=rh,
            elevation=elevation,
            lat=lat,
            lon=lon,
            lz=lz,
            rso=rso,
            a=a,
            b=b,
            ea=ea,
            kab=kab,
            period=period,
        )  # [MJ/m2/d], [MJ/m2/h]
        rn = rns - rnl
        rn = check_rad(rn)
        return rn


def calc_rad_long(
    rs,
    tmean=None,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    elevation=None,
    lat=None,
    lon=None,
    lz=None,
    rso=None,
    a=1.35,
    b=-0.35,
    ea=None,
    kab=None,
    period="daily",
):
    """Net longwave radiation [MJ m-2 d-1].

    Parameters
    ----------
    rs: float or pandas.Series or xarray.DataArray, optional
        incoming solar radiation [MJ m-2 d-1].
    tmean: float or pandas.Series or xarray.DataArray, optional
        average day temperature [°C].
    tmax: float or pandas.Series or xarray.DataArray, optional
        maximum day temperature [°C].
    tmin: float or pandas.Series or xarray.DataArray, optional
        minimum day temperature [°C].
    rhmax: float or pandas.Series or xarray.DataArray, optional
        maximum daily relative humidity [%].
    rhmin: float or pandas.Series or xarray.DataArray, optional
        minimum daily relative humidity [%].
    rh: float or pandas.Series or xarray.DataArray, optional
        mean daily relative humidity [%].
    elevation: float/xarray.DataArray, optional
        the site elevation [m].
    lat: float/xarray.DataArray, optional
        the site latitude [rad].
    lon: float or array_like
        Longitude of the solar radiation measurement site expressed as
        positive degrees west of Greenwich, England [deg].
    lz: float or array_like
        Longitude of the center of the local time zone expressed as
        positive degrees west of Greenwich, England. [deg]
    rso: float or pandas.Series or xarray.DataArray, optional
        clear-sky solar radiation [MJ m-2 day-1].
    a: float, optional
        empirical coefficient for Net Long-Wave radiation [-].
    b: float, optional
        empirical coefficient for Net Long-Wave radiation [-].
    ea: float or pandas.Series or xarray.DataArray, optional
        actual vapor pressure [kPa].
    kab: float, optional
        coefficient that can be derived from the as and bs coefficients of the
        Angstrom formula, where Kab = as + bs, and where Kab represents the
        fraction of extraterrestrial radiation reaching the earth on clear-sky
        days [-].
    period: str, optional
        "daily" => ASCE-PM method is applied for daily time steps. "hourly" => 
        ASCE-PM method is applied for hourly time steps.

    Returns
    -------
    float or pandas.Series or xarray.DataArray, optional containing the calculated net
    longwave radiation.

    Notes
    -----
    Based on equation 39 in :cite:t:`allen_crop_1998`.

    """
    if ea is None:
        ea = calc_ea(tmean=tmean, tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    if period == "hourly":
        sbc = STEFAN_BOLTZMANN_HOUR
    else:
        sbc = STEFAN_BOLTZMANN_DAY

    fcd = calc_ntcc(
        rs=rs,
        tmean=tmean,
        rh=rh,
        rso=rso,
        lat=lat,
        lon=lon,
        lz=lz,
        elevation=elevation,
        kab=kab,
        period="hourly",
        a=a,
        b=b
    )
    if tmax is not None and period == "daily":
        tmp1 = sbc * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
    else:
        tmp1 = sbc * (tmean + 273.16) ** 4
    tmp2 = 0.34 - 0.14 * sqrt(ea)  # OK
    tmp3 = fcd

    rnl = tmp1 * tmp2 * tmp3
    return rnl


def calc_rad_short(rs=None, lat=None, albedo=0.23, n=None, nn=None, as1=0.25, bs1=0.5):
    """Net shortwave radiation [MJ m-2 d-1].

    Parameters
    ----------
    rs: float or pandas.Series or xarray.DataArray, optional
        incoming solar radiation [MJ m-2 d-1].
    lat: float, optional
        the site latitude [rad].
    albedo: float or pandas.Series or xarray.DataArray, optional
        surface albedo [-].
    n: pandas.Series/xarray.DataArray, optional
        actual duration of sunshine [hour].
    as1: float, optional
        regression constant,  expressing the fraction of extraterrestrial reaching the
        earth on overcast days (n = 0) [-].
    bs1: float, optional
        empirical coefficient for extraterrestrial radiation [-].
    nn: float or pandas.Series or xarray.DataArray, optional
        maximum possible duration of sunshine or daylight hours [hour].

    Returns
    -------
    float or pandas.Series or xarray.DataArray, optional containing the calculated
    net shortwave radiation.

    Notes
    -----
    Based on equation 38 in :cite:t:`allen_crop_1998`.

    """
    (vrs,) = vectorize(rs)
    if vrs is not None:
        return (1 - albedo) * vrs
    else:
        return (1 - albedo) * calc_rad_sol_in(n, lat, as1=as1, bs1=bs1, nn=nn)


def calc_rad_sol_in(n, lat, as1=0.25, bs1=0.5, nn=None):
    """Incoming solar radiation [MJ m-2 d-1].

    Parameters
    ----------
    n: pandas.Series or xarray.DataArray
        actual duration of sunshine [hour].
    lat: float, optional
        the site latitude [rad].
    as1: float, optional
        regression constant,  expressing the fraction of extraterrestrial reaching the
        earth on overcast days (n = 0) [-].
    bs1: float, optional
        empirical coefficient for extraterrestrial radiation [-].
    nn: pandas.Series/float, optional
        maximum possible duration of sunshine or daylight hours [hour].

    Returns
    -------
    pandas.Series containing the calculated net shortwave radiation.

    Notes
    -----
    Based on equation 35 in :cite:t:`allen_crop_1998`.

    """
    tindex = get_index(n)
    ra = extraterrestrial_r(tindex, lat)
    if nn is None:
        nn = daylight_hours(tindex, lat)
    return (as1 + bs1 * n / nn) * ra


def calc_rso(ra, elevation, kab=None, ea=None, lat=None, lon=None, lz=None, period="daily"):
    """Clear-sky solar radiation [MJ m-2 day-1].

    Parameters
    ----------
    ra: pandas.Series/xarray.DataArray, optional
        Extraterrestrial daily radiation [MJ m-2 d-1].
    elevation: float/xarray.DataArray, optional
        the site elevation [m].
    kab: float, optional
        coefficient that can be derived from the as and bs coefficients of the
        Angstrom formula, where Kab = as + bs, and where Kab represents the
        fraction of extraterrestrial radiation reaching the earth on clear-sky
        days [-].
    ea: array_like, optional
        actual vapour pressure [kPa].
    lat: float/xarray.DataArray, optional
        the site latitude [rad].
    lon: float or xarray.DataArray, optional
        the site longitude [degree].
    lz: float or array_like
        Longitude of the center of the local time zone expressed as
        positive degrees west of Greenwich, England. [deg]
    period: str, optional
        "daily" => ASCE-PM method is applied for daily time steps. "hourly" => 
        ASCE-PM method is applied for hourly time steps [-].

    Returns
    -------
    pandas.Series/xarray.DataArray, optional containing the calculated Clear-sky solar
    radiation.

    Notes
    -----
    Based on equation 37 in :cite:t:`allen_crop_1998` and Appendix D.

    """
    if isinstance(elevation, DataArray):
        tindex = get_index(ra)
        elevation = elevation.expand_dims(dim={"time": tindex}, axis=0)
        if isinstance(ra, Series):
            ra = ra.values[:, newaxis, newaxis]
    if kab is None:
        if period == "hourly":
            tindex = get_index(ra)
            kb = calc_kb(elevation, ea, lat, lon, tindex, lz=lz)
            kd = 0.35 - 0.36 * kb
            return (kb + kd) * ra
        else:
            return (0.75 + (2 * 10 ** -5) * elevation) * ra
    else:
        return kab * ra


def calc_ntcc(
    rs,
    tmean,
    rh,
    lat,
    lon,
    elevation,
    tindex=None,
    sol_dec=None,
    sta=None,
    rso=None,
    kab=None,
    lz=None,
    period="hourly",
    a=1.35,
    b=-0.35
):
    """Nighttime cloudiness coefficient [MJ m-2 day-1].

    Parameters
    ----------
    rs: float or pandas.Series or xarray.DataArray, optional
        incoming solar radiation [MJ m-2 d-1].
    tmean: pandas.Series or xarray.DataArray
        average day temperature [°C].
    rh: float or pandas.Series or xarray.DataArray
        mean daily relative humidity [%].
    lat: float/xarray.DataArray, optional
        the site latitude [rad].
    lon: float or xarray.DataArray, optional
        the site longitude [degree].
    elevation: float/xarray.DataArray, optional
        the site elevation [m].
    tindex: pandas.Dataframe
    sol_dec: float or array_like
        Solar declination [rad]
    sta: array_like
        Solar time angle at the midpoint of the period [rad]
    rso: float or pandas.Series or xarray.DataArray, optional
        clear-sky solar radiation [MJ m-2 day-1].
    lz: float or array_like
        Longitude of the center of the local time zone expressed as
        positive degrees west of Greenwich, England. [deg]
    period: str, optional
        "daily" => ASCE-PM method is applied for daily time steps. "hourly" =>
        ASCE-PM method is applied for hourly time steps [-].
    a: float, optional
        empirical coefficient for Net Long-Wave radiation [-].
    b: float, optional
        empirical coefficient for Net Long-Wave radiation [-].

    Returns
    -------
    pandas.Series/xarray.DataArray, containing the cloudiness function, adjusted
    for the nighttime.

    Notes
    -----
    Based on equation 45, 46 in :cite:t:`allen_crop_1998`.

    """
    hour = pd.Timedelta(1, unit="hour")
    fcd = calc_fcd(
        rs=rs,
        tmean=tmean,
        rh=rh,
        lat=lat,
        lon=lon,
        lz=lz,
        elevation=elevation,
        kab=kab,
        tindex=tindex,
        sol_dec=sol_dec,
        sta=sta,
        rso=rso,
        period=period,
        a=a,
        b=b
    )

    if tindex is None:
        tindex = rs.index
    tindex = tindex.to_series()
    dt_tindex = tindex.copy()
    dt_tindex.loc[fcd.isna()] = pd.NaT

    ff = dt_tindex.ffill()
    ff.loc[ff.isna()] = tindex.loc[ff.isna()]
    bf = dt_tindex.bfill()
    bf.loc[bf.isna()] = tindex.loc[bf.isna()]

    ff_t_delta = tindex - bf
    bf_t_delta = tindex - ff
    t_delta = where(ff_t_delta > bf_t_delta, ff_t_delta, bf_t_delta)
    fcd.loc[t_delta >= hour] = fcd.interpolate(limit_direction="both", limit_area=None, method="nearest")
    fcd = fcd.ffill().bfill()
    return fcd


def calc_fcd(
    rs,
    lat,
    lon,
    elevation,
    kab=None,
    tindex=None,
    sol_dec=None,
    sta=None,
    rso=None,
    ea=None,
    tmean=None,
    rh=None,
    period="hourly",
    lz=90,
    a=1.35,
    b=-0.35
):
    """Nighttime cloudiness coefficient [MJ m-2 day-1].

    Parameters
    ----------
    rs: float or pandas.Series or xarray.DataArray
        incoming solar radiation [MJ m-2 d-1].
    lat: float/xarray.DataArray, optional
        the site latitude [rad].
    lon: float or xarray.DataArray, optional
        the site longitude [degree].
    elevation: float/xarray.DataArray, optional
        the site elevation [m].
    kab: float, optional
        coefficient derived from as1, bs1 for estimating clear-sky radiation [degrees].
    tindex: pandas.DatetimeIndex
    sol_dec: float or array_like, optional
        Solar declination [rad]
    sta: array_like
        Solar time angle at the midpoint of the period [rad]
    rso: float or pandas.Series or xarray.DataArray, optional
        clear-sky solar radiation [MJ m-2 day-1].
    ea: float or pandas.Series or xarray.DataArray
        actual vapor pressure [kPa].
    tmean: pandas.Series or xarray.DataArray
        average day temperature [°C].
    rh: float or pandas.Series or xarray.DataArray
        average relative humidity at 2pm [%].
    period: str, optional
        "daily" => ASCE-PM method is applied for daily time steps. "hourly" =>
        ASCE-PM method is applied for hourly time steps [-].
    lz: float or array_like
        Longitude of the center of the local time zone expressed as
        positive degrees west of Greenwich, England. [deg]
    a: float, optional
        empirical coefficient for Net Long-Wave radiation [-].
    b: float, optional
        empirical coefficient for Net Long-Wave radiation [-].

    Returns
    -------
    pandas.Series/xarray.DataArray, containing the cloudiness function..

    Notes
    -----
    Based on equation 45 in :cite:t:`allen_crop_1998`.

    """
    if tindex is None:
        tindex = rs.index
    if sol_dec is None:
        j = day_of_year(tindex)
        sol_dec = solar_declination(j)
    if sta is None:
        sta = solar_time_mid_angle(tindex, lon, lz=lz)

    beta = sun_angle_mid(lat, sol_dec, sta)

    if rso is None:
        ra = extraterrestrial_r(tindex=tindex, lat=lat, lon=lon, lz=lz, period=period)
        if ea is None:
            ea = calc_ea(tmean=tmean, rh=rh)
        rso = calc_rso(ra=ra, elevation=elevation, kab=kab, ea=ea, lat=lat, lon=lon, lz=lz, period=period)
    # Add a small constant to rso where it is zero to avoid division with zero
    rso = rso.where(rso != 0, 0.001)

    if len(rs.shape) == 3 and len(rso.shape) == 1:
        rso = rso.values[:, newaxis, newaxis]
    solar_rat = clip(rs / rso, 0.3, 1)

    fcd = a * solar_rat + b
    fcd = clip(fcd, 0.05, 1)

    fcd.loc[beta < 0.3] = None

    return fcd


def calc_kb(elevation, ea, lat, lon, tindex, kt=1, lz=None):
    """Clearness index for direct beam radiation [-]

    Parameters
    ----------
    elevation: array_like
        the site elevation [m].
    ea: float or pandas.Series or xarray.DataArray, optional
        actual vapor pressure [kPa].
    lat: float/xarray.DataArray,
        the site latitude [rad].
    lon: float or xarray.DataArray
        the site longitude [degree].
    tindex: pandas.DatetimeIndex
        Hourly time index. Not valid for daily values.
    kt: float/xarray.DataArray, optional [-]
        turbidity coefficient.
    lz: float or array_like
        Longitude of the center of the local time zone expressed as
        positive degrees west of Greenwich, England. [deg]

    Returns
    -------
    pandas.Series/xarray.DataArray, containing the midpoint sun angle.

    Notes
    -----
    Based on appendix D

    """
    p = calc_press(elevation)
    j = day_of_year(tindex)
    sol_dec = solar_declination(j)
    sta = solar_time_mid_angle(tindex, lon, lz=lz)
    theta = sun_angle_mid(lat, sol_dec, sta)
    w = 0.14 * ea * p + 2.1

    lt = (-0.00146 * p)/(kt * sin(theta))
    rt = 0.075 * ((w/sin(theta))**0.4)
    kb = 0.98 * exp(lt - rt)
    kb.loc[theta < 0.3] = None
    return kb
