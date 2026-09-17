"""The meteo_utils module contains utility functions for meteorological data.

"""

from numpy import arccos, arcsin, clip, cos, exp, isnan, log, nanmax, pi, sin, tan, where
from pandas import Series, to_numeric
from xarray import DataArray

# Specific heat of air [MJ kg-1 °C-1]
CP = 1.013 * 10**-3


def calc_psy(pressure, tmean=None):
    """Psychrometric constant [kPa °C-1].

    Parameters
    ----------
    pressure: array_like
        atmospheric pressure [kPa].
    tmean: array_like
        average day temperature [°C].

    Returns
    -------
        array_like containing the Psychrometric
        constant [kPa °C-1].

    Examples
    --------
    >>> psy = calc_psy(pressure, tmean)

    Notes
    -----
    if tmean is none:
        Based on equation 8 in :cite:t:`allen_crop_1998`.
    elif rh is None:
        From FAO (1990), ANNEX V, eq. 4.

    """
    if tmean is None:
        return 0.000665 * pressure
    else:
        lambd = calc_lambda(tmean)  # MJ kg-1
        return CP * pressure / (0.622 * lambd)


def adjust_wind(wind, zw=None, etype="os"):
    """Adjusted wind to 2m height [m/s].

    Parameters
    ----------
    wind: float or pandas.Series or xarray.DataArray, optional
        mean day wind speed [m/s].
    zw: float, optional
        average day temperature [m].
    etype: str, optional
        "os" => ASCE-PM method is applied for a reference surfaces representing
        clipped grass (a short, smooth crop). "rs" => ASCE-PM method is applied for a
        reference surfaces representing alfalfa (a taller, rougher agricultural crop).

    Returns
    -------
        array_like containing the Psychrometric
        constant [kPa °C-1].

    Notes
    -----
    From ACSE Appendix B

    """
    if zw is None:
        return wind
    else:
        if etype == "os":
            return wind * 4.87 / log(67.8 * zw - 5.42)
        if etype == "rs":
            return wind * 3.44 / log(16.3 * zw - 5.42)


def calc_vpc(tmean):
    """Slope of saturation vapour pressure curve at air Temperature [kPa °C-1].

    Parameters
    ----------
    tmean: array_like
        average day temperature [°C].

    Returns
    -------
        array_like containing the calculated
        Saturation vapour pressure [kPa °C-1].

    Examples
    --------
    >>> vpc = calc_vpc(tmean)

    Notes
    -----
    Based on equation 13. in :cite:t:`allen_crop_1998`.

    """
    es = calc_e0(tmean)
    return 4097.904 * es / (tmean + 237.3) ** 2


def calc_lambda(tmean):
    """Latent Heat of Vaporization [MJ kg-1].

    Parameters
    ----------
    tmean: array_like
        average day temperature [°C].

    Returns
    -------
    array_like containing the calculated Latent Heat
        of Vaporization [MJ kg-1].

    Examples
    --------
    >>> lambd = calc_lambda(tmean)

    Notes
    -----
    Based on equation (3-1) in :cite:t:`allen_crop_1998`.

    """
    return 2.501 - 0.002361 * tmean


def calc_press(elevation, pressure=None):
    """Atmospheric pressure [kPa].

    Parameters
    ----------
    elevation: array_like
        the site elevation [m].
    pressure: array_like, optional
        atmospheric pressure [kPa].

    Returns
    -------
    array_like containing the calculated atmospheric pressure [kPa].

    Examples
    --------
    >>> pressure = calc_press(elevation)

    Notes
    -----
    Based on equation 7 in :cite:t:`allen_crop_1998`.

    """
    if pressure is None and elevation is None:
        raise Exception("Please provide either pressure or the elevation!")
    if pressure is None:
        return 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26
    else:
        return pressure


def calc_rho(pressure, tmean, ea):
    """Atmospheric air density calculated according to :cite:t:`allen_crop_1998`.

    Parameters
    ----------
    pressure: array_like
        atmospheric pressure [kPa].
    tmean: array_like
        average day temperature [°C].
    ea: array_like
        actual vapour pressure [kPa].

    Returns
    -------
    float or pandas.Series or xarray.DataArray containing the calculated mean air
    density [kg/m3]

    Examples
    --------
    >>> rho = calc_rho(pressure, tmean, ea)

    Notes
    -----
    Based on equation (3-5) in :cite:t:`allen_crop_1998`.

    .. math:: rho = 3.486 \\frac{P}{T_{KV}}

    """
    # Virtual temperature [tkv]
    tkv = (273.16 + tmean) * (1 - 0.378 * ea / pressure) ** -1
    return 3.486 * pressure / tkv


def calc_g(rn, etype="os"):
    """Soil heat flux density [MJ m-2 h-1] hourly

    Parameters
    ----------
    rn: float or pandas.Series or xarray.DataArray
        net radiation [MJ m-2 h-1].
    etype: str, optional
        "os" => ASCE-PM method is applied for a reference surfaces representing
        clipped grass (a short, smooth crop). "rs" => ASCE-PM method is applied for a
        reference surfaces representing alfalfa (a taller, rougher agricultural crop).

    Returns
    -------
    array_like containing the calculated hourly soil heat density 
    [MJ/m2h]

    Notes
    -----
    Based on equation 65 in ASCE.

    """
    if etype == "os":
        return where((rn > 0), 0.1 * rn, 0.5 * rn)
    if etype == "rs":
        return where((rn > 0), 0.04 * rn, 0.2 * rn)


def calc_cd(rn, etype="os"):
    """denominator constant, hourly

    Parameters
    ----------
    rn: float or pandas.Series or xarray.DataArray
        net radiation [MJ m-2 h-1].
    etype: str, optional
        "os" => ASCE-PM method is applied for a reference surfaces representing
        clipped grass (a short, smooth crop). "rs" => ASCE-PM method is applied for a
        reference surfaces representing alfalfa (a taller, rougher agricultural crop).

    Returns
    -------
    array_like containing the calculated denominator constant
    [s m-1]

    Notes
    -----
    Based on Table 1 in ASCE.

    """
    if etype == "os":
        return where((rn > 0), 0.24, 0.96)
    if etype == "rs":
        return where((rn > 0), 0.25, 1.7)


def calc_e0(tmean):
    """Saturation vapor pressure at the air temperature T [kPa].

    Parameters
    ----------
    tmean: array_like
        average day temperature [°C].

    Returns
    -------
    array_like containing the calculated saturation vapor pressure at the air
    temperature tmean [kPa].

    Examples
    --------
    >>> e0 = calc_e0(tmean)

    Notes
    -----
    Based on equation 11 in :cite:t:`allen_crop_1998`.

    """
    return 0.6108 * exp(17.27 * tmean / (tmean + 237.3))


def calc_es(tmean=None, tmax=None, tmin=None):
    """Saturation vapor pressure [kPa].

    Parameters
    ----------
    tmean: array_like, optional
        average day temperature [°C].
    tmax: array_like, optional
        maximum day temperature [°C].
    tmin: array_like, optional
        minimum day temperature [°C].

    Returns
    -------
    float or pandas.Series or xarray.DataArray containing the calculated saturation
    vapor pressure [kPa].

    Examples
    --------
    >>> es = calc_es(tmean)

    Notes
    -----
    Based on equation 11, 12 in :cite:t:`allen_crop_1998`.

    """
    if tmax is not None:
        eamax = calc_e0(tmax)
        eamin = calc_e0(tmin)
        return (eamax + eamin) / 2
    else:
        return calc_e0(tmean)


def calc_ea(tmean=None, tmax=None, tmin=None, rhmax=None, rhmin=None, rh=None, ea=None):
    """Actual vapor pressure [kPa].

    Parameters
    ----------
    tmean: array_like, optional
        average day temperature [°C].
    tmax: array_like, optional
        maximum day temperature [°C].
    tmin: array_like, optional
        minimum day temperature [°C].
    rhmax: array_like, optional
        maximum daily relative humidity [%].
    rhmin: array_like, optional
        mainimum daily relative humidity [%].
    rh: array_like, optional
        mean daily relative humidity [%].
    ea: array_like, optional
        actual vapor pressure [kPa].

    Returns
    -------
    float or pandas.Series or xarray.DataArray containing the calculated actual vapor
    pressure [kPa].

    Examples
    --------
    >>> ea = calc_ea(tmean, rh)

    Notes
    -----
    Based on equation 17, 19 in :cite:t:`allen_crop_1998`.

    """
    if ea is not None:
        return ea
    if rhmax is not None:  # eq. 11
        esmax = calc_e0(tmax)
        esmin = calc_e0(tmin)
        return (esmin * rhmax / 200) + (esmax * rhmin / 200)
    else:  # eq. 14
        if tmax is not None:
            es = calc_es(tmax=tmax, tmin=tmin)
        else:
            # Hourly case
            es = calc_e0(tmean)
        return rh / 100 * es


def day_of_year(tindex):
    """Day of the year (1-365) based on pandas.Index.

    Parameters
    ----------
    tindex: pandas.DatetimeIndex

    Returns
    -------
    array_like with ints specifying day of year.

    """
    return Series(to_numeric(tindex.strftime("%j")), tindex, dtype=int)


def solar_declination(j):
    """Solar declination from day of year [rad].

    Parameters
    ----------
    j: array_like
        day of the year (1-365).

    Returns
    -------
    array_like of solar declination [rad].

    Notes
    -------
    Based on equations 24 in :cite:t:`allen_crop_1998`.

    """
    return 0.409 * sin(2.0 * pi / 365.0 * j - 1.39)


def sunset_angle(sol_dec, lat):
    """Sunset hour angle from latitude and solar declination - daily [rad].

    Parameters
    ----------
    sol_dec: array_like
        solar declination [rad].
    lat: array_like
        the site latitude [rad].

    Returns
    -------
    array_like containing the calculated sunset hour angle - daily [rad].

    Notes
    -----
    Based on equations 25 in :cite:t:`allen_crop_1998`.

    """
    if isinstance(lat, DataArray):
        lat = lat.expand_dims(dim={"time": sol_dec.index}, axis=0)
        return arccos(clip(-tan(sol_dec.values) * tan(lat).T, -1, 1)).T
    else:
        return arccos(clip(-tan(sol_dec) * tan(lat), -1, 1))


def sun_angle_mid(lat, sol_dec, sta):
    """Angle of the sun above the horizon at the midpoint of the hourly
    time period - hourly [rad]
    
    Parameters
    ----------
    lat: float or array_like
        Latitude [rad]
    sol_dec: float or array_like
        Solar declination [rad]
    sta: array_like
        Solar time angle at the midpoint of the period [rad]
    
    Returns
    -------
    Time series with the angle of the sun above the horizon at the midpoint 
    of the hourly or shorter time period.
    """
    beta = arcsin(sin(lat) * sin(sol_dec) + cos(lat) * cos(sol_dec) * cos(sta))
    return beta


def solar_times(sol_dec=None, lat=None, w=None, t=1):
    """The solar time angles at the beginning and end of each
    period - hourly [rad]
    
    Parameters
    ----------
    sol_dec: array_like
        solar declination [rad].
    lat: float or array_like
        Latitude [rad]
    w: array_like
        Solar time angle at the midpoint of the period [rad]
    t: array_like
        Length of the calculation period [hours]
    
    Returns
    -------
    array_like containing the solar time angles at the beginning or end 
    of each period
    """
    ws = sunset_angle(sol_dec, lat)
    w1 = w - ((pi * t) / 24)
    w2 = w + ((pi * t) / 24)

    w1 = where(w1 < -ws, -ws, w1)
    w2 = where(w2 < -ws, -ws, w2)
    w1 = where(w1 > ws, ws, w1)
    w2 = where(w2 > ws, ws, w2)
    w1 = where(w1 > w2, w2, w1)
    return w1, w2


def solar_time_mid_angle(tindex, lon, j=None, lz=90):
    """Solar time angle at the midpoint of the period - [rad]

    Parameters
    ----------
    tindex: pandas.DatetimeIndex
        Standard clock timestamp at the midpoint of the hour after 
        correcting for any daylight savings shift.
    lon: float or array_like
        Longitude of the solar radiation measurement site expressed as
        positive degrees west of Greenwich, England [deg]
    j: array_like
        day of the year (1-365).
    lz: float or array_like
        Longitude of the center of the local time zone expressed as
        positive degrees west of Greenwich, England. [deg]

    Returns
    -------
    array_like containing the calculated solar time angle at the midpoint 
    of the period - daily [rad].

    Notes
    -----
    Based on equations 55 in ASCE.

    """
    if j is None:
        j = day_of_year(tindex)
    b = (2 * pi * (j - 81)) / 364
    sc = 0.1645 * sin(2 * b) - 0.1255 * cos(b) - 0.025 * sin(b)

    hour = Series(to_numeric(tindex.strftime("%H")), tindex, dtype=int)
    minute = Series(to_numeric(tindex.strftime("%M")), tindex, dtype=int)
    t = hour + minute / 60 - 0.5
    sta = (pi / 12) * ((t + 0.06667 * (lz - lon) + sc) - 12)
    return sta


def daylight_hours(tindex, lat):
    """Daylight hours [hour].

    Parameters
    ----------
    tindex: pandas.DatetimeIndex
    lat: array_like
        the site latitude [rad].

    Returns
    -------
    pandas.Series or xarray.DataArray containing the calculated daylight hours [hour].

    Notes
    -----
    Based on equation 34 in :cite:t:`allen_crop_1998`.

    """
    j = day_of_year(tindex)
    sol_dec = solar_declination(j)
    sangle = sunset_angle(sol_dec, lat)
    # Account for subpolar belt which returns NaN values
    dl = 24 / pi * sangle
    if isinstance(lat, DataArray):
        sol_dec = ((dl / dl).T * sol_dec.values).T
    dl = where((sol_dec > 0) & (isnan(dl)), nanmax(dl), dl)
    dl = where((sol_dec < 0) & (isnan(dl)), 0, dl)
    return dl


def relative_distance(j):
    """Inverse relative distance between earth and sun from day of the year.

    Parameters
    ----------
    j: array_like
        day of the year (1-365).

    Returns
    -------
    pandas.Series specifying relative distance between earth and sun.

    Notes
    -------
    Based on equations 23 in :cite:t:`allen_crop_1998`.

    """
    return 1 + 0.033 * cos(2.0 * pi / 365.0 * j)


def extraterrestrial_r(tindex=None, lat=None, lon=None, lz=90, period="daily"):
    """
    Extraterrestrial daily radiation [MJ m-2 d-1].

    Parameters
    ----------
    tindex: pandas.DatetimeIndex
    lat: array_like
        the site latitude [rad].
    lon: float, array_like, optional
        the site longitude [decimal].
    lz: float or array_like, optional
        Longitude of the center of the local time zone expressed as
        positive degrees west of Greenwich, England. [deg]
    period: str, optional
        "daily" => ASCE-PM method is applied for daily time steps. "hourly" => ASCE-PM
        method is applied for hourly time steps.

    Returns
    -------
    array_like containing the calculated extraterrestrial radiation [MJ m-2 d-1]

    Notes
    -----
    Based on equation 21 in :cite:t:`allen_crop_1998`.

    """
    j = day_of_year(tindex)
    dr = relative_distance(j)
    sol_dec = solar_declination(j)

    if period == "daily":
        omega = sunset_angle(sol_dec, lat)
        if isinstance(lat, DataArray):
            lat = lat.expand_dims(dim={"time": sol_dec.index}, axis=0)
            xx = sin(sol_dec.values) * sin(lat.T)
            yy = cos(sol_dec.values) * cos(lat.T)
            return (118.08 / 3.141592654 * dr.values * (omega.T * xx + yy * sin(omega.T))).T
        else:
            xx = sin(sol_dec) * sin(lat)
            yy = cos(sol_dec) * cos(lat)
            return 118.08 / 3.141592654 * dr * (omega * xx + yy * sin(omega))
    elif period == "hourly":
        sta = solar_time_mid_angle(tindex, lon, lz=lz)
        omega1, omega2 = solar_times(sol_dec, lat, sta)
        xx = sin(sol_dec) * sin(lat)
        yy = cos(sol_dec) * cos(lat)
        return 12 / pi * 4.92 * dr.values * ((omega2 - omega1) * xx + yy * (sin(omega2) - sin(omega1)))


def calc_res_surf(
    lai=None, r_s=None, srs=0.002, co2=300, r_l=100, lai_eff=0, croph=0.12
):
    """Surface resistance [s m-1].

    Parameters
    ----------
    lai: float or pandas.Series or xarray.DataArray, optional
        leaf area index [-].
    r_s: float or pandas.Series or xarray.DataArray, optional
        surface resistance [s m-1].
    r_l: float or pandas.Series or xarray.DataArray, optional
        bulk stomatal resistance [s m-1].
    lai_eff: float, optional
        1 => LAI_eff = 0.5 * LAI
        2 => LAI_eff = lai / (0.3 * lai + 1.2)
        3 => LAI_eff = 0.5 * LAI; (LAI>4=4)
        4 => see :cite:t:`zhang_comparison_2008`.
    srs: float or pandas.Series or xarray.DataArray, optional
        Relative sensitivity of rl to Δ[CO2] :cite:t:`yang_hydrologic_2019`.
    co2: float or pandas.Series or xarray.DataArray
        CO2 concentration [ppm].
    croph: float or pandas.Series or xarray.DataArray, optional crop height [m].

    Returns
    -------
    float or pandas.Series or xarray.DataArray containing the calculated surface
    resistance [s / m]

    """
    if r_s:
        return r_s
    else:
        fco2 = 1 + srs * (co2 - 300)
        if lai is None:
            return fco2 * r_l / (0.5 * croph * 24)  # after FAO-56
        else:
            return fco2 * r_l / calc_laieff(lai=lai, lai_eff=lai_eff)


def calc_laieff(lai=None, lai_eff=0):
    """Effective leaf area index [-].

    Parameters
    ----------
    lai: pandas.Series/float, optional
        leaf area index [-].
    lai_eff: float, optional
        0 => LAI_eff = 0.5 * LAI
        1 => LAI_eff = lai / (0.3 * lai + 1.2)
        2 => LAI_eff = 0.5 * LAI; (LAI>4=4)
        3 => see :cite:t:`zhang_comparison_2008`.

    Returns
    -------
    pandas.Series containing the calculated effective leaf area index.

    """
    if lai_eff == 0:
        return 0.5 * lai
    if lai_eff == 1:
        return lai / (0.3 * lai + 1.2)
    if lai_eff == 2:
        laie = lai.copy()
        laie[(lai > 2) & (lai < 4)] = 2
        laie[lai > 4] = 0.5 * lai
        return laie
    if lai_eff == 3:
        laie = lai.copy()
        laie[lai > 4] = 4
        return laie * 0.5


def calc_res_aero(wind, croph=0.12, zw=2, zh=2, ra_method=0):
    """Aerodynamic resistance [s m-1].

    Parameters
    ----------
    wind: float or pandas.Series or xarray.DataArray
        mean day wind speed [m/s].
    croph: float or pandas.Series or xarray.DataArray, optional
        crop height [m].
    zw: float, optional
        height of wind measurement [m].
    zh: float, optional
         height of humidity and or air temperature measurement [m].
    ra_method: float, optional
        0 => ra = 208/wind
        1 => ra is calculated based on equation 36 in FAO (1990), ANNEX V.

    Returns
    -------
    pandas.Series containing the calculated aerodynamic resistance.

    """
    if ra_method == 0:
        wind = wind.where(wind != 0, 0.0001)
        return 208 / wind
    else:
        d = 0.667 * croph
        zom = 0.123 * croph
        zoh = 0.0123 * croph
        return (log((zw - d) / zom)) * (log((zh - d) / zoh) / (0.41**2) / wind)
