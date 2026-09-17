import pyet as et
import pandas as pd
import unittest
import numpy as np

from pandas import testing
from xarray import testing as xr_testing
from pyet import day_of_year


class TestHourly(unittest.TestCase):
    lat = 0.7053    # radians
    lon = 104.78    # degrees
    elevation = 1462.4
    zw = 3
    lz = 105

    tmean = pd.Series(
        [30.9, 31.2, 29.1, 28.3, 26.0, 22.9, 20.1, 19.9],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    rh = pd.Series(
        [24.4, 25.31, 30.03, 31.46, 33.62, 42.97, 57.38, 58.09],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    rs = pd.Series(
        [2.24, 1.65, 0.34, 0.32, 0.08, 0.00, 0.00, 0.00],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    wind_speed = pd.Series(
        [4.07, 3.58, 1.15, 3.04, 2.21, 1.04, 0.58, 0.95],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_day_of_year(self):
        tindex = self.rs.index
        j = day_of_year(tindex)
        testing.assert_series_equal(self.julian, j)

    julian = pd.Series(
        [183, 183, 183, 183, 183, 183, 183, 183],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_calc_e0(self):
        es_result = et.calc_e0(self.tmean)
        es_result = es_result.round(3)
        testing.assert_series_equal(self.es, es_result)

    # Testing calc_e0(tmean)
    es = pd.Series(
        [4.467, 4.544, 4.029, 3.846, 3.361, 2.792, 2.353, 2.324],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_adjust_wind(self):
        wind_result = et.meteo_utils.adjust_wind(self.wind_speed, zw=self.zw)
        wind_result = wind_result.round(2)
        xr_testing.assert_allclose(self.wind_speed_2m.to_xarray(), wind_result.to_xarray(), rtol=1)

    # Testing adjust_wind(wind, zw=None, etype="os")
    wind_speed_2m = pd.Series(
        [3.74, 3.30, 1.06, 2.80, 2.03, 0.96, 0.53, 0.87],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_relative_distance(self):
        dr_result = et.meteo_utils.relative_distance(self.julian)
        testing.assert_series_equal(self.dr, dr_result)

    # Testing relative_distance(j)
    dr = pd.Series(
        [0.9670, 0.9670, 0.9670, 0.9670, 0.9670, 0.9670, 0.9670, 0.9670],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_solar_declination(self):
        declination_result = et.rad_utils.solar_declination(self.julian)
        declination_result = declination_result.round(4)
        testing.assert_series_equal(self.declination, declination_result)

    # Testing solar_declination(j)
    declination = pd.Series(
        [0.4017, 0.4017, 0.4017, 0.4017, 0.4017, 0.4017, 0.4017, 0.4017],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_sunset_angle(self):
        sunset_hr_angle_result = et.meteo_utils.sunset_angle(self.declination, self.lat)
        sunset_hr_angle_result = sunset_hr_angle_result.round(3)
        testing.assert_series_equal(self.sunset_hr_angle, sunset_hr_angle_result)

    # Testing sunset_angle(sol_dec, lat)
    sunset_hr_angle = pd.Series(
        [1.941, 1.941, 1.941, 1.941, 1.941, 1.941, 1.941, 1.941],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_solar_time_mid_angle(self):
        tindex = self.rs.index
        solar_time_angle_result = et.meteo_utils.solar_time_mid_angle(tindex, lon=self.lon, lz=105)
        solar_time_angle_result = round(solar_time_angle_result, 3)
        testing.assert_series_equal(self.solar_time_angle, solar_time_angle_result)

    # Testing solar_time_mid_angle(tindex, lon, lz=90)
    solar_time_angle = pd.Series(
        [0.904, 1.166, 1.428, 1.689, 1.951, 2.213, 2.475, 2.737],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_solar_times(self):
        w1_r, w2_r = et.meteo_utils.solar_times(sol_dec=self.declination, lat=self.lat, w=self.solar_time_angle, t=1)
        w1_r = w1_r.round(3)
        w2_r = w2_r.round(3)
        np.testing.assert_allclose(self.w1, w1_r, 0.01)
        np.testing.assert_allclose(self.w2, w2_r, 0.01)

    # Testing solar_times(sol_dec=None, lat=None, w=None, t=1)
    w1 = pd.Series(
        [0.773, 1.035, 1.297, 1.558, 1.820, 1.941, 1.941, 1.941],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )
    w2 = pd.Series(
        [1.035, 1.297, 1.558, 1.820, 1.941, 1.941, 1.941, 1.941],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def testing_extraterrestrial_r(self):
        tindex = self.rs.index
        ra_result = et.rad_utils.extraterrestrial_r(tindex=tindex,
                                                    lat=self.lat,
                                                    lon=self.lon,
                                                    lz=self.lz,
                                                    period="hourly")
        ra_result = ra_result.round(2)
        np.testing.assert_allclose(ra_result, self.ra)

    # Testing extraterrestrial_r(tindex=None, lat=None, lon=None, period="daily")
    ra = pd.Series(
        [3.26, 2.52, 1.68, 0.81, 0.09, 0.00, 0.00, 0.00],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    # Not hourly, but needed
    def testing_calc_rso(self):
        ea_r = et.meteo_utils.calc_ea(tmean=self.tmean, rh=self.rh).round(3)
        rso_result = et.rad_utils.calc_rso(self.ra, self.elevation, ea=ea_r, lat=self.lat, lon=self.lon, lz=self.lz, period="daily")
        rso_result = rso_result.round(2)
        testing.assert_series_equal(rso_result, self.rso)

    # Calculated Clear Sky Radiation
    # Testing calc_rso(ra, elevation, kab=None, ea=None, lat=None, lon=None, period="daily")
    rso = pd.Series(
        [2.54, 1.96, 1.31, 0.63, 0.07, 0.00, 0.00, 0.00],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_calc_rad_long(self):
        ea_r = et.meteo_utils.calc_ea(tmean=self.tmean, rh=self.rh).round(3)
        rnl_result = et.rad_utils.calc_rad_long(self.rs,
                                                tmean=self.tmean,
                                                rh=self.rh,
                                                elevation=self.elevation,
                                                lat=self.lat,
                                                lon=self.lon,
                                                lz=self.lz,
                                                rso=self.rso,
                                                ea=ea_r,
                                                period="hourly")
        rnl_result = rnl_result.round(3)
        # results are slightly different as expected since methods vary
        # Appendix D and Appendix B
        np.testing.assert_allclose(rnl_result, self.rnl, atol=0.3)

    # Testing calc_rad_long( ... )
    rnl = pd.Series(
        [0.284, 0.262, 0.000, 0.104, 0.313, 0.230, 0.211, 0.210],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_calc_rad_net(self):
        rn_r = et.rad_utils.calc_rad_net(rn=self.rn,
                                         rs=self.rs,
                                         lat=self.lat,
                                         lon=self.lon,
                                         tmean=self.tmean,
                                         rh=self.rh,
                                         elevation=self.elevation,
                                         rso=self.rso)
        rn_r = rn_r.round(3)
        testing.assert_series_equal(rn_r, self.rn)

    # Testing calc_rad_net( ... )
    rn = pd.Series(
        [1.440, 1.009, 0.262, 0.142, -0.251, -0.230, -0.211, -0.210],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_calc_g(self):
        g_os_result = et.meteo_utils.calc_g(self.rn, etype="os")
        g_os_result = g_os_result.round(3)
        # testing.assert_series_equal(g_os_result, self.g_os)
        np.testing.assert_allclose(g_os_result, self.g_os, rtol=0.01)

        g_rs_result = et.meteo_utils.calc_g(self.rn, etype="rs")
        g_rs_result = g_rs_result.round(3)
        # testing.assert_series_equal(g_rs_result, self.g_rs)
        np.testing.assert_allclose(g_rs_result, self.g_rs, rtol=0.01)

    # Testing calc_g(rn, etype="os")
    g_os = pd.Series(
        [0.144, 0.101, 0.026, 0.014, -0.126, -0.115, -0.105, -0.105],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )
    g_rs = pd.Series(
        [0.058, 0.040, 0.010, 0.006, -0.050, -0.046, -0.042, -0.042],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

    def test_pm_asce(self):
        et_os_r = et.pm_asce(tmean=self.tmean,
                             wind=self.wind_speed,
                             rs=self.rs,
                             rn=None,
                             g=None,
                             rh=self.rh,
                             zw=self.zw,
                             lat=self.lat,
                             lon=self.lon,
                             lz=self.lz,
                             elevation=self.elevation,
                             etype="os",
                             period="hourly")
        # results are slightly different as expected since methods vary
        # Appendix D and Appendix B
        np.testing.assert_allclose(et_os_r, self.et_os, atol=0.2)

        et_rs_r = et.pm_asce(tmean=self.tmean,
                             wind=self.wind_speed,
                             rs=self.rs,
                             rn=None,
                             g=None,
                             rh=self.rh,
                             zw=self.zw,
                             lat=self.lat,
                             lon=self.lon,
                             lz=self.lz,
                             elevation=self.elevation,
                             etype="rs",
                             period="hourly")
        # results are slightly different as expected since methods vary
        # Appendix D and Appendix B
        np.testing.assert_allclose(et_rs_r, self.et_rs, atol=0.2)

    # Testing pm_asce( ... )
    et_os = pd.Series(
        [0.61, 0.48, 0.14, 0.20, 0.06, 0.01, -0.01, -0.00],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )
    et_rs = pd.Series(
        [0.82, 0.66, 0.20, 0.33, 0.09, 0.02, -0.01, 0.00],
        index=pd.date_range("2000-07-01 16:00:00", "2000-07-01 23:00:00", freq="h")
    )

