import datetime as dt
import http.client
import json
import logging
import threading
import urllib.error
import urllib.parse
import urllib.request

from typing import Mapping, Optional, Union

import ephem

from sats_receiver import HOMEDIR
from sats_receiver.utils import hpa_to_mmhg


class Observer:
    TD_ERR_DEF = dt.timedelta(seconds=5)
    FETCH_IDLE = 0
    FETCH_PROC = 1
    FETCH_DONE = 2
    FETCH_ERR = 3

    def __init__(self, config: Mapping, tpool):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        self.config = {}
        self.last_weather_time = dt.datetime.fromtimestamp(0, dt.timezone.utc)
        self.update_period = 1  # hours
        self.t_next = self.last_weather_time + dt.timedelta(hours=self.update_period, minutes=1)
        self._observer = ephem.Observer()
        self.t_err = self.last_weather_time
        self.td_err = self.TD_ERR_DEF
        self.weather_fp = HOMEDIR / 'weather.json'
        self.tpool = tpool
        self.future = None

        if not self.update_config(config):
            raise ValueError(f'{self.prefix}: Invalid config!')

    @property
    def with_weather(self) -> bool:
        return self.config['weather']

    @property
    def fetch_elev(self) -> bool:
        return self.config['elevation'] is None

    @property
    def lon(self) -> Union[int, float]:
        return self.config['longitude']

    @property
    def lat(self) -> Union[int, float]:
        return self.config['latitude']

    @property
    def elev(self) -> Union[int, float]:
        return self.config['elevation'] or 0

    @property
    def lonlat(self) -> tuple[Union[int, float], Union[int, float]]:
        return self.lon, self.lat

    def update_config(self, config: Mapping) -> Optional[int]:
        """
        :return: True if config update success
        """

        if self.config != config:
            if not self._validate_config(config):
                self.log.warning('invalid new config!')
                return

            self.log.debug('reconf')
            self.config = config

            self._observer = ephem.Observer()
            self._observer.lat = str(self.lat)
            self._observer.lon = str(self.lon)
            self._observer.elev = self.elev
            self._observer.compute_pressure()

            if self.with_weather:
                try:
                    self.set_weather(json.loads(self.weather_fp.read_bytes()))
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    self.log.warning('Failed to load weather from file: %s', e)

            return 1

    @staticmethod
    def _validate_config(config: Mapping) -> bool:
        return all(map(lambda x: x in config, [
            'latitude',
            'longitude',
            'elevation',
            'weather',
        ]))

    def fetch_weather(self) -> Optional[int]:
        q = urllib.parse.urlencode({
            'latitude': self._observer.lat / ephem.degree,
            'longitude': self._observer.lon / ephem.degree,
            'hourly': 'temperature_2m,surface_pressure',
            'current_weather': 'true',
            'windspeed_unit': 'ms',
            'start_date': dt.datetime.utcnow().date(),
            'end_date': dt.datetime.utcnow().date(),
        }, safe=',')

        with urllib.request.urlopen('https://api.open-meteo.com/v1/forecast?' + q) as r:
            return r.read()

    def set_weather(self, j):
        self.last_weather_time = dt.datetime.fromisoformat(j['current_weather']['time']).replace(tzinfo=dt.timezone.utc)
        self._observer.temp = float(j['current_weather']['temperature'])
        if self.fetch_elev:
            self._observer.elev = j.get('elevation', self._observer.elev)

        hourly = (self.last_weather_time.replace(minute=0, second=0, microsecond=0)
                  + dt.timedelta(hours=self.last_weather_time.minute // 30))
        press = None
        for i, val in enumerate(j['hourly']['time']):
            if dt.datetime.fromisoformat(val).replace(tzinfo=dt.timezone.utc) == hourly:
                try:
                    press = float(j['hourly']['surface_pressure'][i])
                except TypeError:
                    pass
                break
        if press is None:
            self._observer.compute_pressure()
        else:
            self._observer.pressure = press

    def action(self, t: dt.datetime) -> Optional[int]:
        self.set_date(t)

        if self.future and self.future.done():
            msg = ''
            try:
                result = self.future.result()
                j = json.loads(result)
            except urllib.error.HTTPError as e:
                msg = str(e)
                if e.code == 400:
                    msg = f'{msg}: "{e.url}"'
            except (ConnectionError, http.client.error, urllib.error.URLError) as e:
                msg = str(e)
            except json.JSONDecodeError as e:
                msg = f'JSON error: {e}'
            else:
                self.weather_fp.write_bytes(result)
                self.set_weather(j)
                self.log.info('weather updated: %.01f°C %.01fhPa (%.01fmmHg)',
                              self._observer.temp, self._observer.pressure, hpa_to_mmhg(self._observer.pressure))
                self.t_next = self.last_weather_time + dt.timedelta(hours=self.update_period, minutes=1)
                self.td_err = self.TD_ERR_DEF
                self.t_err = t
                return 1
            finally:
                self.future = None
                if msg and t >= self.t_err:
                    self.t_err = t + self.td_err
                    self.td_err *= 2
                    self.log.error('Weather not fetched: %s', msg)

        elif self.with_weather and t >= self.t_next and self.future is None:
            self.future = self.tpool.submit(self.fetch_weather)

    def next_pass(self,
                  body: ephem.EarthSatellite,
                  start_time: dt.datetime = None) -> tuple[dt.datetime, float,
                                                           dt.datetime, float,
                                                           dt.datetime, float]:
        """
        Calculate next pass of the `body` from `start_time`

        :return: rise_t, rise_az, culm_t, culm_alt, set_t, set_az
        """

        o = self._observer.copy()
        if start_time is not None:
            o.date = start_time

        rise_t, rise_az, culm_t, culm_alt, set_t, set_az = o.next_pass(body, False)

        return (ephem.to_timezone(rise_t, dt.timezone.utc), rise_az / ephem.degree,
                ephem.to_timezone(culm_t, dt.timezone.utc), culm_alt / ephem.degree,
                ephem.to_timezone(set_t, dt.timezone.utc), set_az / ephem.degree)

    def set_date(self, t: dt.datetime):
        self._observer.date = t

    def get_obj(self) -> ephem.Observer:
        return self._observer
