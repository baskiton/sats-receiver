import datetime as dt
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

import ephem


class Observer:
    def __init__(self, config):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        self.config = {}
        self.last_weather_time = dt.datetime.fromtimestamp(0, dt.timezone.utc)
        self.update_period = 1  # hours
        self.t_next = self.last_weather_time + dt.timedelta(hours=self.update_period, minutes=1)
        self._observer = ephem.Observer()

        if not self.update_config(config):
            raise ValueError(f'{self.prefix}: Invalid config!')

    @property
    def with_weather(self):
        return self.config['weather']

    @property
    def fetch_elev(self):
        return self.config['elevation'] is None

    def update_config(self, config):
        if self.config != config:
            if not self._validate_config(config):
                self.log.warning('invalid new config!')
                return

            self.log.debug('reconf')
            self.config = config

            self._observer = ephem.Observer()
            self._observer.lat = str(config['latitude'])
            self._observer.lon = str(config['longitude'])
            self._observer.elev = config['elevation'] or 0
            self._observer.compute_pressure()

            return 1

    def _validate_config(self, config):
        return all(map(lambda x: x in config, [
            'latitude',
            'longitude',
            'elevation',
            'weather',
        ]))

    def fetch_weather(self):
        q = urllib.parse.urlencode({
            'latitude': self._observer.lat / ephem.degree,
            'longitude': self._observer.lon / ephem.degree,
            'hourly': 'temperature_2m,surface_pressure',
            'current_weather': 'true',
            'windspeed_unit': 'ms',
            'start_date': dt.datetime.utcnow().date(),
            'end_date': dt.datetime.utcnow().date(),
        }, safe=',')

        try:
            with urllib.request.urlopen('https://api.open-meteo.com/v1/forecast?' + q) as r:
                j = json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = f'Weather not fetched!\n{e}'
            if e.code == 400:
                msg = f'{msg}:\n"{e.url}"'
            self.log.error('%s', msg)
            return

        self.last_weather_time = dt.datetime.fromisoformat(j['current_weather']['time']).replace(tzinfo=dt.timezone.utc)
        self._observer.temp = float(j['current_weather']['temperature'])
        if self.fetch_elev:
            self._observer.elev = j.get('elevation', self._observer.elev)

        press = None
        for i, val in enumerate(j['hourly']['time']):
            if dt.datetime.fromisoformat(val).replace(tzinfo=dt.timezone.utc) == self.last_weather_time:
                try:
                    press = float(j['hourly']['surface_pressure'][i])
                except TypeError:
                    pass
                break
        if press is None:
            self._observer.compute_pressure()
        else:
            self._observer.pressure = press

        self.log.info('weather updated: %s°C %shPa', self._observer.temp, self._observer.pressure)

        return 1

    def action(self, t):
        self.set_date(t)
        if self.with_weather and t >= self.t_next and self.fetch_weather():
            self.t_next = self.last_weather_time + dt.timedelta(hours=self.update_period, minutes=1)

    def next_pass(self, body: ephem.EarthSatellite, start_time=None):
        """
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

    def get_obj(self):
        return self._observer
