import datetime as dt
import http.client
import logging
import pathlib
import shutil
import urllib.error
import urllib.parse
import urllib.request

from typing import Mapping, Optional, Union

import ephem

from sats_receiver import TLEDIR, utils


class Tle:
    TD_ERR_DEF = dt.timedelta(seconds=5)

    def __init__(self, config: Mapping):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        self.config = {}
        self.tle_file = pathlib.Path(TLEDIR / 'dummy')
        self.last_update_tle = dt.datetime.fromtimestamp(0, dt.timezone.utc)
        self.objects: dict[str, tuple[ephem.EarthSatellite, tuple[str, str, str]]] = {}
        self.t_err = self.last_update_tle
        self.td_err = self.TD_ERR_DEF

        if not self.update_config(config):
            raise ValueError(f'{self.prefix}: Invalid config!')

        self.t_next = self.last_update_tle + dt.timedelta(days=self.update_period)

    def fill_objects(self, tle_f: pathlib.Path, t: dt.datetime):
        if tle_f is None:
            if t >= self.t_err:
                self.t_err = t + self.td_err
                self.td_err *= 2
                self.log.error('TLE file failed')
            return

        objects = {}

        with tle_f.open() as f:
            for line in f:
                names = set()
                while 0 < len(line) <= 69:
                    names.add(line.strip())
                    line = f.readline()

                try:
                    names.add(int(line[2:7]))
                except ValueError:
                    if t >= self.t_err:
                        self.t_err = t + self.td_err
                        self.td_err *= 2
                        self.log.error('Not TLE. Break')
                    return

                l1 = line.rstrip()
                l2 = f.readline().rstrip()
                for name in names:
                    x = utils.tle_generate(str(name), l1, l2, self.ignore_checksum, self.log)
                    if x:
                        objects[name] = x

        self.objects = objects
        shutil.move(tle_f, self.tle_file)
        self.td_err = self.TD_ERR_DEF
        self.t_err = t

        return 1

    def fetch_tle(self, t: dt.datetime):
        try:
            x = urllib.request.urlretrieve(self.url)
        except urllib.error.HTTPError as e:
            if t >= self.t_err:
                self.t_err = t + self.td_err
                self.td_err *= 2
                msg = f'Tle not fetched: {e}'
                if e.code == 400:
                    msg = f'{msg}: "{e.url}"'
                self.log.error('%s', msg)
            return
        except (ConnectionError, http.client.error, urllib.error.URLError, ValueError) as e:
            if t >= self.t_err:
                self.t_err = t + self.td_err
                self.td_err *= 2
                self.log.error('Tle not fetched: %s', e)
            return

        if self.fill_objects(x and pathlib.Path(x[0]) or None, t):
            self.last_update_tle = t
            self.log.info('Tle updated')

        return 1

    def update_config(self, config: Mapping):
        """
        :return: True if config update success
        """

        if config != self.config:
            if not self._validate_config(config):
                self.log.warning('invalid new config!')
                return

            self.log.debug('reconf')
            self.config = config

            fn = pathlib.Path(urllib.parse.urlparse(self.url).path).name
            self.tle_file = pathlib.Path(TLEDIR / fn)
            if self.tle_file.is_file():
                self.last_update_tle = dt.datetime.fromtimestamp(self.tle_file.stat().st_mtime, dt.timezone.utc)
                self.t_next = self.last_update_tle + dt.timedelta(days=self.update_period)
            else:
                if self.tle_file.is_dir():
                    shutil.rmtree(self.tle_file, True)
                else:
                    self.tle_file.unlink(True)
                self.tle_file.touch()
                self.t_next = self.last_update_tle = dt.datetime.fromtimestamp(0, dt.timezone.utc)

            self.fill_objects(self.tle_file, dt.datetime.now(dt.timezone.utc))

            return 1

    @staticmethod
    def _validate_config(config: Mapping) -> bool:
        return all(map(lambda x: x in config, [
            'url',
            'update_period',
        ]))

    @property
    def url(self) -> str:
        return self.config['url']

    @property
    def update_period(self) -> Union[int, float]:
        """
        Period of TLE update, days
        """

        return self.config['update_period']

    @property
    def ignore_checksum(self) -> bool:
        """
        Ignore TLE checksum
        """

        return self.config.get('ignore_checksum', False)

    def action(self, t: dt.datetime):
        if t >= self.t_next and self.fetch_tle(t):
            self.t_next = self.last_update_tle + dt.timedelta(days=self.update_period)
            return 1

    def get(self, name: str) -> Optional[tuple[ephem.EarthSatellite, tuple[str, str, str]]]:
        """
        Get TLE info by satellite name or NORAD number

        :return: Tuple of EarthSatellite object and 3 lines of TLE. Or None
        """

        return self.objects.get(name, None)

    def get_ephem(self, name: str) -> Optional[ephem.EarthSatellite]:
        """
        Get TLE object by satellite name or NORAD number
        """

        x = self.objects.get(name, None)
        return x and x[0]

    def get_tle(self, name: str) -> Optional[tuple[str, str, str]]:
        """
        Get raw TLE lines by satellite name or NORAD number
        """

        x = self.objects.get(name, None)
        return x and x[1]
