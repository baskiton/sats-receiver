import datetime as dt
import logging
import pathlib
import shutil
import urllib.error
import urllib.parse
import urllib.request

from typing import Mapping, Optional, Union

import ephem

from sats_receiver import TLEDIR


class Tle:
    def __init__(self, config: Mapping):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        self.config = {}
        self.tle_file = pathlib.Path(TLEDIR / 'dummy')
        self.last_update_tle = dt.datetime.fromtimestamp(0, dt.timezone.utc)
        self.objects: dict[str, tuple[ephem.EarthSatellite, tuple[str, str, str]]] = {}

        if not self.update_config(config):
            raise ValueError(f'{self.prefix}: Invalid config!')

        self.t_next = self.last_update_tle + dt.timedelta(days=self.update_period)

    @staticmethod
    def calc_checksum(full_line: str):
        checksum = 0
        for c in full_line[:-1]:
            if c.isnumeric():
                checksum += int(c)
            elif c == '-':
                checksum += 1
        return str(checksum)[-1]

    def fill_objects(self):
        self.objects.clear()
        with self.tle_file.open() as f:
            for line in f:
                names = set()
                while len(line) <= 69:
                    names.add(line.strip())
                    line = f.readline()
                names.add(int(line[2:7]))

                l1 = line.rstrip()
                l2 = f.readline().rstrip()
                for name in names:
                    try:
                        self.objects[name] = ephem.readtle(str(name), l1, l2), (str(name), l1, l2)
                    except ValueError as e:
                        if str(e).startswith('incorrect TLE checksum'):
                            self.log.warning('%s: for `%s` expect %s:%s, got %s:%s',
                                             e, name,
                                             self.calc_checksum(l1), l1[-1],
                                             self.calc_checksum(l2), l2[-1])
                        else:
                            raise e

    def fetch_tle(self):
        try:
            urllib.request.urlretrieve(self.url, self.tle_file)
        except urllib.error.HTTPError as e:
            msg = f'Tle not fetched: {e}'
            if e.code == 400:
                msg = f'{msg}: "{e.url}"'
            self.log.error('%s', msg)
            return
        except (urllib.error.URLError, ValueError) as e:
            self.log.error('Tle not fetched: %s', e)
            return

        self.last_update_tle = dt.datetime.now(dt.timezone.utc)
        self.fill_objects()

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
            else:
                if self.tle_file.is_dir():
                    shutil.rmtree(self.tle_file, True)
                else:
                    self.tle_file.unlink(True)
                self.tle_file.touch()
                self.last_update_tle = dt.datetime.fromtimestamp(0, dt.timezone.utc)

            self.fill_objects()

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

    def action(self, t: dt.datetime):
        if t >= self.t_next and self.fetch_tle():
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
