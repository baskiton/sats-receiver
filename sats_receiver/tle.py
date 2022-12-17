import datetime as dt
import logging
import pathlib
import shutil
import urllib.error
import urllib.parse
import urllib.request

import ephem

from sats_receiver import TLEDIR


class Tle:
    def __init__(self, config):
        self.config = {}
        self.tle_file = pathlib.Path(TLEDIR / 'dummy')
        self.last_update_tle = dt.datetime.fromtimestamp(0, dt.timezone.utc)
        self.objects: dict[str, ephem.EarthSatellite] = {}

        if not self.update_config(config):
            raise ValueError('Tle: Invalid config!')

    def fill_objects(self):
        self.objects.clear()
        with self.tle_file.open() as f:
            for line in f:
                names = []
                while len(line) <= 69:
                    names.append(line.strip())
                    line = f.readline()

                if not names:
                    names.append(line[2:7])

                l1 = line
                l2 = f.readline()
                for name in names:
                    self.objects[name.rstrip()] = ephem.readtle(name.rstrip(), l1, l2)

    def fetch_tle(self):
        try:
            with urllib.request.urlopen(self.url) as r:
                tle = r.read()
                self.tle_file.write_bytes(tle)
                self.last_update_tle = dt.datetime.now(dt.timezone.utc)
        except urllib.error.HTTPError as e:
            msg = f'Tle not fetched: {e}'
            if e.code == 400:
                msg = f'{msg}: "{e.url}"'
            logging.error('Tle: %s', msg)
            return
        except (urllib.error.URLError, ValueError) as e:
            logging.error('Tle: Tle not fetched: %s', e)
            return

        self.fill_objects()

        logging.info('Tle: Tle updated')

    def update_config(self, config):
        if config != self.config:
            if not self._validate_config(config):
                logging.warning('Tle: invalid new config!')
                return

            logging.debug('Tle: reconf')
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

    def _validate_config(self, config):
        return all(map(lambda x: x in config, [
            'url',
            'update_period',
        ]))

    @property
    def url(self):
        return self.config['url']

    @property
    def update_period(self):
        return self.config['update_period']

    def action(self, t):
        if self.last_update_tle < (t - dt.timedelta(days=self.update_period)):
            self.fetch_tle()
            return 1

    def get(self, name) -> ephem.EarthSatellite:
        return self.objects.get(name, None)
