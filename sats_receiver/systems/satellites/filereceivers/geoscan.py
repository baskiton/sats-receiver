import datetime as dt

import construct
from satellites.filereceiver.imagereceiver import FileReceiver, ImageReceiver

from sats_receiver import utils


# GEOSCAN Telemetry Protocol
# https://download.geoscan.aero/site-files/%D0%9F%D1%80%D0%BE%D1%82%D0%BE%D0%BA%D0%BE%D0%BB%20%D0%BF%D0%B5%D1%80%D0%B5%D0%B4%D0%B0%D1%87%D0%B8%20%D1%82%D0%B5%D0%BB%D0%B5%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%B8.pdf


_frame = construct.Struct(
    'marker' / construct.Int16ul,           # #0
    'dlen' / construct.Int8ul,              # #2
    'mtype' / construct.Int16ul,            # #3
    'offset' / construct.Int16ul,           # #5
    'subsystem_num' / construct.Int8ul,     # #7
    'data' / construct.Bytes(construct.this.dlen - 6)
    # 'data' / construct.Bytes(56)
)


class FileReceiverGeoscan(FileReceiver):
    MARKER_FILE = 0x0002
    CMD_FILE_START = 0x0901
    CMD_FILE_FRAME = 0x0905
    BASE_OFFSET = 0     # old 16384  # old 32768

    def __init__(self, path, verbose=False):
        super().__init__(path, verbose)
        self.base_offset = self.BASE_OFFSET
        self._current_fid = None
        self._last_chunk_hash = None
        self._prev_chunk_sz = -1
        self._miss_cnt = 0
        self._cnt = 0

    def generate_fid(self):
        self._current_fid = f'GEOSCAN_{dt.datetime.now()}'.replace(' ', '_')
        return self._current_fid

    def filename(self, fid):
        return f'{fid}.bin'

    def parse_chunk(self, chunk):
        try:
            chunk = _frame.parse(chunk)
        except construct.ConstructError:
            return

        if chunk.marker != self.MARKER_FILE:
            self._miss_cnt += 1
            return

        if chunk.mtype == self.CMD_FILE_START:
            self.base_offset = chunk.offset
            chunk.offset = 0
            self._cnt += 1

        elif chunk.mtype == self.CMD_FILE_FRAME:
            chunk.offset -= self.base_offset
            self._cnt += 1

        else:
            return

        return chunk

    def file_id(self, chunk):
        ch_hash = hash(chunk.data)
        if chunk.offset == 0 and ch_hash != self._last_chunk_hash:
            # new file
            self.generate_fid()

        self._last_chunk_hash = ch_hash

        return self._current_fid or self.generate_fid()

    def on_completion(self, f):
        utils.close(f.f)
        self._current_fid = None
        self.base_offset = self.BASE_OFFSET


class ImageReceiverGeoscan(ImageReceiver):
    MARKER_IMG = 0x0001
    CMD_IMG_START = 0x0901
    CMD_IMG_FRAME = 0x0905
    BASE_OFFSET = 4     # old 16384  # old 32768

    def __init__(self, path, verbose=False, display=False, fullscreen=True):
        super().__init__(path, verbose, display, fullscreen)
        self.base_offset = self.BASE_OFFSET
        self._current_fid = None
        self._last_chunk_hash = None
        self._prev_chunk_sz = -1
        self._miss_cnt = 0

    def generate_fid(self):
        self._current_fid = f'GEOSCAN_{dt.datetime.now()}'.replace(' ', '_')
        return self._current_fid

    def parse_chunk(self, chunk):
        try:
            chunk = _frame.parse(chunk)
        except construct.ConstructError:
            return

        if chunk.marker != self.MARKER_IMG:
            self._miss_cnt += 1
            return

        if chunk.mtype == self.CMD_IMG_START:
            self.base_offset = chunk.offset
            chunk.offset = 0

        elif chunk.mtype == self.CMD_IMG_FRAME:
            chunk.offset -= self.base_offset

        else:
            return

        return chunk

    def file_id(self, chunk):
        ch_hash = hash(chunk.data)
        if (chunk.offset == 0
                and chunk.data.startswith(b'\xff\xd8')
                and ch_hash != self._last_chunk_hash):
            # new image
            self.generate_fid()

        self._last_chunk_hash = ch_hash

        return self._current_fid or self.generate_fid()

    def is_last_chunk(self, chunk):
        prev_sz = self._prev_chunk_sz
        self._prev_chunk_sz = len(chunk.data)
        return (self._prev_chunk_sz < prev_sz) and b'\xff\xd9' in chunk.data

    def on_completion(self, f):
        utils.close(f.f)
        self._current_fid = None
        self.base_offset = self.BASE_OFFSET
