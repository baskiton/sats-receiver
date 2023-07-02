import construct
from satellites.telemetry.ax25 import Header as ax25_hdr
from satellites.adapters import UNIXTimestampAdapter


# GEOSCAN Telemetry Protocol
# https://download.geoscan.aero/site-files/%D0%9F%D1%80%D0%BE%D1%82%D0%BE%D0%BA%D0%BE%D0%BB%20%D0%BF%D0%B5%D1%80%D0%B5%D0%B4%D0%B0%D1%87%D0%B8%20%D1%82%D0%B5%D0%BB%D0%B5%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%B8.pdf


class SubAdapter(construct.Adapter):
    def __init__(self, v, *args, **kwargs):
        self.v = v
        construct.Adapter.__init__(self, *args, **kwargs)

    def _encode(self, obj, context, path=None):
        return int(obj + self.v)

    def _decode(self, obj, context, path=None):
        return obj - self.v


class MulAdapter(construct.Adapter):
    def __init__(self, v, *args, **kwargs):
        self.v = v
        construct.Adapter.__init__(self, *args, **kwargs)

    def _encode(self, obj, context, path=None):
        return int(round(obj / self.v))

    def _decode(self, obj, context, path=None):
        return float(obj) * self.v


Frame = construct.Struct(
    'time' / UNIXTimestampAdapter(construct.Int32ul),
    'Iab' / MulAdapter(0.0766, construct.Int16ul),      # mA
    'Isp' / MulAdapter(0.03076, construct.Int16ul),     # mA

    'Uab_per' / MulAdapter(0.00006928, construct.Int16ul),   # V
    'Uab_sum' / MulAdapter(0.00013856, construct.Int16ul),   # V

    'Tx_plus' / construct.Int8ul,   # deg C
    'Tx_minus' / construct.Int8ul,  # deg C
    'Ty_plus' / construct.Int8ul,   # deg C
    'Ty_minus' / construct.Int8ul,  # deg C
    'Tz_plus' / construct.Int8ul,   # undef
    'Tz_minus' / construct.Int8ul,  # deg C
    'Tab1' / construct.Int8ul,      # deg C
    'Tab2' / construct.Int8ul,      # deg C
    'CPU_load' / construct.Int8ul,  # %
    'Nres_obc' / SubAdapter(7476, construct.Int16ul),
    'Nres_CommU' / SubAdapter(1505, construct.Int16ul),
    'RSSI' / SubAdapter(99, construct.Int8ul),  # dBm

    'pad' / construct.GreedyBytes
)

geoscan = construct.Struct(
    'ax25' / construct.Peek(ax25_hdr),
    'ax25' / construct.If(lambda this: (bool(this.ax25) and this.ax25.addresses[0].callsign == u'BEACON'), ax25_hdr),
    'geoscan' / construct.If(lambda this: (bool(this.ax25) and this.ax25.pid == 0xF0), Frame),
)
