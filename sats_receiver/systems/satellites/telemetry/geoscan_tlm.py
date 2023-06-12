import construct
from satellites.telemetry import ax25
from satellites.adapters import UNIXTimestampAdapter


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

    # 'Uab_avg' / LinearAdapter(1 / 0.0176, Int8ul),  # V
    # 'Uab1' / LinearAdapter(1 / 0.0352, Int8ul),     # V
    # 'Uab2' / LinearAdapter(1 / 0.0176, Int8ul),     # V
    # 'Uab_sum' / LinearAdapter(1 / 0.0352, Int8ul),  # V

    # 'Uab_avg' / Int8ul,
    # 'Uab1' / Int8ul,
    # 'Uab2' / Int8ul,
    # 'Uab_sum' / Int8ul,

    'Uab_avg' / MulAdapter(0.0352, construct.Int8ul),   # V
    'Uab1' / MulAdapter(0.0176, construct.Int8ul),      # V
    'Uab2' / MulAdapter(0.0352, construct.Int8ul),      # V
    'Uab_sum' / MulAdapter(0.0176, construct.Int8ul),   # V

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
    'pad' / construct.Bytes(22)
)

geoscan = construct.Struct(
    'ax25' / construct.Peek(construct.Hex(construct.Bytes(16))),
    'ax25' / construct.If(construct.this.ax25 == bytes.fromhex('848a82869e9c60a4a66460a640e103f0'), ax25),
    'geoscan' / construct.If(construct.this.ax25, construct.Seek(16) >> Frame),
)
