import datetime as dt

import construct
from satellites.telemetry.ax25 import Header as ax25_hdr
from satellites.adapters import LinearAdapter, UNIXTimestampAdapter


class TimeDeltaAdapter(construct.Adapter):
    def _encode(self, obj, context, path=None):
        return round(obj.total_seconds())

    def _decode(self, obj, context, path=None):
        return dt.timedelta(seconds=obj)


Addr = construct.Hex(construct.Int16ul)
RegTemp = LinearAdapter(100, construct.Int16sl)
Voltage = LinearAdapter(1000, construct.Int16ul)
VoltageS = LinearAdapter(1000, construct.Int16sl)

BEACON = 0x4216
REGULAR = 0xDE21
XF210 = 0xF210
X4235 = 0x4235
X0118 = 0x0118
XDF3D = 0xDF3D

beacon_flags = construct.BitStruct(
    'Uab_crit' / construct.Flag,
    'Uab_min' / construct.Flag,
    'heater2_manual' / construct.Flag,
    'heater1_manual' / construct.Flag,
    'heater2_on' / construct.Flag,
    'heater1_on' / construct.Flag,
    'Tab_max' / construct.Flag,
    'Tab_min' / construct.Flag,
    'channelon4' / construct.Flag,
    'channelon3' / construct.Flag,
    'channelon2' / construct.Flag,
    'channelon1' / construct.Flag,
    'Ich_limit4' / construct.Flag,
    'Ich_limit3' / construct.Flag,
    'Ich_limit2' / construct.Flag,
    'Ich_limit1' / construct.Flag,
    'reserved0' / construct.BitsInteger(7),
    'charger' / construct.Flag,
    'reserved1' / construct.BitsInteger(8),
)

Beacon = construct.Struct(
    'name' / construct.Computed(u'Beacon'),
    'Usb1' / Voltage,
    'Usb2' / Voltage,
    'Usb3' / Voltage,
    'Isb1' / construct.Int16ul,
    'Isb2' / construct.Int16ul,
    'Isb3' / construct.Int16ul,
    'Iab' / construct.Int16sl,
    'Ich1' / construct.Int16ul,
    'Ich2' / construct.Int16ul,
    'Ich3' / construct.Int16ul,
    'Ich4' / construct.Int16ul,
    't1_pw' / construct.Int16sl,
    't2_pw' / construct.Int16sl,
    't3_pw' / construct.Int16sl,
    't4_pw' / construct.Int16sl,
    'flags' / beacon_flags,
    'Uab' / VoltageS,
    'reg_tel_id' / construct.Int32ul,
    'Time' / UNIXTimestampAdapter(construct.Int32sl),
    'Nres' / construct.Int8ul,
    'FL' / construct.Int8ul,
    't_amp' / construct.Int8sl,
    't_uhf' / construct.Int8sl,
    'RSSIrx' / construct.Int8sl,
    'RSSIidle' / construct.Int8sl,
    'Pf' / construct.Int8sl,
    'Pb' / construct.Int8sl,
    'Nres_uhf' / construct.Int8ul,
    'Fl_uhf' / construct.Int8ul,
    'Time_uhf' / UNIXTimestampAdapter(construct.Int32sl),
    'UpTime' / TimeDeltaAdapter(construct.Int32ul),
    'Current' / construct.Int16ul,
    'Uuhf' / VoltageS,
)

regular_flags = construct.BitStruct(
    'Uab_crit' / construct.Flag,
    'Uab_min' / construct.Flag,
    'heater2_manual' / construct.Flag,
    'heater1_manual' / construct.Flag,
    'heater2_on' / construct.Flag,
    'heater1_on' / construct.Flag,
    'Tab_max' / construct.Flag,
    'Tab_min' / construct.Flag,
    'channelon4' / construct.Flag,
    'channelon3' / construct.Flag,
    'channelon2' / construct.Flag,
    'channelon1' / construct.Flag,
    'Ich_limit4' / construct.Flag,
    'Ich_limit3' / construct.Flag,
    'Ich_limit2' / construct.Flag,
    'Ich_limit1' / construct.Flag,
    'reserved0' / construct.BitsInteger(3),
    'add_channelon3' / construct.Flag,
    'add_channelon2' / construct.Flag,
    'add_channelon1' / construct.Flag,
    'fsb' / construct.Flag,
    'charger' / construct.Flag,
    'reserved1' / construct.BitsInteger(8),
)

Regular = construct.Struct(
    'name' / construct.Computed(u'Regular'),
    'Usb1' / construct.Int16ul,
    'Usb2' / construct.Int16ul,
    'Usb3' / construct.Int16ul,
    'Isb1' / construct.Int16ul,
    'Isb2' / construct.Int16ul,
    'Isb3' / construct.Int16ul,
    'Iab' / construct.Int16sl,
    'Ich1' / construct.Int16ul,
    'Ich2' / construct.Int16ul,
    'Ich3' / construct.Int16ul,
    'Ich4' / construct.Int16ul,
    't1_pw' / RegTemp,
    't2_pw' / RegTemp,
    't3_pw' / RegTemp,
    't4_pw' / RegTemp,
    'flags' / regular_flags,
    'Uab' / VoltageS,
    'reg_tel_id' / construct.Int32ul,
    'Time' / UNIXTimestampAdapter(construct.Int32sl),
    'Nres' / construct.Int8ul,
    'FL' / construct.Int8ul,
)

# xf210 = Struct(
#     'name' / Computed(u'0xF210'),
#     # 'data' / Bytes(125),
#     'data' / GreedyRange(Int16ul),
#     'lost' / Byte,
# )
#
# x4235 = Struct(
#     'name' / Computed(u'0x4235'),
#     'aa' / Hex(Int16ul),
#     'bb' / Hex(Int16ul),
#     'cc' / Hex(Int16ul),
#     'dd' / Hex(Int16ul),
#     # 'ee' / Hex(Int16ul),
#     # 'data' / Bytes(10),
#     'ee' / Hex(Int16ul),
#     # 'data' / Bytes(8),
#     'data' / Bytes(this._.size - 10),
# )
#
# x0118 = Struct(
#     'name' / Computed(u'0x0118'),
#     # 'data' / Optional(Bytes(2))     # 2 if `from` == 0x02, 0 if `from` in (0x04, 0x07)
#     'data' / Bytes(this._.size),
# )
#
# xdf3d = Struct(
#     'name' / Computed(u'0xDF3D'),
#     'data' / Bytes(60)
# )

tlm_map = {
    BEACON: 'Beacon' / Beacon,
    REGULAR: 'Regular' / Regular,
    # XF210: 'XF210' / xf210,
    # X4235: 'X4235' / x4235,
    # X0118: 'X0118' / x0118,
    # XDF3D: 'XDF3D' / xdf3d,
}


Data = construct.Struct(
    'message' / Addr,
    'sender' / Addr,
    'receiver' / Addr,
    'size' / construct.Int16ul,
    'telemetry' / construct.Switch(construct.this.message, tlm_map, default=construct.Bytes(construct.this.size)),
)

Frame = construct.Struct(
    'data' / construct.GreedyRange(Data),
    'lost' / construct.GreedyBytes
)

usp = construct.Struct(
    'ax25' / construct.Peek(ax25_hdr),
    'ax25' / construct.If(lambda this: bool(this.ax25), ax25_hdr),
    'usp' / construct.If(lambda this: (bool(this.ax25) and this.ax25.pid == 0xF0), Frame)
)
