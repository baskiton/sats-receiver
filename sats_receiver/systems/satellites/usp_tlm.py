import datetime as dt

from construct import *
from satellites.telemetry import ax25
from satellites.adapters import LinearAdapter, UNIXTimestampAdapter


class TimeDeltaAdapter(Adapter):
    def _encode(self, obj, context, path=None):
        return round(obj.total_seconds())

    def _decode(self, obj, context, path=None):
        return dt.timedelta(seconds=obj)


Addr = Hex(Int16ul)
RegTemp = LinearAdapter(100, Int16sl)
Voltage = LinearAdapter(1000, Int16ul)
VoltageS = LinearAdapter(1000, Int16sl)

BEACON = 0x4216
REGULAR = 0xDE21
XF210 = 0xF210
X4235 = 0x4235
X0118 = 0x0118
XDF3D = 0xDF3D

beacon_flags = BitStruct(
    'Uab_crit' / Flag,
    'Uab_min' / Flag,
    'heater2_manual' / Flag,
    'heater1_manual' / Flag,
    'heater2_on' / Flag,
    'heater1_on' / Flag,
    'Tab_max' / Flag,
    'Tab_min' / Flag,
    'channelon4' / Flag,
    'channelon3' / Flag,
    'channelon2' / Flag,
    'channelon1' / Flag,
    'Ich_limit4' / Flag,
    'Ich_limit3' / Flag,
    'Ich_limit2' / Flag,
    'Ich_limit1' / Flag,
    'reserved0' / BitsInteger(7),
    'charger' / Flag,
    'reserved1' / BitsInteger(8),
)

Beacon = Struct(
    'name' / Computed(u'Beacon'),
    'Usb1' / Voltage,
    'Usb2' / Voltage,
    'Usb3' / Voltage,
    'Isb1' / Int16ul,
    'Isb2' / Int16ul,
    'Isb3' / Int16ul,
    'Iab' / Int16sl,
    'Ich1' / Int16ul,
    'Ich2' / Int16ul,
    'Ich3' / Int16ul,
    'Ich4' / Int16ul,
    't1_pw' / Int16sl,
    't2_pw' / Int16sl,
    't3_pw' / Int16sl,
    't4_pw' / Int16sl,
    'flags' / beacon_flags,
    'Uab' / VoltageS,
    'reg_tel_id' / Int32ul,
    'Time' / UNIXTimestampAdapter(Int32sl),
    'Nres' / Int8ul,
    'FL' / Int8ul,
    't_amp' / Int8sl,
    't_uhf' / Int8sl,
    'RSSIrx' / Int8sl,
    'RSSIidle' / Int8sl,
    'Pf' / Int8sl,
    'Pb' / Int8sl,
    'Nres_uhf' / Int8ul,
    'Fl_uhf' / Int8ul,
    'Time_uhf' / UNIXTimestampAdapter(Int32sl),
    'UpTime' / TimeDeltaAdapter(Int32ul),
    'Current' / Int16ul,
    'Uuhf' / VoltageS,
)

regular_flags = BitStruct(
    'Uab_crit' / Flag,
    'Uab_min' / Flag,
    'heater2_manual' / Flag,
    'heater1_manual' / Flag,
    'heater2_on' / Flag,
    'heater1_on' / Flag,
    'Tab_max' / Flag,
    'Tab_min' / Flag,
    'channelon4' / Flag,
    'channelon3' / Flag,
    'channelon2' / Flag,
    'channelon1' / Flag,
    'Ich_limit4' / Flag,
    'Ich_limit3' / Flag,
    'Ich_limit2' / Flag,
    'Ich_limit1' / Flag,
    'reserved0' / BitsInteger(3),
    'add_channelon3' / Flag,
    'add_channelon2' / Flag,
    'add_channelon1' / Flag,
    'fsb' / Flag,
    'charger' / Flag,
    'reserved1' / BitsInteger(8),
)

Regular = Struct(
    'name' / Computed(u'Regular'),
    'Usb1' / Int16ul,
    'Usb2' / Int16ul,
    'Usb3' / Int16ul,
    'Isb1' / Int16ul,
    'Isb2' / Int16ul,
    'Isb3' / Int16ul,
    'Iab' / Int16sl,
    'Ich1' / Int16ul,
    'Ich2' / Int16ul,
    'Ich3' / Int16ul,
    'Ich4' / Int16ul,
    't1_pw' / RegTemp,
    't2_pw' / RegTemp,
    't3_pw' / RegTemp,
    't4_pw' / RegTemp,
    'flags' / regular_flags,
    'Uab' / VoltageS,
    'reg_tel_id' / Int32ul,
    'Time' / UNIXTimestampAdapter(Int32sl),
    'Nres' / Int8ul,
    'FL' / Int8ul,
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


Data = Struct(
    'message' / Addr,
    'sender' / Addr,
    'receiver' / Addr,
    'size' / Int16ul,
    'telemetry' / Switch(this.message, tlm_map, default=Bytes(this.size)),
)

Frame = Struct(
    'data' / GreedyRange(Data),
    'lost' / GreedyBytes
)

usp = Struct(
    'ax25' / ax25,
    'usp' / If(this.ax25.header.pid == 0xF0,
               Pointer(-len_(this.ax25.info), Frame))
)
