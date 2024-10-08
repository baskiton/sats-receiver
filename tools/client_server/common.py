import collections
import struct


HDR = struct.Struct('!4sBBI')
Hdr = collections.namedtuple('Hdr', 'sign ver cmd sz')

HDR_SIGN = b'SREC'
HDR_VER = 1

HDR_CMD_SEND = 1


def pack_hdr(cmd: int, data: bytes):
    return HDR.pack(*Hdr(HDR_SIGN, HDR_VER, cmd, len(data))) + data


def verify_hdr(hdr: Hdr):
    if hdr.sign != HDR_SIGN:
        return 'SIGN', ()

    if hdr.ver != HDR_VER:
        return 'VER', (REPLY_CMD_VER,)


REPLY_CMD_S = struct.Struct('!B')
REPLY_CMD_ERR = 1   # some err
REPLY_CMD_VER = 2   # invalid version
REPLY_CMD_RDY = 3
REPLY_CMD_END = 255

reply_name = {
    REPLY_CMD_ERR: 'ERR',
    REPLY_CMD_VER: 'VER',
    REPLY_CMD_RDY: 'RDY',
}
_reply_map = {
    REPLY_CMD_ERR: '',
    REPLY_CMD_VER: '',
    REPLY_CMD_RDY: 'Q',     # offset
}


def get_reply_struct(cmd):
    return struct.Struct('!' + _reply_map[cmd])


def pack_reply(cmd, *a):
    return REPLY_CMD_S.pack(cmd) + get_reply_struct(cmd).pack(*a)


def read_cmd(con):
    x = con.recv(REPLY_CMD_S.size)
    if x:
        cmd, = REPLY_CMD_S.unpack(x)
        s = get_reply_struct(cmd)
        return cmd, s.unpack(con.recv(s.size))
