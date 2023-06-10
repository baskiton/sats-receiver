import sys

from ctypes import CDLL, c_char, c_char_p, c_int, c_uint32, c_void_p, POINTER


if sys.platform == 'win32':
    _lib = CDLL('rtlsdr')
else:
    _lib = CDLL('librtlsdr.so.0')


class LibRtlSdrError(RuntimeError):
    pass


class LibUsbError(RuntimeError):
    CODES = {
        0: 'Success (no error)',
        -1: 'Input/output error',
        -2: 'Invalid parameter',
        -3: 'Access denied (insufficient permissions)',
        -4: 'No such device (it may have been disconnected)',
        -5: 'Entity not found',
        -6: 'Resource busy',
        -7: 'Operation timed out',
        -8: 'Overflow',
        -9: 'Pipe error',
        -10: 'System call interrupted (perhaps due to signal)',
        -11: 'Insufficient memory',
        -12: 'Operation not supported or unimplemented on this platform',
        -99: 'Other error',
    }

    def __init__(self, err_code):
        super(LibUsbError, self).__init__(f'[{err_code}] {self.CODES.get(err_code, -99)}')


rtlsdr_dev_p = c_void_p

_f = _lib.rtlsdr_open
_f.argtypes = POINTER(rtlsdr_dev_p), c_uint32
_f.restype = c_int
def rtlsdr_open(idx=0) -> rtlsdr_dev_p:
    dev = rtlsdr_dev_p(None)
    if _lib.rtlsdr_open(dev, idx):
        raise LibRtlSdrError('Could not open RTL-SDR')
    return dev


_f = _lib.rtlsdr_set_bias_tee_gpio
_f.argtypes = rtlsdr_dev_p, c_int, c_int
_f.restype = c_int
def rtlsdr_set_bias_tee_gpio(dev: rtlsdr_dev_p, pin: int, on: int):
    if _lib.rtlsdr_set_bias_tee_gpio(dev, pin, on):
        raise LibRtlSdrError('Invalid device')


_f = _lib.rtlsdr_close
_f.argtypes = rtlsdr_dev_p,
_f.restype = c_int
def rtlsdr_close(dev: rtlsdr_dev_p):
    _lib.rtlsdr_close(dev)


_f = _lib.rtlsdr_get_device_count
_f.restype = c_uint32
def rtlsdr_get_device_count() -> int:
    return _lib.rtlsdr_get_device_count()


_f = _lib.rtlsdr_get_device_name
_f.argtypes = c_uint32,
_f.restype = c_char_p
def rtlsdr_get_device_name(idx: int) -> str:
    return _lib.rtlsdr_get_device_name(idx).decode('utf8')


_f = _lib.rtlsdr_get_device_usb_strings
_f.argtypes = c_uint32, POINTER(c_char), POINTER(c_char), POINTER(c_char)
_f.restype = c_int
def rtlsdr_get_device_usb_strings(index: int) -> tuple[str, str, str]:
    manufact = (c_char * 256)()
    product = (c_char * 256)()
    serial = (c_char * 256)()

    r = _lib.rtlsdr_get_device_usb_strings(index, manufact, product, serial)
    if r:
        raise LibUsbError(r)

    return manufact.value.decode('utf8'), product.value.decode('utf8'), serial.value.decode('utf8')


_f = _lib.rtlsdr_get_index_by_serial
_f.argtypes = POINTER(c_char),
_f.restype = c_int
def rtlsdr_get_index_by_serial(serial: str) -> int:
    r = _lib.rtlsdr_get_index_by_serial(serial.encode('utf8'))
    if r >= 0:
        return r

    # values below gettings from library source code
    if r == -1:
        m = 'Null serial'
    elif r == -2:
        m = 'No devices'
    elif r == -3:
        m = 'Device not found'
    else:   # unreachable
        m = f'Unknown {r}'

    raise LibRtlSdrError(m)


def get_serials() -> str:
    for i in range(rtlsdr_get_device_count()):
        yield rtlsdr_get_device_usb_strings(i)[2]


def set_bt(bt: int, serial: str = None, pin=0):
    """
    Set Bias-T state

    :param bt: Bias-t value (1 - enable, 0 - disable)
    :param serial: Serial Number of the device that needs to enable bias-t
    :param pin: GPIO pin to change bias-t state. 0 by default
    """
    i = serial and rtlsdr_get_index_by_serial(serial) or 0
    dev = rtlsdr_open(i)
    rtlsdr_set_bias_tee_gpio(dev, pin, bt)
    rtlsdr_close(dev)
