import errno
import fcntl
import os
import select
import signal
import time


SIG = {}
for n, v in signal.__dict__.items():
    if n.startswith('SIG') and n.isalnum() and isinstance(v, int):
        SIG[n] = v
        SIG[n[3:]] = v
        SIG[v] = n


class AsyncSignal:
    def __init__(self, signals=None):
        super(AsyncSignal, self).__init__()

        self.olds = {}

        if signals is None:
            signals = ['SIGINT']

        for s in signals:
            try:
                sig = int(s)
            except ValueError:
                sig = SIG[s]

            assert 0 < sig < 256 and sig in SIG

            old = signal.signal(sig, self._handler)
            if old is None:
                old = signal.SIG_DFL
            self.olds[sig] = old

        if self.olds:
            self.pid = os.getpid()
            self.rd, self.wr = os.pipe()
            self._tune(self.rd)
            self._tune(self.wr)

    def close(self):
        if self.olds:
            for sig, old in self.olds.items():
                signal.signal(sig, old)
            os.close(self.rd)
            os.close(self.wr)
        self.olds = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _tune(self, fd):
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
        fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.fcntl(fd, fcntl.F_GETFD) | fcntl.FD_CLOEXEC)

    def _handler(self, signum, frame):
        if os.getpid() != self.pid:
            os.kill(self.pid, signum)
        else:
            try:
                os.write(self.wr, chr(signum).encode())
            except ValueError:
                pass

    def wait(self, t=None):
        if t and t > 0:
            till = time.monotonic() + t

        while 1:
            try:
                select.select((self.rd,), (), (), t)
                break
            except select.error as e:
                if e[0] == errno.EINTR:
                    if t and t > 0:
                        t = till - time.monotonic()
                        if t <= 0:
                            break
                    continue
                if e[0] == errno.EBADF:
                    return
                raise

        try:
            return SIG[ord(os.read(self.rd, 1))]
        except:
            return
