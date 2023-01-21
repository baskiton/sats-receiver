import signal

from unittest import TestCase

from sats_receiver.async_signal import AsyncSignal


class TestAsyncSignal(TestCase):
    def test_wait_sigint(self):
        signame = 'SIGINT'
        with AsyncSignal() as asig:
            signal.raise_signal(signal.SIGINT)

            self.assertEqual(signame, asig.wait(1))

    def test_sig_invalid(self):
        signame = 'SIGAAA'

        with AsyncSignal([signame]) as asig:
            with self.assertRaises(AttributeError):
                asig.wait(1)
