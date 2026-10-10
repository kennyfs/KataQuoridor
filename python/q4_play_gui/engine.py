"""The QTP process of the Q4 play GUI: QTPEngine plus the streamed "info" lines of an analysis.

Same design as Duel's StreamingEngine (python/play_gui/game.py): `info` lines reach `on_info` while `streaming` is set
and the response header "=" has been read; everything else is a response. Not thread-safe: the controller holds its
engine lock around every command.
"""
import queue
import sys
import os

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from quoridor_arena.qtp import QTPCrash, QTPEngine, QTPError, QTPTimeout, ResponseReader, parse_response  # noqa: E402


class StreamingQTP(QTPEngine):
    streaming = False
    on_info = None

    def _read_stdout(self, proc, q):
        reader = ResponseReader()
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if self.streaming and reader.started and line.startswith("info "):
                    cb = self.on_info
                    if cb is not None:
                        cb(line)
                    continue
                done = reader.feed(line)
                if done is not None:
                    q.put(done)
        except (ValueError, OSError):
            pass
        q.put(None)

    def write(self, command):
        """Send a line without waiting for its response (the response is read with read_response)."""
        if not self.alive():
            raise QTPCrash("%s: process not running" % self.name)
        try:
            self.proc.stdin.write(command.strip() + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            self.kill()
            raise QTPCrash("%s: write failed for %r: %s" % (self.name, command, e))

    def read_response(self, what, timeout):
        try:
            lines = self._queue.get(timeout=timeout)
        except queue.Empty:
            self.kill()
            raise QTPTimeout("%s: no response to %r within %.0fs" % (self.name, what, timeout))
        if lines is None:
            code = self.proc.poll() if self.proc else None
            self.kill()
            raise QTPCrash("%s: exited (code %s) while waiting for %r" % (self.name, code, what))
        return parse_response(lines)


__all__ = ["StreamingQTP", "QTPError", "QTPCrash", "QTPTimeout"]
