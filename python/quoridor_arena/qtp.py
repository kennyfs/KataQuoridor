"""QTP (GTP for Quoridor) engine process wrapper.

A QTP response is either "= <text>" (success) or "? <text>" (failure), optionally with a numeric
command id right after the marker, possibly spanning several lines, and terminated by an empty line.
"""
import os
import queue
import subprocess
import threading
import time


class QTPError(Exception):
    """Base class for engine failures that are not a normal "?" response."""


class QTPTimeout(QTPError):
    pass


class QTPCrash(QTPError):
    pass


def parse_response(lines):
    """Parse the lines of one response (without the terminating empty line).

    Returns (ok, text). Lines before the first "="/"?" line are ignored (some engines print
    diagnostics to stdout). Raises ValueError if no marker line is present.
    """
    start = None
    for i, line in enumerate(lines):
        if line.startswith("=") or line.startswith("?"):
            start = i
            break
    if start is None:
        raise ValueError("no '=' or '?' line in response: %r" % (lines,))
    first = lines[start]
    ok = first[0] == "="
    rest = first[1:]
    # Optional numeric id directly after the marker, e.g. "=12 e8".
    j = 0
    while j < len(rest) and rest[j].isdigit():
        j += 1
    rest = rest[j:]
    if rest.startswith(" "):
        rest = rest[1:]
    body = [rest] + list(lines[start + 1:])
    return ok, "\n".join(body).strip()


class ResponseReader:
    """Collects stdout lines and splits them into responses (terminated by an empty line)."""

    def __init__(self):
        self.pending = []
        self.started = False

    def feed(self, line):
        """Feed one line (without newline). Returns a completed list of lines or None."""
        line = line.rstrip("\r")
        if not self.started:
            if line.startswith("=") or line.startswith("?"):
                self.started = True
                self.pending = [line]
            # Ignore stray lines between responses.
            return None
        if line.strip() == "":
            done = self.pending
            self.pending = []
            self.started = False
            return done
        self.pending.append(line)
        return None


class QTPEngine:
    """One engine subprocess. Not thread-safe: each worker owns its own engines."""

    def __init__(self, name, argv, stderr_path=None, cwd=None, default_timeout=60.0):
        self.name = name
        self.argv = list(argv)
        self.stderr_path = stderr_path
        self.cwd = cwd
        self.default_timeout = default_timeout
        self.proc = None
        self._queue = None
        self._stderr_file = None
        self.starts = 0
        self._known = {}

    # -- process management -------------------------------------------------
    def start(self):
        if self.stderr_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.stderr_path)), exist_ok=True)
            self._stderr_file = open(self.stderr_path, "ab")
            self._stderr_file.write(("\n==== start %s: %s\n" % (time.strftime("%F %T"), " ".join(self.argv))).encode())
            self._stderr_file.flush()
        stderr = self._stderr_file if self._stderr_file else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            cwd=self.cwd, text=True, bufsize=1)
        self._queue = queue.Queue()
        t = threading.Thread(target=self._read_stdout, args=(self.proc, self._queue), daemon=True)
        t.start()
        self._known = {}
        self.starts += 1

    @property
    def restarts(self):
        return max(0, self.starts - 1)

    @staticmethod
    def _read_stdout(proc, q):
        reader = ResponseReader()
        try:
            for line in proc.stdout:
                done = reader.feed(line.rstrip("\n"))
                if done is not None:
                    q.put(done)
        except (ValueError, OSError):
            pass
        q.put(None)  # EOF marker

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def close(self, timeout=5.0):
        if self.proc is None:
            return
        if self.alive():
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=timeout)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        self.kill()

    def kill(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except (OSError, ValueError):
                    pass
            self.proc = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    def restart(self):
        self.kill()
        self.start()

    def ensure_running(self):
        if not self.alive():
            self.restart()

    # -- commands -----------------------------------------------------------
    def send(self, command, timeout=None):
        """Send one command. Returns (ok, text). Raises QTPTimeout / QTPCrash.

        After a timeout or crash the process is killed; call ensure_running() before reuse.
        """
        if timeout is None:
            timeout = self.default_timeout
        if not self.alive():
            raise QTPCrash("%s: process not running" % self.name)
        try:
            self.proc.stdin.write(command.strip() + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            self.kill()
            raise QTPCrash("%s: write failed for %r: %s" % (self.name, command, e))
        try:
            lines = self._queue.get(timeout=timeout)
        except queue.Empty:
            self.kill()
            raise QTPTimeout("%s: no response to %r within %.0fs" % (self.name, command, timeout))
        if lines is None:
            code = self.proc.poll() if self.proc else None
            self.kill()
            raise QTPCrash("%s: exited (code %s) while waiting for %r" % (self.name, code, command))
        return parse_response(lines)

    def must(self, command, timeout=None):
        ok, text = self.send(command, timeout)
        if not ok:
            raise QTPError("%s: command %r failed: %s" % (self.name, command, text))
        return text

    def known_command(self, cmd):
        if cmd not in self._known:
            ok, text = self.send("known_command " + cmd)
            self._known[cmd] = ok and text.strip().lower() == "true"
        return self._known[cmd]
