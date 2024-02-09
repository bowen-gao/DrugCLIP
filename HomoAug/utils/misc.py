import hashlib
import os
import signal
import subprocess
import sys
import time


def time2str(time_used):
    gaps = [
        ('days', 86400000),
        ('h', 3600000),
        ('min', 60000),
        ('s', 1000),
        ('ms', 1)
    ]
    time_used *= 1000
    time_str = []
    for unit, gap in gaps:
        val = time_used // gap
        if val > 0:
            time_str.append('{}{}'.format(int(val), unit))
            time_used -= val * gap
    if len(time_str) == 0:
        time_str.append('0ms')
    return ' '.join(time_str)


def get_date():
    return time.strftime('%Y-%m-%d', time.localtime(time.time()))

def get_time(t=None):
    if t is None:
        t = time.time()
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))


def hash_seed(*items, width=32):
    # width: range of seed: [0, 2**width)
    sha = hashlib.sha256()
    for item in items:
        sha.update(str(item).encode('utf-8'))
    return int(sha.hexdigest()[23:23+width//4], 16)


def execute(command, verbose=False, cmd_out_off=False):
    if verbose:
        print(command)
        print()
    if cmd_out_off:
        popen = silent(subprocess.Popen)
    else:
        popen = subprocess.Popen
    p = popen(command, shell=True)
    try:
        p.wait()
    except KeyboardInterrupt:
        try:
            os.kill(p.pid, signal.SIGINT)
        except OSError:
            pass
        p.wait()


def silent(func):
    """
    Make function silent.
    Useful for closing the output if you don't find a switch.
    """
    def wrap_silent(*args, **kwargs):
        fd = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(os.devnull, os.O_WRONLY)

        res = func(*args, **kwargs)

        os.close(1)
        os.dup(fd)
        os.close(fd)

        return res
    return wrap_silent


def with_time(func, pretty_time=False):
    """
    Usage:

    1. as a function decorator
    ``` python
    @with_time
    def func(...):
        ...
    result, cost_in_seconds = func(...)
    ```

    2. directly apply
    ``` python
    result, cost_in_seconds = with_time(func)(...)
    ```
    """
    def wrap_time(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        time_cost = time.time() - start
        if pretty_time:
            time_cost = time2str(time_cost)
        return res, time_cost
    return wrap_time

