from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import sysconfig
import time

import numpy as np


HERE = Path(__file__).resolve().parent
SRC = HERE / "trail_echo_avx.cpp"
LIB = HERE / ".build" / "libtrail_echo_avx.dylib"


def _compile() -> None:
    LIB.parent.mkdir(parents=True, exist_ok=True)
    if LIB.exists() and LIB.stat().st_mtime >= SRC.stat().st_mtime:
        return
    lock = LIB.with_suffix(".lock")
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if LIB.exists() and LIB.stat().st_mtime >= SRC.stat().st_mtime:
                return
            time.sleep(0.1)
    cxx = sysconfig.get_config_var("CXX") or "clang++"
    tmp = LIB.with_name(f"{LIB.name}.{os.getpid()}.tmp")
    cmd = [
        cxx,
        "-std=c++17",
        "-O3",
        "-march=native",
        "-ffast-math",
        "-shared",
        "-fPIC",
        str(SRC),
        "-o",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp.replace(LIB)
    finally:
        if tmp.exists():
            tmp.unlink()
        if lock.exists():
            lock.unlink()


def _load() -> ctypes.CDLL:
    _compile()
    lib = ctypes.CDLL(str(LIB))
    lib.range_doppler_peak_scores_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.range_doppler_peak_scores_f32.restype = None
    return lib


_LIB: ctypes.CDLL | None = None


def range_doppler_peak_scores_avx(
    y_block: np.ndarray,
    shifted_codes: np.ndarray,
    window: np.ndarray,
    window_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    global _LIB
    if _LIB is None:
        _LIB = _load()
    y = np.ascontiguousarray(y_block, dtype=np.complex64)
    shifted = np.ascontiguousarray(shifted_codes, dtype=np.complex64)
    win = np.ascontiguousarray(window, dtype=np.float32)
    n_codes, n_ch, code_len = y.shape
    n_tx, n_range, shifted_code_len = shifted.shape
    if shifted_code_len != code_len:
        raise ValueError(f"code length mismatch: {shifted_code_len} != {code_len}")
    power = np.empty(n_range, dtype=np.float32)
    peak_bin = np.empty(n_range, dtype=np.int32)
    _LIB.range_doppler_peak_scores_f32(
        y.view(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(n_codes),
        int(n_ch),
        int(code_len),
        shifted.view(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(n_tx),
        int(n_range),
        win.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        float(window_norm),
        power.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        peak_bin.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    return power, peak_bin
