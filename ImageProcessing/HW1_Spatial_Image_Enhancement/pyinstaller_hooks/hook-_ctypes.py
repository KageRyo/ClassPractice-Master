"""Custom PyInstaller hook for _ctypes to avoid compatibility issues on Python 3.13."""

from PyInstaller.utils.hooks import collect_dynamic_libs

try:
    binaries = collect_dynamic_libs('_ctypes')
except Exception:
    binaries = []

hiddenimports = []
