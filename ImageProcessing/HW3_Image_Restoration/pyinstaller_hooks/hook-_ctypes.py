# PyInstaller hook for _ctypes module
# This hook ensures proper bundling of ctypes dependencies

from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = collect_dynamic_libs('_ctypes')
