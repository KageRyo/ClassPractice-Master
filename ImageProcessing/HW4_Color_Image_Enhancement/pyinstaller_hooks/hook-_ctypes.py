# Hook for _ctypes to handle PyInstaller bundling
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('_ctypes')
