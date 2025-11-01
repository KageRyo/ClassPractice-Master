# -*- mode: python ; coding: utf-8 -*-

import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from PyInstaller.utils.hooks import collect_data_files

additional_datas = [('test_image', 'test_image')]
additional_datas += collect_data_files('PIL', include_py_files=False)

additional_hiddenimports = [
    'matplotlib.backends.backend_tkagg',
    'PIL._tkinter_finder',
    'pydantic_core._pydantic_core',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=additional_datas,
    hiddenimports=additional_hiddenimports,
    hookspath=['pyinstaller_hooks'],
    hooksconfig={'matplotlib': {'backends': ['TkAgg']}},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'pytest',
        'sphinx',
        'sphinxcontrib',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_qt6agg',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HW1_Image_Enhancement',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
