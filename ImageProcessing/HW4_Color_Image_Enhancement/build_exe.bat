@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
	echo Python executable not found in PATH.
	exit /b 1
)

python -m PyInstaller --clean --noconfirm HW4_Color_Image_Enhancement.spec
if errorlevel 1 (
	echo PyInstaller build failed.
	exit /b %errorlevel%
)

echo Build succeeded. Executable available in dist\HW4_Color_Image_Enhancement.exe
endlocal
pause
