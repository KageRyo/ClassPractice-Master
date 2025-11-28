@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
	echo Python executable not found in PATH.
	exit /b 1
)

python -m PyInstaller --clean --noconfirm HW2_Image_Sharpening.spec
if errorlevel 1 (
	echo PyInstaller build failed.
	exit /b %errorlevel%
)

echo Build succeeded. Executable available in dist\HW2_Image_Sharpening.exe
endlocal
pause
