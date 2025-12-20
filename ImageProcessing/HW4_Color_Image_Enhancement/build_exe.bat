@echo off
REM Build executable for HW4 Color Image Enhancement
REM Made by Chien-Hsun Chang (614410073)

echo Building HW4_Color_Image_Enhancement executable...

REM Activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Install pyinstaller if not installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Build the executable
pyinstaller --noconfirm ^
    --onedir ^
    --windowed ^
    --name "HW4_Color_Image_Enhancement" ^
    --add-data "src;src" ^
    --add-data "test_image;test_image" ^
    --additional-hooks-dir=pyinstaller_hooks ^
    --hidden-import=PIL ^
    --hidden-import=PIL._imagingtk ^
    --hidden-import=PIL._tkinter_finder ^
    --hidden-import=numpy ^
    --hidden-import=matplotlib ^
    --hidden-import=pydantic ^
    main.py

echo.
echo Build complete! Check the dist folder for the executable.
pause
