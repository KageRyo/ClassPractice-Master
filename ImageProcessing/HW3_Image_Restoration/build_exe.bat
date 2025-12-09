@echo off
REM Build script for HW3 Image Restoration

echo Building HW3 Image Restoration...

pyinstaller --noconfirm ^
    --onedir ^
    --windowed ^
    --name "HW3_Image_Restoration" ^
    --add-data "test_image;test_image" ^
    --additional-hooks-dir "pyinstaller_hooks" ^
    main.py

echo Build complete!
pause
