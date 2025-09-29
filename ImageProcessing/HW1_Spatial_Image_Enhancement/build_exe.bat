@echo off
pyinstaller --onefile --add-data "test_image;test_image" --name HW1_Image_Enhancement main.py
pause