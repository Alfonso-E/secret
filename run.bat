@echo off
REM Wrapper so you don't need to type the full Python path every time.
REM Usage:  run.bat check_connection.py
REM         run.bat live_bot.py --mock-account
REM         run.bat live_bot.py
"C:\Users\Lenovo\anaconda3\envs\Python311\python.exe" %*
