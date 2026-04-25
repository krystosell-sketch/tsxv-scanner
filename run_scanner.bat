@echo off
cd /d "C:\Users\CoWork\Desktop\Claude\Stocks scanner CSE TSXV"
if not exist logs mkdir logs
"C:\Users\CoWork\AppData\Local\Python\bin\python.exe" -m src.main --save --explain --alert >> "C:\Users\CoWork\Desktop\Claude\Stocks scanner CSE TSXV\logs\scheduler.log" 2>&1
