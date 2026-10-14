@echo off
start cmd /k "cd /d C:\Users\Administrator\Documents\carEvaluationModel && python api.py"
timeout /t 2
start index.html