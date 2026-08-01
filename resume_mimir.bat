@echo off
schtasks /Change /TN "Mimir Boot Check" /Enable
echo Mimir's reminder checks are back on.
pause