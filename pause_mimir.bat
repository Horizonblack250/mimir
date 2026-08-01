@echo off
schtasks /Change /TN "Mimir Boot Check" /Disable
echo Mimir's reminder checks are now paused.
pause