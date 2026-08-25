@echo off
title Bareq System Server
color 0B
echo ==================================================
echo         تشغيل خادم تطبيق برق (Bareq Server)
echo ==================================================
echo.
echo 1. يتم الآن تشغيل السيرفر المحلي...
echo 2. سيتم فتح لوحة التحكم في المتصفح تلقائيا على البورت 8500.
echo.
start "" "http://127.0.0.1:8500"
python server.py
pause
