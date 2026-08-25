#!/bin/bash
# =========================================================
# ⚡ سكربت التثبيت والتشغيل السريع لنظام برق على سيرفر هوستنجر (Hostinger VPS)
# =========================================================

echo "======================================================"
echo "   ⚡ بدء تثبيت خادم نظام برق (Bareq Server) على هوستنجر"
echo "======================================================"

# 1. تحديث حزم النظام
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv nginx git curl

# 2. إعداد البيئة
PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. إنشاء خدمة Systemd للتشغيل التلقائي في الخلفية
SERVICE_FILE="/etc/systemd/system/bareq.service"
sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Bareq Server Daemon
After=network.target

[Service]
User=root
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python server.py
Restart=always
RestartSec=5
Environment="PORT=8500"
Environment="HOST=0.0.0.0"

[Install]
WantedBy=multi-user.target
EOF

# 4. تفعيل الخدمة
sudo systemctl daemon-reload
sudo systemctl enable bareq
sudo systemctl restart bareq

echo ""
echo "======================================================"
echo "   ✅ تم تشغيل نظام برق بنجاح على المنفذ 8500!"
echo "   حالة الخدمة: sudo systemctl status bareq"
echo "======================================================"
