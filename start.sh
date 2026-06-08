#!/bin/bash
# DrugCLIP Web 应用启动脚本

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         🚀 启动 DrugCLIP Web 应用                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")/web_app/backend"

echo "📊 系统信息:"
echo "  - Python: $(python3 --version 2>&1)"
echo "  - Flask: $(python3 -c 'import flask; print(flask.__version__)' 2>/dev/null || echo '未安装')"
echo ""

echo "🌐 Web 应用启动中..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📱 打开浏览器访问: http://localhost:5000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "按 Ctrl+C 停止应用"
echo ""

# 启动 Flask 应用
python3 app.py
