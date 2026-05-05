@echo off
echo ========================================
echo   启动 RAG 智能问答系统
echo ========================================
echo.

cd /d "%~dp0"

echo 正在检查依赖...
pip install -r requirements.txt

echo.
echo 启动 Streamlit 应用...
streamlit run app.py

pause
