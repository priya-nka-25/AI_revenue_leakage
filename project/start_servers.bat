@echo off
echo ============================================================
echo 🎯 AI-POWERED REVENUE LEAKAGE DETECTION SYSTEM
echo ============================================================
echo 📋 Tech Stack:
echo    Frontend: React + TypeScript + Tailwind CSS + Vite
echo    Backend:  Python + Flask + SQLite
echo    AI:       LLM + Agentic AI + Crew AI Simulation
echo ============================================================

echo 🔍 Checking system requirements...

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH!
    echo 📥 Please install Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✅ Python detected

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed or not in PATH!
    echo 📥 Please install Node.js from: https://nodejs.org/
    pause
    exit /b 1
)
echo ✅ Node.js detected

REM Check npm
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm is not installed or not in PATH!
    echo 📥 Please install Node.js from: https://nodejs.org/
    pause
    exit /b 1
)
echo ✅ npm detected

echo 🚀 Starting servers...

REM Start backend in background
echo 🚀 Starting Flask Backend Server...
start /B cmd /c "cd backend && pip install -r requirements.txt && python run.py"

REM Wait for backend to start
echo ⏳ Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Start frontend
echo 🚀 Starting Vite Frontend Server...
npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install frontend dependencies
    pause
    exit /b 1
)

echo ✅ Frontend dependencies installed successfully!
echo 🌐 Opening application...
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:5000/api

npm run dev