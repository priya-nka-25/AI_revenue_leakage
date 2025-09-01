#!/bin/bash

echo "============================================================"
echo "🎯 AI-POWERED REVENUE LEAKAGE DETECTION SYSTEM"
echo "============================================================"
echo "📋 Tech Stack:"
echo "   Frontend: React + TypeScript + Tailwind CSS + Vite"
echo "   Backend:  Python + Flask + SQLite"
echo "   AI:       LLM + Agentic AI + Crew AI Simulation"
echo "============================================================"

echo "🔍 Checking system requirements..."

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python is not installed!"
    echo "📥 Please install Python from: https://www.python.org/downloads/"
    exit 1
fi
echo "✅ Python detected"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed!"
    echo "📥 Please install Node.js from: https://nodejs.org/"
    exit 1
fi
echo "✅ Node.js detected"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed!"
    echo "📥 Please install Node.js from: https://nodejs.org/"
    exit 1
fi
echo "✅ npm detected"

echo "🚀 Starting servers..."

# Start backend in background
echo "🚀 Starting Flask Backend Server..."
cd backend
pip install -r requirements.txt
python run.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Return to project root
cd ..

# Start frontend
echo "🚀 Starting Vite Frontend Server..."
npm install
if [ $? -ne 0 ]; then
    echo "❌ Failed to install frontend dependencies"
    kill $BACKEND_PID
    exit 1
fi

echo "✅ Frontend dependencies installed successfully!"
echo "🌐 Opening application..."
echo "📱 Frontend: http://localhost:5173"
echo "🔧 Backend API: http://localhost:5000/api"

# Trap to kill backend when frontend stops
trap "kill $BACKEND_PID" EXIT

npm run dev