#!/usr/bin/env python3
"""
AI-powered Revenue Leakage Detection System
Complete Project Runner with Real AI Integration

This script starts both the backend Flask server and frontend Vite development server.
"""

import subprocess
import sys
import os
import time
import threading
import platform
from pathlib import Path

def check_python():
    """Check if Python is installed and version is 3.8+"""
    try:
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            return False, f"{version.major}.{version.minor}"
        return True, f"{version.major}.{version.minor}"
    except:
        return False, "unknown"

def check_node_npm():
    """Check if Node.js and npm are installed"""
    try:
        node_result = subprocess.run(["node", "--version"], check=True, capture_output=True, text=True)
        npm_result = subprocess.run(["npm", "--version"], check=True, capture_output=True, text=True)
        return True, node_result.stdout.strip(), npm_result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, None, None

def check_ai_config():
    """Check if AI configuration exists"""
    env_file = Path(__file__).parent / "backend" / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            return 'OPENAI_API_KEY=' in content and not 'your_openai_api_key_here' in content
    return False

def run_backend():
    """Start the Flask backend server with real AI"""
    print("🚀 Starting Flask Backend Server with Real AI...")
    backend_dir = Path(__file__).parent / "backend"
    
    try:
        os.chdir(backend_dir)
        
        # Install Python dependencies
        print("📦 Installing Python AI dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Backend AI dependencies installed successfully!")
        
        # Check AI configuration
        if not check_ai_config():
            print("⚠️  WARNING: OpenAI API key not configured!")
            print("📝 Please edit backend/.env and add your OpenAI API key")
            print("🔑 Get your key from: https://platform.openai.com/api-keys")
            print("📄 Example: OPENAI_API_KEY=sk-your-actual-key-here")
            print("🤖 AI features will use fallback mode without API key")
            print("")
        
        # Start Flask server
        subprocess.run([sys.executable, "run.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend server failed: {e}")
        print("💡 Try running manually: cd backend && pip install -r requirements.txt && python run.py")

def run_frontend():
    """Start the Vite frontend development server"""
    print("🚀 Starting Vite Frontend Server...")
    project_dir = Path(__file__).parent
    
    try:
        os.chdir(project_dir)
        
        # Check if Node.js and npm are available
        node_ok, node_version, npm_version = check_node_npm()
        if not node_ok:
            print("❌ Node.js and npm are required but not found!")
            print("📥 Please install Node.js from: https://nodejs.org/")
            print("🔄 After installation, restart this script.")
            return
        
        print(f"✅ Node.js {node_version} and npm {npm_version} detected")
        
        # Install Node.js dependencies
        print("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], check=True)
        print("✅ Frontend dependencies installed successfully!")
        
        # Start Vite dev server
        print("🌐 Starting development server...")
        subprocess.run(["npm", "run", "dev"], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend server failed: {e}")
        print("💡 Try running manually: npm install && npm run dev")

def main():
    """Main function to run both servers"""
    print("=" * 70)
    print("🎯 AI-POWERED REVENUE LEAKAGE DETECTION SYSTEM")
    print("=" * 70)
    print("📋 Real AI Tech Stack:")
    print("   Frontend: React + TypeScript + Tailwind CSS + Vite")
    print("   Backend:  Python + Flask + SQLite")
    print("   AI Stack: OpenAI LLM + Agentic AI + Crew AI + Vector DB")
    print("   Vector:   FAISS/Pinecone/Weaviate + RAG Chatbot")
    print("=" * 70)
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    
    # Check Python
    python_ok, python_version = check_python()
    if not python_ok:
        print(f"❌ Python 3.8+ is required! Current: {python_version}")
        print("📥 Please install Python 3.8+ from: https://www.python.org/downloads/")
        return
    print(f"✅ Python {python_version} detected")
    
    # Check Node.js and npm
    node_ok, node_version, npm_version = check_node_npm()
    if not node_ok:
        print("❌ Node.js and npm are required!")
        print("📥 Please install Node.js from: https://nodejs.org/")
        print("🔄 After installation, restart this script.")
        return
    print(f"✅ Node.js {node_version} and npm {npm_version} detected")
    
    # Check AI configuration
    if check_ai_config():
        print("✅ OpenAI API key configured - Real AI features enabled")
    else:
        print("⚠️  OpenAI API key not configured - Using fallback mode")
        print("🔑 For full AI features, configure backend/.env with your OpenAI API key")
    
    print("🚀 Starting servers...")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Wait for backend to start
    print("⏳ Waiting for backend AI initialization...")
    time.sleep(5)
    
    # Start frontend (this will block)
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        print("✅ AI-powered system stopped successfully!")

if __name__ == "__main__":
    main()