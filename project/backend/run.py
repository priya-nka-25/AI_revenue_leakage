#!/usr/bin/env python3
"""
Revenue Leakage Detection System Backend
Run this script to start the Flask development server
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, init_enhanced_db  # Fixed: import correct function name

if __name__ == '__main__':
    print("🚀 Starting Revenue Leakage Detection System Backend...")
    print("📊 Initializing database...")
    init_enhanced_db()  # Fixed: use correct function name
    print("✅ Database initialized successfully!")
    print("🌐 Starting Flask server on http://localhost:5000")
    print("📝 API Documentation:")
    print("   - POST /api/auth/login")
    print("   - POST /api/datasets/upload")
    print("   - POST /api/datasets/<id>/process")
    print("   - GET  /api/leakages")
    print("   - GET  /api/leakages/<id>/details")
    print("   - POST /api/tickets/generate")
    print("   - GET  /api/tickets")
    print("   - POST /api/tickets/<id>/resolve")
    print("   - GET  /api/stats")
    print("\n" + "="*50)
    
    app.run(debug=True, port=5000, host='0.0.0.0')