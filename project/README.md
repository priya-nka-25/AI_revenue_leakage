# 🎯 AI-Powered Revenue Leakage Detection System

A comprehensive full-stack application for detecting, analyzing, and resolving revenue leakages using AI technology.

## 🚀 Tech Stack

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Vite** for development server and build tooling
- **Lucide React** for icons
- **Axios** for API communication

### Backend
- **Python 3.8+** with Flask framework
- **SQLite** database for data persistence
- **Flask-CORS** for cross-origin requests
- **Werkzeug** for password hashing

### AI Components (Simulated)
- **LLM Integration** for anomaly detection
- **Agentic AI** for decision making
- **Crew AI** for task orchestration
- **Vector Database** simulation for embeddings

## 📁 Project Structure

```
/
├── src/                          # Frontend React application
│   ├── components/              # React components
│   │   ├── Auth/               # Authentication components
│   │   ├── Dashboard/          # Dashboard components
│   │   ├── Layout/             # Layout components
│   │   └── UI/                 # Reusable UI components
│   ├── context/                # React context providers
│   ├── services/               # API service layer
│   └── main.tsx               # Application entry point
├── backend/                     # Python Flask backend
│   ├── app.py                  # Main Flask application
│   ├── run.py                  # Backend server runner
│   └── requirements.txt        # Python dependencies
└── run_project.py              # Complete project runner
```

## 🏃‍♂️ How to Run

### Option 1: One Command (Recommended)
```bash
python run_project.py
```

### Option 2: Manual Setup
```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python run.py

# Terminal 2 - Frontend
npm install
npm run dev
```

## 🔐 Demo Credentials

| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| Admin | `admin` | `password123` | Full system access, dataset upload, leakage monitoring |
| Finance | `finance` | `password123` | Financial ticket management and resolution |
| IT Support | `it` | `password123` | Technical issue resolution |

## 🎯 Key Features

### 1. **Multi-Role Authentication**
- Secure login system with role-based access control
- Admin, Finance, and IT team dashboards

### 2. **AI-Powered Dataset Processing**
- Upload datasets for Telecom, Healthcare, Banking sectors
- 5-stage AI pipeline: Chunking → Embeddings → Vector DB → LLM Analysis → Root Cause Detection
- Real-time processing visualization

### 3. **Intelligent Leakage Detection**
- AI-powered anomaly detection with severity ranking
- Detailed root cause analysis for each leakage
- Revenue impact calculation and tracking

### 4. **Smart Ticket Management**
- Automated ticket generation with department assignment
- AI-assisted resolution with intelligent suggestions
- Manual resolution workflow with editable AI recommendations

### 5. **Comprehensive Analytics**
- Real-time dashboards with interactive charts
- Severity distribution and sector impact analysis
- Resolution efficiency tracking (AI vs Manual)

## 🌐 API Endpoints

### Authentication
- `POST /api/auth/login` - User authentication

### Dataset Management
- `POST /api/datasets/upload` - Upload new dataset
- `POST /api/datasets/<id>/process` - Process dataset with AI

### Leakage Detection
- `GET /api/leakages` - Get all detected leakages
- `GET /api/leakages/<id>/details` - Get leakage details

### Ticket Management
- `POST /api/tickets/generate` - Generate ticket from leakage
- `GET /api/tickets` - Get tickets by role
- `POST /api/tickets/<id>/resolve` - Resolve ticket

### Analytics
- `GET /api/stats` - Get system statistics and metrics

## 🎨 Design Features

- **Modern Dark Theme** with professional color palette
- **Responsive Design** optimized for all screen sizes
- **Smooth Animations** and micro-interactions
- **Interactive Charts** with real-time data updates
- **Intuitive Navigation** with role-based UI adaptation

## 🔧 Development

### Frontend Development
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
```

### Backend Development
```bash
cd backend
python run.py        # Start Flask development server
```

## 📊 Database Schema

### Tables
- **users**: Authentication and role management
- **datasets**: Uploaded dataset tracking
- **leakages**: AI-detected revenue leakages
- **tickets**: Generated tickets for resolution

## 🤖 AI Pipeline Simulation

The system simulates a complete AI pipeline:
1. **Data Chunking**: Splits large datasets into manageable pieces
2. **Embedding Generation**: Creates vector representations
3. **Vector Database Storage**: Stores embeddings for similarity search
4. **LLM Analysis**: Analyzes patterns for anomaly detection
5. **Crew AI Root Cause**: Determines underlying causes

## 🎯 Usage Workflow

1. **Login** with appropriate role credentials
2. **Upload Dataset** (Admin only) - select sector and upload file
3. **AI Processing** - watch real-time pipeline execution
4. **Review Leakages** - analyze detected issues and revenue impact
5. **Generate Tickets** - create tickets with smart department assignment
6. **Resolve Issues** - use AI assistance or manual resolution
7. **Monitor Analytics** - track system performance and efficiency

## 🌟 Production Ready Features

- Error handling and validation
- Secure authentication with password hashing
- CORS configuration for cross-origin requests
- Responsive design for mobile and desktop
- Real-time data updates and synchronization
- Professional UI/UX suitable for enterprise use



Option 1: One Command (Recommended)
bashpython run_project.py
Option 2: Manual Setup
bash# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python run.py

# Terminal 2 - Frontend (from project root)
npm install
npm run dev