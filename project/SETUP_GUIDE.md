# 🎯 AI-Powered Revenue Leakage Detection System - Complete Setup Guide

## 📋 **System Requirements**

### **Required Software:**
1. **Python 3.8+** - [Download from python.org](https://www.python.org/downloads/)
2. **Node.js 16+** - [Download from nodejs.org](https://nodejs.org/)
3. **npm** (comes with Node.js)

### **Check if installed:**
```bash
python --version    # Should show 3.8+
node --version      # Should show 16+
npm --version       # Should show 8+
```

---

## 🔑 **AI Configuration (Required)**

### **Step 1: Get OpenAI API Key**
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create account and generate API key
3. Copy the API key (starts with `sk-`)

### **Step 2: Configure Environment**
```bash
# Copy the example environment file
cp backend/.env.example backend/.env

# Edit backend/.env and add your API key:
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
VECTOR_DB_TYPE=faiss
```

---

## 🚀 **Quick Start (One Command)**

### **Windows:**
```bash
python run_project.py
```

### **Mac/Linux:**
```bash
python3 run_project.py
```

This command will:
- ✅ Install all Python AI dependencies (OpenAI, CrewAI, LangChain, FAISS)
- ✅ Start Flask backend with real AI integration on `http://localhost:5000`
- ✅ Install all Node.js frontend dependencies
- ✅ Start Vite frontend server on `http://localhost:5173`

---

## 🔧 **Manual Setup (Alternative)**

### **Step 1: Backend Setup**
```bash
cd backend
pip install -r requirements.txt
python run.py
```

### **Step 2: Frontend Setup (New Terminal)**
```bash
npm install
npm run dev
```

---

## 🌐 **Access the Application**

1. **Frontend:** Open `http://localhost:5173` in your browser
2. **Backend API:** Available at `http://localhost:5000/api`
3. **Health Check:** `http://localhost:5000/api/health`

---

## 🔐 **Demo Login Credentials**

| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| **Admin** | `admin` | `password123` | Full system access, dataset upload, AI monitoring, RAG chatbot |
| **Finance** | `finance` | `password123` | Financial ticket management and resolution |
| **IT Support** | `it` | `password123` | Technical issue resolution and system tickets |

---

## 🤖 **Real AI Pipeline Workflow**

### **1. Dataset Upload & Processing**
- Admin uploads CSV/Excel/JSON dataset
- System performs real AI processing:
  1. **📊 Chunking** - Intelligent text splitting
  2. **🧠 Embeddings** - Vector generation using OpenAI
  3. **💾 Vector DB** - Storage in FAISS/Pinecone/Weaviate
  4. **🤖 LLM Analysis** - Pattern recognition using GPT
  5. **🔍 Crew AI** - Multi-agent root cause analysis

### **2. Leakage Detection (NO Auto-Tickets)**
- AI detects revenue leakages with severity ranking
- Stores results with confidence scores
- **NO tickets are generated automatically**
- Status remains "detected" until manual action

### **3. Manual Ticket Generation**
- Admin reviews AI-detected leakages
- Clicks "Analyze & Generate Ticket" for specific leakage
- Views detailed AI analysis and smart department assignment
- **Must click "OK" to confirm** ticket generation
- Only then ticket is created and sent to assigned department

### **4. Ticket Resolution**
- **AI Resolution:** Uses real LLM to generate step-by-step solutions
- **Manual Resolution:** Team edits AI suggestions or writes custom solutions
- Updates leakage status to "resolved" after completion

### **5. RAG Chatbot**
- Uses vector database embeddings for context
- Answers questions about leakages, tickets, datasets
- Provides intelligent insights based on system data

---

## 📁 **Complete Project Architecture**

### **Frontend (React + TypeScript + Tailwind)**
```
src/
├── components/
│   ├── Auth/
│   │   └── LoginPage.tsx                 # Multi-role authentication
│   ├── Dashboard/
│   │   ├── AdminDashboard.tsx            # Admin control panel with RAG chatbot
│   │   ├── TeamDashboard.tsx             # Finance/IT team interface
│   │   ├── DatasetUpload.tsx             # Real AI processing upload
│   │   ├── LeakageTable.tsx              # AI detection results display
│   │   ├── TicketDetailsModal.tsx        # Two-step ticket generation with confirmation
│   │   ├── TicketCard.tsx                # Individual ticket management
│   │   ├── ChartsSection.tsx             # Real-time analytics and visualization
│   │   ├── StatsCards.tsx                # KPI metrics display
│   │   ├── ManualResolutionModal.tsx     # Manual ticket resolution interface
│   │   └── RAGChatbot.tsx                # AI assistant chatbot with vector search
│   ├── Layout/
│   │   └── Header.tsx                    # Navigation header with role indicators
│   └── UI/
│       └── SuccessModal.tsx              # Success notifications and confirmations
├── context/
│   ├── AuthContext.tsx                   # Authentication state management
│   └── DataContext.tsx                   # Data fetching and state management
├── services/
│   └── api.ts                            # Axios API client with all real endpoints
└── main.tsx                              # Application entry point
```

### **Backend (Python + Flask + Real AI)**
```
backend/
├── app.py                                # Main Flask application with real AI integration
├── run.py                                # Server startup script
├── requirements.txt                      # Python AI dependencies
├── .env.example                          # Environment configuration template
└── .env                                  # Your actual API keys (create this)
```

---

## 🔗 **Real AI API Endpoints**

### **Authentication**
- `POST /api/auth/login` - User authentication with role-based access

### **Dataset Management**
- `POST /api/datasets/upload` - Upload dataset with sector selection
- `POST /api/datasets/<id>/process` - **Real LLM + Agentic AI + Crew AI Pipeline**

### **Leakage Detection (AI-Powered)**
- `GET /api/leakages` - Get all AI-detected leakages with confidence scores
- `GET /api/leakages/<id>/details` - Detailed AI analysis with smart assignment

### **Ticket Management (Manual Confirmation)**
- `POST /api/tickets/generate` - Generate ticket **ONLY after admin clicks OK**
- `GET /api/tickets` - Get role-based tickets with AI suggestions
- `POST /api/tickets/<id>/resolve` - Resolve tickets (Real AI/Manual methods)

### **Analytics & Reporting**
- `GET /api/stats` - Real-time system statistics and metrics

### **RAG Chatbot (Vector Database)**
- `POST /api/chat` - Ask questions using vector database context
- `GET /api/chat/history` - Get chat conversation history

### **System Health**
- `GET /api/health` - Check AI configuration and system status

---

## 🎯 **Complete Workflow**

1. **Login** → Choose role (Admin/Finance/IT)
2. **Upload Dataset** → Select sector and upload real data file
3. **AI Processing** → Real 5-stage AI pipeline (LLM + Crew AI)
4. **Review Leakages** → AI shows detected issues with confidence scores
5. **Manual Ticket Generation** → Admin reviews and **clicks OK to confirm**
6. **Team Resolution** → Finance/IT teams resolve with real AI assistance
7. **RAG Chatbot** → Ask intelligent questions about system data
8. **Analytics** → Monitor real-time performance and AI efficiency

---

## 🤖 **AI Technologies Used**

### **LLM Integration:**
- **OpenAI GPT-3.5/4** for natural language processing
- **Google Gemini** (optional alternative)

### **Agentic AI:**
- **Multi-agent orchestration** for complex decision making
- **Intelligent task delegation** between specialized agents

### **Crew AI:**
- **Finance Agent** - Revenue analysis specialist
- **IT Agent** - Technical systems expert
- **Data Science Agent** - Statistical analysis expert
- **Root Cause Agent** - Synthesis and correlation specialist
- **Resolution Agent** - Solution architecture expert

### **Vector Database:**
- **FAISS** (default) - Facebook AI Similarity Search
- **Pinecone** (optional) - Managed vector database
- **Weaviate** (optional) - Open-source vector database

### **RAG Chatbot:**
- **Vector similarity search** for context retrieval
- **LLM-powered responses** with system data integration
- **Conversation history** tracking and analysis

---

## 🛠 **Troubleshooting**

### **Common Issues:**

1. **"OpenAI API key not configured" error:**
   - Copy `backend/.env.example` to `backend/.env`
   - Add your OpenAI API key: `OPENAI_API_KEY=sk-your-key-here`
   - Restart the backend server

2. **"AI processing failed" error:**
   - Check your OpenAI API key is valid and has credits
   - Verify internet connection for API calls
   - Check backend logs for specific error details

3. **"npm not found" error:**
   - Install Node.js from [nodejs.org](https://nodejs.org/)
   - Restart terminal after installation

4. **Python module errors:**
   - Ensure Python 3.8+ is installed
   - Run: `pip install -r backend/requirements.txt`
   - Check for any dependency conflicts

5. **Port already in use:**
   - Backend: Change port in `backend/app.py` (line with `app.run`)
   - Frontend: Change port in `vite.config.ts`

6. **Vector database errors:**
   - Default FAISS should work without additional setup
   - For Pinecone/Weaviate, configure API keys in `.env`

---

## 💡 **RAG Chatbot Usage**

**Example Questions:**
- "Show me leakage summary for banking sector"
- "What's the current ticket resolution rate?"
- "How many critical issues are pending?"
- "Analyze AI vs manual resolution efficiency"
- "What are the top revenue leakage causes?"

---

## 🎨 **UI Features (Dark Theme)**

- **Modern Dark Theme** with professional purple/blue gradients
- **Responsive Design** optimized for all screen sizes
- **Real-time Updates** after every AI operation
- **Interactive Charts** with dynamic data visualization
- **Smooth Animations** and micro-interactions
- **Role-based Navigation** with intuitive workflows

---

## 🔧 **Development Commands**

### **Frontend Development**
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
```

### **Backend Development**
```bash
cd backend
python run.py        # Start Flask development server with AI
```

---

## 📊 **Database Schema**

### **Tables:**
- **users** - Authentication and role management
- **datasets** - Uploaded dataset tracking with AI processing status
- **leakages** - AI-detected revenue leakages (status: detected → ticket-generated → resolved)
- **tickets** - Manually confirmed tickets with AI suggestions
- **chat_history** - RAG chatbot conversation tracking

---

## 🌟 **Production Ready Features**

✅ Real LLM + Agentic AI + Crew AI integration  
✅ Vector database with embeddings storage  
✅ Manual ticket confirmation workflow  
✅ RAG chatbot with intelligent context retrieval  
✅ Multi-role authentication system  
✅ Real-time AI processing visualization  
✅ Comprehensive analytics and reporting  
✅ Professional dark theme UI  
✅ Error handling and validation  
✅ Secure API endpoints with proper CORS  
✅ Responsive design for all devices  

The system is now ready for mentor demonstrations with real AI capabilities!