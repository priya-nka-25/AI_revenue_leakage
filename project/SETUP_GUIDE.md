# 🤖 Ollama Setup Guide for Revenue Leakage Detection System

## 📋 **Prerequisites**

1. **Python 3.8+** - [Download](https://python.org/downloads/)
2. **Node.js 16+** - [Download](https://nodejs.org/)
3. **Ollama** - [Download](https://ollama.ai/download)
4. **Minimum 8GB RAM** (16GB recommended for larger models)
5. **10GB free disk space** (for models)

---

## 🔧 **Step 1: Install Ollama**

### **Windows:**
```powershell
# Download and install from: https://ollama.ai/download
# Or use winget:
winget install Ollama.Ollama
```

### **macOS:**
```bash
# Download and install from: https://ollama.ai/download
# Or use Homebrew:
brew install ollama
```

### **Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

---

## 🚀 **Step 2: Start Ollama Server**

```bash
# Start Ollama server (required to be running)
ollama serve
```

**Keep this terminal open!** Ollama server must be running for the AI system to work.

---

## 📦 **Step 3: Install AI Models**

```bash
# Install embedding model (for vector search)
ollama pull nomic-embed-text

# Install LLM model (for analysis and chat)
ollama pull mistral:7b

# Verify models are installed
ollama list
```

### **Alternative Models (Optional):**

**For Better Performance:**
```bash
# Larger, more capable models (requires more RAM)
ollama pull llama2:13b          # 13B parameter model
ollama pull codellama:13b       # Code-focused model
ollama pull mxbai-embed-large   # Better embeddings (1024 dim)
```

**For Lower Resource Usage:**
```bash
# Smaller, faster models (less RAM)
ollama pull llama2:7b           # Standard 7B model
ollama pull neural-chat:7b      # Optimized for chat
ollama pull all-minilm:l6-v2    # Lightweight embeddings
```

---

## 🎯 **Step 4: Project Setup**

```powershell
# Navigate to project directory
cd project

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install Python dependencies
cd backend
pip install -r requirements.txt

# Go back to project root
cd ..

# Install frontend dependencies
npm install

# Create environment configuration
copy backend\.env.example backend\.env
```

---

## ⚙️ **Step 5: Configure Models**

Edit `backend/.env`:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=mistral:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Performance Settings
EMBEDDING_DIMENSION=768
EMBEDDING_BATCH_SIZE=16
```

### **Model Recommendations:**

| Use Case | LLM Model | Embedding Model | RAM Required |
|----------|-----------|-----------------|--------------|
| **Development/Testing** | `mistral:7b` | `nomic-embed-text` | 8GB |
| **Production** | `llama2:13b` | `mxbai-embed-large` | 16GB |
| **Low Resource** | `neural-chat:7b` | `all-minilm:l6-v2` | 6GB |

---

## 🚀 **Step 6: Start the System**

```powershell
# Method 1: Automated (Recommended)
python run_project.py

# Method 2: Manual (Two terminals)
# Terminal 1 - Backend:
cd backend
python run.py

# Terminal 2 - Frontend:
npm run dev
```

---

## 🔍 **Step 7: Verify Setup**

1. **Check Ollama Status:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Check Backend Health:**
   ```bash
   curl http://localhost:5000/api/health
   ```

3. **Access Frontend:**
   - Open: http://localhost:5173
   - Login: `admin` / `password123`

---

## 🎯 **Usage Workflow**

1. **Login** with demo credentials
2. **Upload Customer Dataset** (CSV format)
3. **Process with Ollama AI** - Watch real-time pipeline
4. **Review AI-Detected Leakages** 
5. **Generate Tickets** with Ollama insights
6. **Chat with RAG Assistant** using Ollama embeddings

---

## 📊 **Expected Customer Dataset Format**

```csv
customer_id,customer_type,account_age_days,plan_type,plan_price,activation_date
CUST_001,prepaid,41,basic,199,2024-01-15
CUST_002,corporate,245,premium,799,2023-06-12
CUST_003,postpaid,368,standard,499,2022-11-08
```

---

## 🛠 **Troubleshooting**

### **"Connection refused" Error:**
```bash
# Ensure Ollama is running
ollama serve

# Check if models are installed
ollama list

# Test Ollama connection
curl http://localhost:11434/api/tags
```

### **"Model not found" Error:**
```bash
# Pull required models
ollama pull mistral:7b
ollama pull nomic-embed-text

# Verify installation
ollama list
```

### **High Memory Usage:**
```bash
# Use smaller models
ollama pull neural-chat:7b        # Instead of mistral:7b
ollama pull all-minilm:l6-v2      # Instead of nomic-embed-text

# Update backend/.env:
OLLAMA_LLM_MODEL=neural-chat:7b
OLLAMA_EMBEDDING_MODEL=all-minilm:l6-v2
EMBEDDING_DIMENSION=384
```

### **Slow Performance:**
```bash
# Reduce batch size in backend/.env:
EMBEDDING_BATCH_SIZE=8
MAX_CHUNKS_PER_DATASET=200

# Or use faster models:
ollama pull neural-chat:7b
```

---

## 🎯 **Model Performance Comparison**

| Model | Size | Speed | Quality | RAM | Use Case |
|-------|------|-------|---------|-----|----------|
| `neural-chat:7b` | 4GB | ⚡⚡⚡ | ⭐⭐⭐ | 6GB | Development |
| `mistral:7b` | 4GB | ⚡⚡ | ⭐⭐⭐⭐ | 8GB | Balanced |
| `llama2:13b` | 7GB | ⚡ | ⭐⭐⭐⭐⭐ | 16GB | Production |
| `codellama:13b` | 7GB | ⚡ | ⭐⭐⭐⭐⭐ | 16GB | Code Analysis |

**Embedding Models:**
- `all-minilm:l6-v2` - Fast, lightweight (384 dim)
- `nomic-embed-text` - Balanced performance (768 dim)  
- `mxbai-embed-large` - Best quality (1024 dim)

---

## 🚀 **Advanced Configuration**

### **Custom Model Setup:**
```bash
# Create custom model with specific settings
ollama create my-mistral --file=Modelfile

# Example Modelfile:
FROM mistral:7b
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER max_tokens 2000
```

### **GPU Acceleration (Optional):**
```bash
# Ensure CUDA is available
nvidia-smi

# Ollama will automatically use GPU if available
# Check GPU usage during model execution
```

---

## 📈 **Performance Monitoring**

Monitor your system during operation:

```bash
# Check CPU/Memory usage
htop

# Check GPU usage (if available)  
nvidia-smi

# Check disk usage
df -h

# Monitor Ollama logs
tail -f ~/.ollama/logs/server.log
```

---

## ✅ **Success Indicators**

You've successfully set up the system when you see:

1. ✅ **Ollama running**: `ollama list` shows your models
2. ✅ **Backend healthy**: Health endpoint returns Ollama status
3. ✅ **Frontend accessible**: Login page loads at localhost:5173
4. ✅ **AI Pipeline working**: Dataset processing completes successfully
5. ✅ **RAG Chat functional**: Chatbot responds with context

---

## 🎯 **Next Steps**

1. **Test with sample data**: Upload a small CSV file
2. **Monitor performance**: Check RAM/CPU usage  
3. **Experiment with models**: Try different LLM/embedding combinations
4. **Scale up**: Move to larger models for production use
5. **Customize prompts**: Modify AI agents for your specific use case

Your Ollama-powered AI system is now ready for customer revenue leakage detection! 🚀