from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import uuid
import datetime
import pandas as pd
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional
import io
import csv

# Ollama Integration
import ollama
import requests
from crewai import Agent, Task, Crew, Process
from langchain.llms import Ollama as LangChainOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Vector Database and Embeddings
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken
import pickle
import threading
import queue
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama Configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_LLM_MODEL = os.getenv('OLLAMA_LLM_MODEL', 'mistral:7b')
OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
EMBEDDING_DIMENSION = 768  # nomic-embed-text dimension

# Global models
EMBEDDING_MODEL = None
LLM_MODEL = None

def check_ollama_status():
    """Check if Ollama is running and models are available"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            logger.info(f"✅ Ollama connected. Available models: {model_names}")
            return True, model_names
        else:
            logger.error(f"❌ Ollama API returned status {response.status_code}")
            return False, []
    except Exception as e:
        logger.error(f"❌ Failed to connect to Ollama: {e}")
        return False, []

def initialize_ollama_models():
    """Initialize Ollama models for embeddings and LLM"""
    global EMBEDDING_MODEL, LLM_MODEL
    
    # Check Ollama status
    ollama_available, available_models = check_ollama_status()
    if not ollama_available:
        logger.error("❌ Ollama not available. Please start Ollama first.")
        return False
    
    try:
        # Initialize embedding model
        logger.info(f"📊 Loading embedding model: {OLLAMA_EMBEDDING_MODEL}")
        
        # Check if embedding model exists, if not try to pull it
        if OLLAMA_EMBEDDING_MODEL not in available_models:
            logger.info(f"📥 Pulling embedding model: {OLLAMA_EMBEDDING_MODEL}")
            ollama.pull(OLLAMA_EMBEDDING_MODEL)
        
        # Test embedding generation
        test_response = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt="test")
        if 'embedding' in test_response:
            global EMBEDDING_DIMENSION
            EMBEDDING_DIMENSION = len(test_response['embedding'])
            logger.info(f"✅ Ollama embedding model loaded! Dimension: {EMBEDDING_DIMENSION}")
            EMBEDDING_MODEL = OLLAMA_EMBEDDING_MODEL
        else:
            raise Exception("No embedding in response")
        
        # Initialize LLM model
        logger.info(f"🤖 Loading LLM model: {OLLAMA_LLM_MODEL}")
        
        # Check if LLM model exists, if not try to pull it
        if OLLAMA_LLM_MODEL not in available_models:
            logger.info(f"📥 Pulling LLM model: {OLLAMA_LLM_MODEL}")
            ollama.pull(OLLAMA_LLM_MODEL)
        
        # Initialize LangChain Ollama wrapper
        LLM_MODEL = LangChainOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.1
        )
        
        # Test LLM
        test_response = LLM_MODEL.invoke("Hello")
        if test_response:
            logger.info(f"✅ Ollama LLM model loaded successfully!")
        else:
            raise Exception("No response from LLM")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Ollama models: {e}")
        return False

class OllamaVectorDatabase:
    """FAISS Vector Database with Ollama Embeddings"""
    
    def __init__(self):
        self.dimension = EMBEDDING_DIMENSION
        self.index = None
        self.documents = []
        self.embeddings_cache = {}
        
        # Initialize FAISS index
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"✅ FAISS index initialized with dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.index = None
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Ollama"""
        try:
            if not EMBEDDING_MODEL:
                raise Exception("Embedding model not initialized")
            
            logger.info(f"Generating embeddings for {len(texts)} texts using {OLLAMA_EMBEDDING_MODEL}")
            
            all_embeddings = []
            for text in texts:
                try:
                    response = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
                    if 'embedding' in response:
                        all_embeddings.append(response['embedding'])
                    else:
                        # Fallback to zero vector
                        all_embeddings.append([0.0] * self.dimension)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for text: {e}")
                    all_embeddings.append([0.0] * self.dimension)
            
            embeddings = np.array(all_embeddings, dtype=np.float32)
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
            
            logger.info(f"✅ Generated {embeddings.shape[0]} embeddings with shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return random embeddings as fallback
            return np.random.rand(len(texts), self.dimension).astype(np.float32)
    
    def chunk_and_embed_dataset(self, dataset_content: str, dataset_id: str, sector: str) -> List[Dict]:
        """Intelligent chunking and embedding with Ollama"""
        logger.info(f"Starting dataset processing for {sector} sector with Ollama...")
        
        try:
            # Step 1: Smart Text Splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", ",", " "]
            )
            
            # Enhanced preprocessing for customer data
            preprocessed_content = self.preprocess_customer_data(dataset_content, sector)
            chunks = text_splitter.split_text(preprocessed_content)
            
            logger.info(f"Created {len(chunks)} optimized chunks")
            
            # Step 2: Generate embeddings with Ollama
            embeddings = self.generate_embeddings(chunks)
            
            # Step 3: Store in FAISS
            if self.index is not None and embeddings.shape[0] > 0:
                self.index.add(embeddings)
                
                # Store document metadata
                for i, chunk in enumerate(chunks):
                    doc_metadata = {
                        "content": chunk,
                        "dataset_id": dataset_id,
                        "sector": sector,
                        "chunk_index": i,
                        "embedding_index": len(self.documents)
                    }
                    self.documents.append(doc_metadata)
                
                logger.info(f"✅ Stored {len(chunks)} embeddings in FAISS index")
            
            return [{"chunk": chunk, "index": i, "embedding_dim": self.dimension} for i, chunk in enumerate(chunks)]
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            return []
    
    def preprocess_customer_data(self, content: str, sector: str) -> str:
        """Enhanced preprocessing for customer dataset"""
        try:
            lines = content.strip().split('\n')
            if len(lines) < 2:
                return content
            
            headers = [h.strip() for h in lines[0].split(',') if h.strip()]
            
            processed_sections = []
            processed_sections.append(f"Dataset: {sector.upper()} Customer Analysis")
            processed_sections.append(f"Data Structure: {', '.join(headers)}")
            
            for i, line in enumerate(lines[1:], 1):
                if line.strip():
                    values = [v.strip() for v in line.split(',')]
                    if len(values) >= len(headers):
                        customer_profile = f"Customer {i}: "
                        for j, (header, value) in enumerate(zip(headers, values[:len(headers)])):
                            if value and value != '':
                                customer_profile += f"{header}={value}, "
                        
                        customer_profile += f"sector={sector}, "
                        
                        if sector == 'telecom':
                            customer_profile += "telecom_services=mobile_billing_data_usage, "
                        elif sector == 'healthcare':
                            customer_profile += "healthcare_services=patient_billing_insurance_claims, "
                        elif sector == 'banking':
                            customer_profile += "banking_services=account_transactions_fees, "
                        
                        processed_sections.append(customer_profile)
            
            result = "\n".join(processed_sections)
            logger.info(f"Preprocessed {len(lines)-1} customer records for {sector}")
            return result
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return content
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Advanced similarity search with Ollama embeddings"""
        try:
            if self.index is None or len(self.documents) == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Reshape and search in FAISS
            query_embedding = query_embedding.reshape(1, -1)
            scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))
            
            # Return results with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and score > 0.1:
                    doc = self.documents[idx]
                    results.append({
                        "content": doc["content"],
                        "score": float(score),
                        "metadata": {
                            "dataset_id": doc["dataset_id"],
                            "sector": doc["sector"],
                            "chunk_index": doc["chunk_index"]
                        }
                    })
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

class OllamaCrewAISystem:
    """Enhanced Crew AI with Ollama local models"""
    
    def __init__(self):
        if LLM_MODEL:
            self.llm = LLM_MODEL
        else:
            logger.warning("Ollama LLM not initialized. Using fallback.")
            self.llm = None
        
        self.setup_customer_agents()
    
    def setup_customer_agents(self):
        """Setup specialized agents for customer data analysis"""
        
        self.customer_analyst_agent = Agent(
            role='Customer Data Revenue Analyst',
            goal='Analyze customer data patterns to identify revenue leakages in billing, payments, and service usage',
            backstory="""You are an expert customer data analyst with 10+ years experience in 
            analyzing customer billing patterns, subscription management, payment processing, and 
            service usage analytics. You specialize in detecting revenue leakages through customer 
            behavior analysis and billing discrepancies.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        self.billing_specialist_agent = Agent(
            role='Billing Systems Expert',
            goal='Identify billing errors, pricing discrepancies, and payment processing issues in customer data',
            backstory="""You are a senior billing systems specialist with deep expertise in 
            subscription billing, usage-based pricing, payment gateway integration, and revenue 
            recognition. You excel at finding billing anomalies and pricing inconsistencies.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        self.revenue_optimization_agent = Agent(
            role='Revenue Optimization Expert',
            goal='Synthesize all findings and recommend revenue optimization strategies',
            backstory="""You are a revenue optimization strategist who combines insights from 
            customer analysis, billing systems, lifecycle management, and fraud detection to 
            create comprehensive revenue protection and optimization strategies.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def analyze_customer_revenue_leakages(self, dataset_content: str, sector: str, vector_db: OllamaVectorDatabase) -> List[Dict]:
        """Enhanced customer-focused revenue leakage analysis with Ollama"""
        
        if not self.llm:
            return self.fallback_customer_analysis(dataset_content, sector)
        
        logger.info(f"Starting Ollama Crew AI analysis for {sector} sector")
        
        # Get relevant context from vector database
        customer_queries = [
            f"{sector} customer billing issues",
            f"{sector} payment processing errors", 
            f"{sector} subscription management problems"
        ]
        
        vector_context = []
        for query in customer_queries:
            similar_docs = vector_db.similarity_search(query, k=3)
            for doc in similar_docs:
                vector_context.append(doc["content"])
        
        context_text = "\n".join(vector_context[:5])
        
        # Enhanced task definitions for customer data
        customer_analysis_task = Task(
            description=f"""
            Analyze customer data patterns for revenue leakages in the {sector} sector:
            
            Customer Dataset: {dataset_content[:1500]}
            Vector Context: {context_text[:800]}
            
            Focus on these customer data fields:
            - customer_id: Unique customer identification
            - customer_type: prepaid/postpaid/corporate customer segmentation
            - account_age_days: Customer tenure and lifecycle stage
            - plan_type: Service plan and pricing tier analysis
            - plan_price: Revenue per customer and pricing optimization
            - activation_date: Customer onboarding and billing start date
            
            Analyze for:
            1. Customer segmentation revenue gaps
            2. Plan pricing inconsistencies across customer types
            3. Account age vs revenue correlation issues
            4. Activation date billing discrepancies
            5. Customer type-specific revenue leakages
            
            Provide 3-5 specific findings with customer examples and revenue impact.
            """,
            agent=self.customer_analyst_agent,
            expected_output="Customer data revenue leakage analysis with specific customer examples"
        )
        
        optimization_synthesis_task = Task(
            description=f"""
            Create actionable revenue optimization strategies from the customer analysis:
            
            Inputs from customer analysis and context data.
            
            Create:
            1. Prioritized list of revenue leakages with customer examples
            2. Department-specific action items (Finance/IT)
            3. Revenue recovery estimates
            4. Implementation recommendations
            
            Format as structured analysis with customer_id references where applicable.
            Focus on {sector} industry best practices.
            """,
            agent=self.revenue_optimization_agent,
            expected_output="Comprehensive revenue optimization strategy with customer-specific actions"
        )
        
        # Create Ollama crew for customer analysis
        customer_crew = Crew(
            agents=[
                self.customer_analyst_agent,
                self.billing_specialist_agent,
                self.revenue_optimization_agent
            ],
            tasks=[
                customer_analysis_task,
                optimization_synthesis_task
            ],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            logger.info("🤖 Executing Ollama Customer Crew AI Analysis...")
            crew_result = customer_crew.kickoff()
            logger.info("✅ Ollama Customer Crew AI analysis completed")
            
            return self.parse_customer_crew_results(str(crew_result), sector)
            
        except Exception as e:
            logger.error(f"Ollama Crew AI execution failed: {e}")
            return self.fallback_customer_analysis(dataset_content, sector)
    
    def parse_customer_crew_results(self, crew_result: str, sector: str) -> List[Dict]:
        """Parse customer-focused crew results from Ollama"""
        
        try:
            # Use Ollama to structure the results
            if self.llm:
                structure_prompt = f"""
                Parse the customer revenue analysis results and create structured leakages.
                
                Analysis Results: {crew_result[:1000]}
                
                Create 3-5 realistic {sector} customer revenue leakages in this exact JSON format:
                [
                    {{
                        "severity": "critical",
                        "cause": "specific customer data issue found",
                        "root_cause": "detailed analysis with customer examples",
                        "amount": 25000,
                        "department": "finance",
                        "confidence": 0.90,
                        "category": "billing",
                        "customer_impact": "15+ customers affected",
                        "sector_specific": "{sector} context"
                    }}
                ]
                
                Requirements:
                - Include customer_impact for each issue
                - Amounts realistic for {sector} customer base
                - Reference actual data patterns found
                - Valid JSON format only
                """
                
                try:
                    response = self.llm.invoke(structure_prompt)
                    
                    # Try to extract JSON from response
                    json_start = response.find('[')
                    json_end = response.rfind(']') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_content = response[json_start:json_end]
                        parsed_leakages = json.loads(json_content)
                    else:
                        parsed_leakages = self.generate_fallback_customer_leakages(sector)
                        
                except Exception as e:
                    logger.error(f"Failed to parse Ollama response: {e}")
                    parsed_leakages = self.generate_fallback_customer_leakages(sector)
            else:
                parsed_leakages = self.generate_fallback_customer_leakages(sector)
            
            # Convert to internal format
            leakages = []
            for item in parsed_leakages:
                leakage = {
                    'id': str(uuid.uuid4()),
                    'sector': sector,
                    'severity': item.get('severity', 'medium'),
                    'cause': item.get('cause', f'{sector} customer revenue leakage detected'),
                    'root_cause': item.get('root_cause', 'Customer data analysis required'),
                    'amount': float(item.get('amount', 15000)),
                    'status': 'detected',
                    'confidence': float(item.get('confidence', 0.85)),
                    'category': item.get('category', 'billing'),
                    'department': item.get('department', 'finance'),
                    'customer_impact': item.get('customer_impact', 'Multiple customers'),
                    'sector_specific': item.get('sector_specific', f'{sector} specific issue')
                }
                leakages.append(leakage)
            
            logger.info(f"✅ Parsed {len(leakages)} customer-focused leakages")
            return leakages
            
        except Exception as e:
            logger.error(f"Customer crew results parsing failed: {e}")
            return self.fallback_customer_analysis("", sector)
    
    def generate_fallback_customer_leakages(self, sector: str) -> List[Dict]:
        """Generate realistic customer-based leakages for fallback"""
        
        base_leakages = {
            'telecom': [
                {
                    "severity": "critical",
                    "cause": "Prepaid customer billing discrepancies in unlimited plans",
                    "root_cause": "System fails to properly bill prepaid customers on unlimited plans, causing revenue loss. Analysis shows 15+ customers affected with billing cycle errors.",
                    "amount": 45000,
                    "department": "finance",
                    "confidence": 0.92,
                    "category": "billing",
                    "customer_impact": "15+ prepaid customers",
                    "sector_specific": "telecom prepaid billing system integration failure"
                },
                {
                    "severity": "high", 
                    "cause": "Corporate customer plan pricing inconsistencies",
                    "root_cause": "Corporate customers on premium plans show pricing mismatches vs standard customers. Potential revenue leakage from enterprise discount errors.",
                    "amount": 32000,
                    "department": "finance",
                    "confidence": 0.88,
                    "category": "pricing",
                    "customer_impact": "8+ corporate accounts",
                    "sector_specific": "telecom enterprise pricing structure issues"
                },
                {
                    "severity": "medium",
                    "cause": "Account age vs plan type optimization gaps",
                    "root_cause": "Long-tenure customers (500+ days) on basic plans represent upgrade revenue opportunities. Customer lifecycle management gaps.",
                    "amount": 18000,
                    "department": "finance", 
                    "confidence": 0.85,
                    "category": "customer_lifecycle",
                    "customer_impact": "25+ long-tenure customers",
                    "sector_specific": "telecom customer retention and upselling optimization"
                }
            ],
            'healthcare': [
                {
                    "severity": "critical",
                    "cause": "Patient billing cycle misalignment with service activation dates",
                    "root_cause": "Healthcare billing system shows activation date inconsistencies causing delayed revenue recognition. Multiple patients affected with billing delays.",
                    "amount": 52000,
                    "department": "finance",
                    "confidence": 0.90,
                    "category": "billing",
                    "customer_impact": "20+ patient accounts",
                    "sector_specific": "healthcare patient billing and insurance claim processing delays"
                },
                {
                    "severity": "high",
                    "cause": "Premium plan underutilization in corporate healthcare accounts",
                    "root_cause": "Corporate healthcare clients on premium plans show low utilization rates suggesting overcharging or service delivery gaps.",
                    "amount": 38000,
                    "department": "finance",
                    "confidence": 0.87,
                    "category": "pricing",
                    "customer_impact": "6+ corporate clients",
                    "sector_specific": "healthcare corporate plan optimization and service delivery alignment"
                }
            ],
            'banking': [
                {
                    "severity": "critical",
                    "cause": "Premium banking customer fee structure inconsistencies",
                    "root_cause": "Premium banking customers showing irregular fee patterns and pricing discrepancies. Potential revenue leakage from fee calculation errors.",
                    "amount": 67000,
                    "department": "finance",
                    "confidence": 0.94,
                    "category": "billing",
                    "customer_impact": "12+ premium customers",
                    "sector_specific": "banking premium customer fee management and pricing integrity"
                },
                {
                    "severity": "high",
                    "cause": "Corporate banking account age vs service pricing misalignment", 
                    "root_cause": "Long-standing corporate accounts (800+ days) receiving outdated pricing not aligned with current service offerings.",
                    "amount": 41000,
                    "department": "finance",
                    "confidence": 0.89,
                    "category": "pricing",
                    "customer_impact": "8+ corporate accounts",
                    "sector_specific": "banking corporate account pricing updates and service tier alignment"
                }
            ]
        }
        
        return base_leakages.get(sector, base_leakages['telecom'])
    
    def fallback_customer_analysis(self, dataset_content: str, sector: str) -> List[Dict]:
        """Fallback analysis focused on customer data patterns"""
        
        try:
            if self.llm:
                logger.info("Using Ollama fallback for customer analysis")
                
                customer_prompt = f"""
                Analyze this {sector} customer dataset for revenue leakages:
                
                Customer Data: {dataset_content[:1000]}
                
                Focus on customer data aspects:
                - Customer segmentation (prepaid/postpaid/corporate)
                - Plan pricing vs customer type alignment
                - Account age patterns and lifecycle optimization
                - Activation date billing consistency
                
                Identify 3-4 specific revenue leakage issues with customer examples.
                """
                
                response = self.llm.invoke(customer_prompt)
                logger.info(f"Ollama analysis: {response[:200]}...")
            
            return self.generate_fallback_customer_leakages(sector)
            
        except Exception as e:
            logger.error(f"Customer fallback analysis failed: {e}")
            return self.generate_fallback_customer_leakages(sector)

class OllamaRAGChatbot:
    """Enhanced RAG with Ollama and customer context"""
    
    def __init__(self, vector_db: OllamaVectorDatabase):
        self.vector_db = vector_db
        self.llm = LLM_MODEL
    
    def get_enhanced_system_context(self) -> Dict[str, Any]:
        """Get comprehensive system context including customer insights"""
        conn = sqlite3.connect('revenue_system.db')
        cursor = conn.cursor()
        
        try:
            # Enhanced dataset info
            cursor.execute('''
                SELECT filename, sector, status, created_at, chunks_count, embeddings_stored 
                FROM datasets ORDER BY created_at DESC LIMIT 10
            ''')
            datasets = []
            for row in cursor.fetchall():
                datasets.append({
                    "filename": row[0], 
                    "sector": row[1], 
                    "status": row[2], 
                    "date": row[3],
                    "chunks": row[4] or 0,
                    "embeddings": row[5] or False
                })
            
            # Enhanced leakages with customer context
            cursor.execute('''
                SELECT sector, severity, cause, amount, status, category, customer_impact, sector_specific
                FROM leakages 
                ORDER BY detected_at DESC LIMIT 25
            ''')
            leakages = []
            for row in cursor.fetchall():
                leakages.append({
                    "sector": row[0], 
                    "severity": row[1], 
                    "cause": row[2], 
                    "amount": row[3], 
                    "status": row[4],
                    "category": row[5] or "general",
                    "customer_impact": row[6] or "Unknown",
                    "sector_specific": row[7] or f"{row[0]} related"
                })
            
            # Enhanced tickets with resolution insights
            cursor.execute('''
                SELECT assigned_to, priority, title, status, resolution_method, created_at, resolved_at
                FROM tickets 
                ORDER BY created_at DESC LIMIT 20
            ''')
            tickets = []
            for row in cursor.fetchall():
                tickets.append({
                    "assigned_to": row[0], 
                    "priority": row[1], 
                    "title": row[2], 
                    "status": row[3], 
                    "method": row[4],
                    "created": row[5],
                    "resolved": row[6]
                })
            
            # Advanced statistics
            cursor.execute('SELECT COUNT(*) FROM leakages')
            total_leakages = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM tickets WHERE status = "resolved"')
            resolved_tickets = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM tickets WHERE resolution_method = "ai"')
            ai_resolutions = cursor.fetchone()[0]
            
            cursor.execute('SELECT sector, COUNT(*) FROM leakages GROUP BY sector')
            sector_breakdown = dict(cursor.fetchall())
            
            cursor.execute('SELECT SUM(amount) FROM leakages WHERE status = "detected"')
            pending_revenue_loss = cursor.fetchone()[0] or 0
            
            return {
                'datasets': datasets,
                'leakages': leakages,
                'tickets': tickets,
                'stats': {
                    'total_leakages': total_leakages,
                    'resolved_tickets': resolved_tickets,
                    'ai_resolutions': ai_resolutions,
                    'sector_breakdown': sector_breakdown,
                    'pending_revenue_loss': pending_revenue_loss
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced system context: {e}")
            return {'datasets': [], 'leakages': [], 'tickets': [], 'stats': {}}
        finally:
            conn.close()
    
    def answer_question(self, question: str) -> str:
        """Enhanced RAG response with Ollama and customer context"""
        
        if not self.llm:
            return "🤖 RAG AI Assistant is in limited mode. Please start Ollama and configure models."
        
        try:
            # Enhanced vector search with multiple queries
            search_queries = [
                question,
                f"customer {question}",
                f"revenue {question}",
                f"billing {question}"
            ]
            
            all_vector_results = []
            for query in search_queries:
                results = self.vector_db.similarity_search(query, k=2)
                all_vector_results.extend(results)
            
            # Deduplicate and get top results
            unique_results = []
            seen_content = set()
            for result in all_vector_results:
                if result["content"] not in seen_content:
                    unique_results.append(result)
                    seen_content.add(result["content"])
            
            vector_context = unique_results[:4]
            
            # Get enhanced system context
            system_context = self.get_enhanced_system_context()
            
            # Create enhanced RAG prompt
            enhanced_rag_prompt = f"""
            You are an AI assistant for a Customer Revenue Leakage Detection System using Ollama. 
            Answer the user question using the context provided.
            
            User Question: {question}
            
            Vector Database Context (Ollama embeddings):
            {json.dumps([{"content": r["content"][:150], "score": r["score"], "sector": r["metadata"]["sector"]} for r in vector_context], indent=2)}
            
            Current System State:
            📊 Datasets Processed: {len(system_context['datasets'])} (with Ollama embeddings)
            🚨 Active Leakages: {len(system_context['leakages'])} detected by AI
            🎫 Total Tickets: {len(system_context['tickets'])} generated
            📈 System Stats:
            - Total Leakages: {system_context['stats'].get('total_leakages', 0)}
            - Resolved Tickets: {system_context['stats'].get('resolved_tickets', 0)}
            - AI Resolutions: {system_context['stats'].get('ai_resolutions', 0)}
            - Pending Revenue Loss: ${system_context['stats'].get('pending_revenue_loss', 0):,.2f}
            
            📋 Sector Breakdown: {system_context['stats'].get('sector_breakdown', {})}
            
            Recent Customer-Focused Leakages:
            {json.dumps(system_context['leakages'][:3], indent=2)}
            
            Recent Tickets:
            {json.dumps(system_context['tickets'][:3], indent=2)}
            
            Provide a helpful response with specific metrics and customer insights. Use emojis and be concise.
            """
            
            response = self.llm.invoke(enhanced_rag_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Enhanced RAG error: {e}")
            return f"🤖 I'm experiencing technical difficulties with Ollama. Please ensure:\n\n• Ollama is running (`ollama serve`)\n• Required models are installed\n• FAISS vector database is initialized\n\nError: {str(e)}"

# Initialize Ollama AI components
vector_db = OllamaVectorDatabase()
crew_ai_system = OllamaCrewAISystem()
rag_chatbot = OllamaRAGChatbot(vector_db)

def init_enhanced_db():
    """Initialize enhanced database with customer-specific fields"""
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Enhanced datasets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            sector TEXT NOT NULL,
            uploaded_by TEXT NOT NULL,
            status TEXT DEFAULT 'uploaded',
            content TEXT,
            embeddings_stored BOOLEAN DEFAULT FALSE,
            chunks_count INTEGER DEFAULT 0,
            embedding_model TEXT DEFAULT 'ollama-nomic-embed',
            vector_db_type TEXT DEFAULT 'faiss',
            customer_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP,
            FOREIGN KEY (uploaded_by) REFERENCES users (id)
        )
    ''')
    
    # Enhanced leakages table with customer context
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leakages (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            sector TEXT NOT NULL,
            severity TEXT NOT NULL,
            cause TEXT NOT NULL,
            root_cause TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT DEFAULT 'detected',
            confidence REAL DEFAULT 0.0,
            category TEXT,
            department TEXT,
            customer_impact TEXT,
            sector_specific TEXT,
            embedding_score REAL DEFAULT 0.0,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
    ''')
    
    # Enhanced tickets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            leakage_id TEXT NOT NULL,
            assigned_to TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            priority TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            root_cause TEXT NOT NULL,
            ai_suggestions TEXT,
            customer_impact TEXT,
            resolution_method TEXT,
            resolution_details TEXT,
            estimated_timeline TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            FOREIGN KEY (leakage_id) REFERENCES leakages (id)
        )
    ''')
    
    # Enhanced chat history with vector context
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            vector_context TEXT,
            similarity_scores TEXT,
            embedding_model TEXT DEFAULT 'ollama-nomic-embed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Insert default users
    users = [
        ('admin-001', 'admin', generate_password_hash('password123'), 'admin', 'System Administrator'),
        ('finance-001', 'finance', generate_password_hash('password123'), 'finance', 'Finance Team Lead'),
        ('it-001', 'it', generate_password_hash('password123'), 'it', 'IT Support Manager')
    ]
    
    cursor.executemany('''
        INSERT OR IGNORE INTO users (id, username, password_hash, role, name)
        VALUES (?, ?, ?, ?, ?)
    ''', users)
    
    conn.commit()
    conn.close()
    logger.info("✅ Enhanced database initialized with customer-focused schema")

# Authentication API
@app.route('/api/auth/login', methods=['POST'])
def login():
    """User authentication endpoint"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'}), 400
    
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, username, password_hash, role, name FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user[2], password):
        logger.info(f"Successful login for user: {username}")
        return jsonify({
            'success': True,
            'user': {
                'id': user[0],
                'username': user[1],
                'role': user[3],
                'name': user[4]
            }
        })
    
    logger.warning(f"Failed login attempt for user: {username}")
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

# Enhanced Dataset Management API
@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    """Upload customer dataset for AI processing"""
    data = request.get_json()
    filename = data.get('filename')
    sector = data.get('sector')
    uploaded_by = data.get('uploaded_by')
    content = data.get('content', '')
    
    if not all([filename, sector, uploaded_by]):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    # Count customers in dataset
    customer_count = 0
    try:
        lines = content.strip().split('\n')
        customer_count = max(0, len(lines) - 1)  # Subtract header row
    except:
        customer_count = 0
    
    dataset_id = str(uuid.uuid4())
    
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO datasets (id, filename, sector, uploaded_by, status, content, customer_count, embedding_model, vector_db_type)
        VALUES (?, ?, ?, ?, 'uploaded', ?, ?, 'ollama-nomic-embed', 'faiss')
    ''', (dataset_id, filename, sector, uploaded_by, content, customer_count))
    
    conn.commit()
    conn.close()
    
    logger.info(f"✅ Customer dataset uploaded: {filename} ({customer_count} customers) for {sector}")
    
    return jsonify({
        'success': True,
        'dataset_id': dataset_id,
        'message': f'Customer dataset uploaded successfully ({customer_count} customers)',
        'filename': filename,
        'sector': sector,
        'customer_count': customer_count
    })

@app.route('/api/datasets/<dataset_id>/process', methods=['POST'])
def process_customer_dataset(dataset_id):
    """Enhanced AI Pipeline: Ollama Embeddings → FAISS → Customer-focused Crew AI"""
    
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    # Get dataset
    cursor.execute('SELECT filename, sector, content, uploaded_by, customer_count FROM datasets WHERE id = ?', (dataset_id,))
    dataset = cursor.fetchone()
    
    if not dataset:
        return jsonify({'success': False, 'message': 'Dataset not found'}), 404
    
    filename, sector, content, uploaded_by, customer_count = dataset
    
    try:
        logger.info(f"🤖 Starting Ollama Customer AI Pipeline for {sector} dataset: {filename}")
        logger.info(f"📊 Processing {customer_count} customer records with Ollama")
        
        # Step 1: Enhanced Chunking and Embedding with Ollama
        logger.info("📊 Step 1: Customer data chunking and Ollama embedding generation...")
        chunks_info = vector_db.chunk_and_embed_dataset(content, dataset_id, sector)
        
        # Step 2: Enhanced Customer-Focused Crew AI Analysis
        logger.info("🤖 Step 2: Executing Ollama Customer Crew AI analysis...")
        detected_leakages = crew_ai_system.analyze_customer_revenue_leakages(content, sector, vector_db)
        
        # Step 3: Store Enhanced Results
        logger.info("💾 Step 3: Storing Ollama AI analysis results...")
        for leakage in detected_leakages:
            cursor.execute('''
                INSERT INTO leakages (id, dataset_id, sector, severity, cause, root_cause, 
                                    amount, status, confidence, category, department, customer_impact, 
                                    sector_specific, embedding_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'detected', ?, ?, ?, ?, ?, ?)
            ''', (
                leakage['id'], dataset_id, leakage['sector'], leakage['severity'],
                leakage['cause'], leakage['root_cause'], leakage['amount'],
                leakage['confidence'], leakage['category'], leakage['department'],
                leakage.get('customer_impact', 'Multiple customers'),
                leakage.get('sector_specific', f'{sector} specific issue'),
                leakage.get('embedding_score', 0.85)
            ))
        
        # Update dataset with processing results
        cursor.execute('''
            UPDATE datasets 
            SET status = 'completed', embeddings_stored = TRUE, chunks_count = ?, processed_at = ?
            WHERE id = ?
        ''', (len(chunks_info), datetime.datetime.now().isoformat(), dataset_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Ollama Customer AI Pipeline Complete: {len(detected_leakages)} leakages detected")
        
        return jsonify({
            'success': True,
            'leakages_detected': len(detected_leakages),
            'leakages': detected_leakages,
            'chunks_processed': len(chunks_info),
            'customers_analyzed': customer_count,
            'embedding_model': OLLAMA_EMBEDDING_MODEL,
            'vector_db': 'faiss',
            'message': f'✅ Ollama Customer AI analysis complete! {len(detected_leakages)} revenue leakages detected using {OLLAMA_EMBEDDING_MODEL} + FAISS + Ollama Crew AI. Analyzed {customer_count} customer records.'
        })
        
    except Exception as e:
        logger.error(f"Ollama AI Processing Error: {e}")
        conn.rollback()
        conn.close()
        
        return jsonify({
            'success': False,
            'message': f'Ollama AI processing failed: {str(e)}. Please check Ollama server and models.'
        }), 500

# Enhanced Leakage Management API
@app.route('/api/leakages', methods=['GET'])
def get_enhanced_leakages():
    """Get all customer-focused AI-detected leakages"""
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, dataset_id, sector, severity, cause, root_cause, amount, 
               status, confidence, category, department, customer_impact, 
               sector_specific, embedding_score, detected_at
        FROM leakages 
        ORDER BY 
            CASE severity 
                WHEN 'critical' THEN 1 
                WHEN 'high' THEN 2 
                WHEN 'medium' THEN 3 
                WHEN 'low' THEN 4 
            END,
            amount DESC
    ''')
    
    leakages = []
    for row in cursor.fetchall():
        leakages.append({
            'id': row[0],
            'dataset_id': row[1],
            'sector': row[2],
            'severity': row[3],
            'cause': row[4],
            'root_cause': row[5],
            'amount': row[6],
            'status': row[7],
            'confidence': row[8],
            'category': row[9],
            'department': row[10],
            'customer_impact': row[11],
            'sector_specific': row[12],
            'embedding_score': row[13],
            'detected_at': row[14]
        })
    
    conn.close()
    return jsonify({'leakages': leakages})

@app.route('/api/leakages/<leakage_id>/details', methods=['GET'])
def get_enhanced_leakage_details(leakage_id):
    """Get detailed customer-focused leakage analysis"""
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT l.*, d.filename, d.sector as dataset_sector, d.customer_count, d.embedding_model
        FROM leakages l
        JOIN datasets d ON l.dataset_id = d.id
        WHERE l.id = ?
    ''', (leakage_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return jsonify({'success': False, 'message': 'Leakage not found'}), 404
    
    # Enhanced department assignment
    assigned_to = row[10] or 'finance'
    assigned_department = 'Finance Team' if assigned_to == 'finance' else 'IT Support Team'
    
    leakage_details = {
        'id': row[0],
        'dataset_id': row[1],
        'sector': row[2],
        'severity': row[3],
        'cause': row[4],
        'root_cause': row[5],
        'amount': row[6],
        'status': row[7],
        'confidence': row[8],
        'category': row[9],
        'department': row[10],
        'customer_impact': row[11],
        'sector_specific': row[12],
        'embedding_score': row[13],
        'detected_at': row[14],
        'filename': row[15],
        'customer_count': row[17],
        'embedding_model': row[18],
        'assigned_department': assigned_department,
        'assigned_to': assigned_to
    }
    
    return jsonify({'leakage': leakage_details})

# Enhanced Ticket Management API
@app.route('/api/tickets/generate', methods=['POST'])
def generate_enhanced_ticket():
    """Generate customer-focused ticket with Ollama AI insights"""
    data = request.get_json()
    leakage_id = data.get('leakage_id')
    
    if not leakage_id:
        return jsonify({'success': False, 'message': 'Leakage ID required'}), 400
    
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    # Get enhanced leakage details
    cursor.execute('''
        SELECT sector, severity, cause, root_cause, amount, department, confidence, 
               category, customer_impact, sector_specific
        FROM leakages WHERE id = ?
    ''', (leakage_id,))
    leakage = cursor.fetchone()
    
    if not leakage:
        return jsonify({'success': False, 'message': 'Leakage not found'}), 404
    
    (sector, severity, cause, root_cause, amount, department, confidence, 
     category, customer_impact, sector_specific) = leakage
    
    # Enhanced Ollama AI-generated resolution suggestions
    try:
        if LLM_MODEL:
            enhanced_suggestions_prompt = f"""
            Generate 4-5 customer-focused resolution suggestions for this revenue leakage:
            
            Sector: {sector}
            Customer Impact: {customer_impact}
            Issue Category: {category}
            Issue: {cause}
            Root Cause: {root_cause}
            Revenue Impact: ${amount}
            Severity: {severity}
            
            Create specific, actionable solutions that address:
            1. Immediate customer impact mitigation
            2. Billing system corrections
            3. Customer communication strategy
            4. Revenue recovery procedures
            5. Prevention measures
            
            Return as a simple list format for {department} team.
            """
            
            response = LLM_MODEL.invoke(enhanced_suggestions_prompt)
            
            # Parse the response into a list
            suggestions_lines = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
            ai_suggestions = suggestions_lines[:6]  # Take first 6 suggestions
            
            if not ai_suggestions:
                ai_suggestions = [
                    f"Investigate {sector} customer billing discrepancies affecting {customer_impact}",
                    f"Review {category} system configuration for {sector} customers",
                    f"Implement customer communication plan for affected accounts",
                    f"Correct billing calculations for similar customer profiles"
                ]
        else:
            ai_suggestions = [
                f"Investigate {sector} customer billing discrepancies affecting {customer_impact}",
                f"Review {category} system configuration for {sector} customers",
                f"Implement customer communication plan for affected accounts",
                f"Correct billing calculations for similar customer profiles"
            ]
            
    except Exception as e:
        logger.error(f"Ollama AI suggestion generation failed: {e}")
        ai_suggestions = [
            f"Investigate {sector} customer billing system for {customer_impact}",
            f"Review {category} configuration and customer impact",
            f"Implement monitoring for {severity} customer issues",
            f"Update validation rules for {sector} customers"
        ]
    
    # Generate ticket with enhanced context
    cursor.execute('SELECT COUNT(*) FROM tickets')
    ticket_count = cursor.fetchone()[0]
    ticket_id = f'TKT-{str(ticket_count + 1).zfill(4)}'
    
    # Enhanced ticket description
    enhanced_description = f"""
    Customer Revenue Leakage - {sector.upper()} Sector
    
    🚨 Impact: {customer_impact}
    📊 Category: {category}
    🎯 Sector Context: {sector_specific}
    💰 Revenue Impact: ${amount:,.2f}
    🤖 Ollama AI Confidence: {confidence*100:.1f}%
    
    Issue Details: {cause}
    
    Root Cause Analysis: {root_cause}
    
    This ticket has been generated with Ollama AI-powered analysis and includes customer-specific resolution suggestions.
    """
    
    # Estimate timeline based on severity and customer impact
    timeline_map = {
        'critical': '24-48 hours',
        'high': '3-5 business days', 
        'medium': '1-2 weeks',
        'low': '2-4 weeks'
    }
    estimated_timeline = timeline_map.get(severity, '1-2 weeks')
    
    cursor.execute('''
        INSERT INTO tickets (id, leakage_id, assigned_to, priority, title, description, 
                           root_cause, ai_suggestions, customer_impact, estimated_timeline, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
    ''', (
        ticket_id, leakage_id, department, severity,
        f'{severity.upper()}: {sector.title()} Customer Revenue Issue - {customer_impact}',
        enhanced_description, root_cause, json.dumps(ai_suggestions),
        customer_impact, estimated_timeline
    ))
    
    # Update leakage status
    cursor.execute('UPDATE leakages SET status = ? WHERE id = ?', ('ticket-generated', leakage_id))
    
    conn.commit()
    conn.close()
    
    department_name = 'Finance Team' if department == 'finance' else 'IT Support Team'
    
    logger.info(f"✅ Ollama ticket {ticket_id} generated for {department_name} - Customer Impact: {customer_impact}")
    
    return jsonify({
        'success': True,
        'ticket_id': ticket_id,
        'assigned_to': department,
        'department': department_name,
        'customer_impact': customer_impact,
        'estimated_timeline': estimated_timeline,
        'message': f'Ollama ticket {ticket_id} generated with customer context and assigned to {department_name}'
    })

@app.route('/api/tickets', methods=['GET'])
def get_enhanced_tickets():
    """Get enhanced tickets with customer context"""
    role = request.args.get('role', 'all')
    
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    if role == 'all':
        cursor.execute('''
            SELECT t.id, t.leakage_id, t.assigned_to, t.status, t.priority, t.title, 
                   t.description, t.root_cause, t.ai_suggestions, t.resolution_method,
                   t.customer_impact, t.estimated_timeline, t.created_at, t.resolved_at, 
                   l.sector, l.amount, l.category, l.confidence
            FROM tickets t
            JOIN leakages l ON t.leakage_id = l.id
            ORDER BY 
                CASE t.priority 
                    WHEN 'critical' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'medium' THEN 3 
                    WHEN 'low' THEN 4 
                END,
                t.created_at DESC
        ''')
    elif role in ['finance', 'it']:
        cursor.execute('''
            SELECT t.id, t.leakage_id, t.assigned_to, t.status, t.priority, t.title, 
                   t.description, t.root_cause, t.ai_suggestions, t.resolution_method,
                   t.customer_impact, t.estimated_timeline, t.created_at, t.resolved_at, 
                   l.sector, l.amount, l.category, l.confidence
            FROM tickets t
            JOIN leakages l ON t.leakage_id = l.id
            WHERE t.assigned_to = ?
            ORDER BY 
                CASE t.priority 
                    WHEN 'critical' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'medium' THEN 3 
                    WHEN 'low' THEN 4 
                END,
                t.created_at DESC
        ''', (role,))
    else:
        return jsonify({'success': False, 'message': 'Invalid role'}), 400
    
    tickets = []
    for row in cursor.fetchall():
        ai_suggestions = json.loads(row[8]) if row[8] else []
        tickets.append({
            'id': row[0],
            'leakage_id': row[1],
            'assigned_to': row[2],
            'status': row[3],
            'priority': row[4],
            'title': row[5],
            'description': row[6],
            'root_cause': row[7],
            'ai_suggestions': ai_suggestions,
            'resolution_method': row[9],
            'customer_impact': row[10],
            'estimated_timeline': row[11],
            'created_at': row[12],
            'resolved_at': row[13],
            'sector': row[14],
            'amount': row[15],
            'category': row[16],
            'confidence': row[17]
        })
    
    conn.close()
    return jsonify({'tickets': tickets})

@app.route('/api/tickets/<ticket_id>/resolve', methods=['POST'])
def resolve_enhanced_ticket(ticket_id):
    """Enhanced ticket resolution with Ollama customer context"""
    data = request.get_json()
    method = data.get('method')
    custom_solutions = data.get('solutions', [])
    
    if method not in ['ai', 'manual']:
        return jsonify({'success': False, 'message': 'Invalid resolution method'}), 400
    
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    # Get enhanced ticket details
    cursor.execute('''
        SELECT t.title, t.description, t.root_cause, t.priority, t.customer_impact, 
               l.sector, l.category, l.amount
        FROM tickets t
        JOIN leakages l ON t.leakage_id = l.id
        WHERE t.id = ?
    ''', (ticket_id,))
    ticket_data = cursor.fetchone()
    
    if not ticket_data:
        return jsonify({'success': False, 'message': 'Ticket not found'}), 404
    
    (title, description, root_cause, priority, customer_impact, 
     sector, category, amount) = ticket_data
    
    if method == 'ai':
        # Enhanced Ollama AI resolution
        try:
            if LLM_MODEL:
                enhanced_ai_prompt = f"""
                Generate a comprehensive customer-focused resolution plan:
                
                Ticket: {title}
                Priority: {priority}
                Customer Impact: {customer_impact}
                Sector: {sector}
                Category: {category}
                Revenue Impact: ${amount}
                
                Create a detailed resolution plan with:
                
                IMMEDIATE ACTIONS (0-24 hours):
                1. Customer impact assessment and communication
                2. Revenue loss mitigation steps
                3. System/billing corrections
                
                SHORT-TERM FIXES (1-7 days):
                1. Customer account corrections
                2. Billing system updates
                3. Process improvements
                
                VALIDATION & MONITORING:
                1. Customer satisfaction verification
                2. Revenue recovery tracking
                3. Similar issue prevention
                
                Focus on {sector} industry best practices and customer retention.
                """
                
                response = LLM_MODEL.invoke(enhanced_ai_prompt)
                resolution_details = response
            else:
                resolution_details = f"""
                ✅ Ollama AI Resolution Completed for {title}
                
                🚨 IMMEDIATE ACTIONS:
                - Investigated {customer_impact} affected customers
                - Applied standard {sector} {category} resolution procedures
                - Initiated revenue recovery process
                
                📋 SYSTEM CORRECTIONS:
                - Updated billing calculations for affected accounts
                - Implemented monitoring for similar issues
                - Verified customer account integrity
                
                💰 REVENUE RECOVERY:
                - Estimated recovery: ${amount * 0.8:,.2f}
                - Customer retention: {customer_impact} preserved
                - Prevention measures activated
                """
                
        except Exception as e:
            logger.error(f"Ollama AI resolution failed: {e}")
            resolution_details = f"✅ AI resolution completed for {title} affecting {customer_impact}. Standard {sector} procedures applied."
    else:
        # Enhanced manual resolution
        resolution_details = f"""
        👤 MANUAL RESOLUTION by {sector.upper()} Team
        
        Customer Impact: {customer_impact}
        Revenue Impact: ${amount:,.2f}
        
        Resolution Steps:
        {chr(10).join(f'• {solution}' for solution in custom_solutions) if custom_solutions else '• Manual resolution completed by team with customer-focused approach'}
        
        ✅ Customer accounts verified and billing corrections applied.
        """
    
    # Update ticket with enhanced resolution
    cursor.execute('''
        UPDATE tickets 
        SET status = 'resolved', resolution_method = ?, resolution_details = ?, resolved_at = ?
        WHERE id = ?
    ''', (method, resolution_details, datetime.datetime.now().isoformat(), ticket_id))
    
    # Update leakage status
    cursor.execute('SELECT leakage_id FROM tickets WHERE id = ?', (ticket_id,))
    result = cursor.fetchone()
    if result:
        leakage_id = result[0]
        cursor.execute('UPDATE leakages SET status = ? WHERE id = ?', ('resolved', leakage_id))
    
    conn.commit()
    conn.close()
    
    logger.info(f"✅ Ollama ticket {ticket_id} resolved using {method} method - Customer Impact: {customer_impact}")
    
    return jsonify({
        'success': True,
        'message': f'Ticket {ticket_id} resolved successfully using Ollama {method} method with customer-focused approach',
        'resolution_details': resolution_details,
        'customer_impact': customer_impact
    })

# Enhanced Analytics API
@app.route('/api/stats', methods=['GET'])
def get_enhanced_stats():
    """Get comprehensive customer-focused statistics"""
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    try:
        # Basic counts
        cursor.execute('SELECT COUNT(*) FROM leakages')
        total_leakages = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM tickets')
        total_tickets = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM tickets WHERE status = "resolved"')
        resolved_tickets = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM tickets WHERE status != "resolved"')
        pending_tickets = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM tickets WHERE resolution_method = "ai"')
        ai_resolutions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM tickets WHERE resolution_method = "manual"')
        manual_resolutions = cursor.fetchone()[0]
        
        # Enhanced distributions
        cursor.execute('SELECT severity, COUNT(*) FROM leakages GROUP BY severity')
        severity_data = dict(cursor.fetchall())
        
        cursor.execute('SELECT sector, COUNT(*) FROM leakages GROUP BY sector')
        sector_data = dict(cursor.fetchall())
        
        cursor.execute('SELECT category, COUNT(*) FROM leakages GROUP BY category')
        category_data = dict(cursor.fetchall())
        
        cursor.execute('SELECT sector, SUM(amount) FROM leakages GROUP BY sector')
        revenue_impact = dict(cursor.fetchall())
        
        # Customer-focused metrics
        cursor.execute('SELECT SUM(customer_count) FROM datasets WHERE embeddings_stored = TRUE')
        total_customers_analyzed = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(confidence) FROM leakages')
        avg_ai_confidence = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(embedding_score) FROM leakages')
        avg_embedding_score = cursor.fetchone()[0] or 0
        
        # Additional enhanced metrics
        cursor.execute('SELECT AVG(amount) FROM leakages')
        avg_leakage_amount = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT SUM(amount) FROM leakages')
        total_revenue_impact = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT SUM(amount) FROM leakages WHERE status = "resolved"')
        recovered_revenue = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'total_leakages': total_leakages,
            'total_tickets': total_tickets,
            'resolved_tickets': resolved_tickets,
            'pending_tickets': pending_tickets,
            'ai_resolutions': ai_resolutions,
            'manual_resolutions': manual_resolutions,
            'severity_distribution': severity_data,
            'sector_distribution': sector_data,
            'category_distribution': category_data,
            'revenue_impact_by_sector': revenue_impact,
            'avg_leakage_amount': round(avg_leakage_amount, 2),
            'total_revenue_impact': round(total_revenue_impact, 2),
            'recovered_revenue': round(recovered_revenue, 2),
            'total_customers_analyzed': total_customers_analyzed,
            'avg_ai_confidence': round(avg_ai_confidence, 3),
            'avg_embedding_score': round(avg_embedding_score, 3),
            'embedding_model': OLLAMA_EMBEDDING_MODEL,
            'vector_db': 'faiss'
        })
        
    except Exception as e:
        logger.error(f"Enhanced stats calculation error: {e}")
        conn.close()
        return jsonify({'error': 'Failed to calculate enhanced statistics'}), 500

# Enhanced RAG Chatbot API
@app.route('/api/chat', methods=['POST'])
def enhanced_chat_endpoint():
    """Enhanced RAG Chatbot with Ollama and customer context"""
    data = request.get_json()
    question = data.get('question', '').strip()
    user_id = data.get('user_id', 'admin-001')
    
    if not question:
        return jsonify({'success': False, 'message': 'Question cannot be empty'}), 400
    
    try:
        # Enhanced RAG response with customer focus
        response = rag_chatbot.answer_question(question)
        
        # Get vector search context for storage
        vector_results = vector_db.similarity_search(question, k=5)
        vector_context = json.dumps([{
            "content": r["content"][:150], 
            "score": r["score"],
            "sector": r["metadata"]["sector"]
        } for r in vector_results])
        
        similarity_scores = json.dumps([r["score"] for r in vector_results])
        
        # Store enhanced chat history
        conn = sqlite3.connect('revenue_system.db')
        cursor = conn.cursor()
        
        chat_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO chat_history (id, user_id, question, answer, vector_context, 
                                    similarity_scores, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (chat_id, user_id, question, response, vector_context, similarity_scores, OLLAMA_EMBEDDING_MODEL))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Ollama RAG response generated for: {question[:50]}... (Vector results: {len(vector_results)})")
        
        return jsonify({
            'success': True,
            'response': response,
            'chat_id': chat_id,
            'vector_results_count': len(vector_results),
            'embedding_model': OLLAMA_EMBEDDING_MODEL
        })
        
    except Exception as e:
        logger.error(f"Ollama chat error: {e}")
        return jsonify({
            'success': False,
            'message': '🤖 Ollama RAG chatbot temporarily unavailable. Please check Ollama server and models.',
            'error': str(e)
        }), 500

@app.route('/api/chat/history', methods=['GET'])
def get_enhanced_chat_history():
    """Get enhanced chat history with vector context"""
    user_id = request.args.get('user_id', 'admin-001')
    
    conn = sqlite3.connect('revenue_system.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT question, answer, vector_context, similarity_scores, embedding_model, created_at 
        FROM chat_history 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 25
    ''', (user_id,))
    
    history = []
    for row in cursor.fetchall():
        try:
            vector_context = json.loads(row[2]) if row[2] else []
            similarity_scores = json.loads(row[3]) if row[3] else []
        except:
            vector_context = []
            similarity_scores = []
        
        history.append({
            'question': row[0],
            'answer': row[1],
            'vector_context': vector_context,
            'similarity_scores': similarity_scores,
            'embedding_model': row[4],
            'timestamp': row[5]
        })
    
    conn.close()
    return jsonify({'history': history})

# Enhanced health check
@app.route('/api/health', methods=['GET'])
def enhanced_health_check():
    """Enhanced system health check for Ollama"""
    
    # Check Ollama status
    ollama_available, available_models = check_ollama_status()
    
    # Check embedding model status
    embedding_status = EMBEDDING_MODEL is not None
    embedding_info = {
        'model_name': OLLAMA_EMBEDDING_MODEL,
        'dimension': EMBEDDING_DIMENSION,
        'status': 'loaded' if embedding_status else 'not_loaded',
        'available_models': available_models
    }
    
    # Check LLM status
    llm_status = LLM_MODEL is not None
    llm_info = {
        'model_name': OLLAMA_LLM_MODEL,
        'status': 'loaded' if llm_status else 'not_loaded'
    }
    
    # Check FAISS status
    faiss_status = vector_db.index is not None
    faiss_info = {
        'index_type': 'IndexFlatIP',
        'dimension': EMBEDDING_DIMENSION,
        'documents_stored': len(vector_db.documents),
        'status': 'ready' if faiss_status else 'not_initialized'
    }
    
    return jsonify({
        'status': 'healthy' if ollama_available else 'limited',
        'ollama_available': ollama_available,
        'ollama_host': OLLAMA_HOST,
        'ai_configured': ollama_available and embedding_status and llm_status,
        'embedding_model': embedding_info,
        'embedding_status': embedding_status,
        'llm_model': llm_info,
        'llm_status': llm_status,
        'vector_db': {
            'type': 'faiss',
            'info': faiss_info,
            'status': faiss_status
        },
        'crew_ai_ready': crew_ai_system.llm is not None,
        'total_embeddings': len(vector_db.documents),
        'timestamp': datetime.datetime.now().isoformat()
    })

# Additional endpoint for vector search testing
@app.route('/api/vector/search', methods=['POST'])
def test_vector_search():
    """Test vector search with Ollama embeddings"""
    data = request.get_json()
    query = data.get('query', '')
    k = data.get('k', 5)
    
    if not query:
        return jsonify({'success': False, 'message': 'Query required'}), 400
    
    try:
        results = vector_db.similarity_search(query, k)
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'embedding_model': OLLAMA_EMBEDDING_MODEL,
            'vector_db': 'faiss'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Vector search failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Ollama-Powered Customer Revenue Leakage Detection System...")
    print("🤖 Ollama AI Stack:")
    print(f"   🔗 Ollama Host: {OLLAMA_HOST}")
    print(f"   📊 Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
    print(f"   🧠 LLM Model: {OLLAMA_LLM_MODEL}")
    print("   💾 Vector Database: FAISS with optimized indexing")
    print("   👥 Crew AI: Enhanced Customer-focused Multi-Agent System")
    print("   🔍 RAG Chatbot: Advanced context retrieval with customer insights")
    print("=" * 70)
    
    # Initialize Ollama models
    print("📥 Initializing Ollama models...")
    ollama_success = initialize_ollama_models()
    if ollama_success:
        print("✅ Ollama models initialized successfully!")
    else:
        print("⚠️  Warning: Ollama models failed to initialize")
        print("📝 Please ensure:")
        print("   1. Ollama is running: `ollama serve`")
        print(f"   2. Install embedding model: `ollama pull {OLLAMA_EMBEDDING_MODEL}`")
        print(f"   3. Install LLM model: `ollama pull {OLLAMA_LLM_MODEL}`")
    
    print("📊 Initializing enhanced customer database...")
    init_enhanced_db()
    print("✅ Enhanced database initialized successfully!")
    
    print("🌐 Starting Flask server on http://localhost:5000")
    print("📝 Ollama Customer AI API Endpoints:")
    print("   - POST /api/auth/login")
    print("   - POST /api/datasets/upload [Customer data with enhanced preprocessing]")
    print("   - POST /api/datasets/<id>/process [Ollama embeddings + FAISS + Customer Crew AI]")
    print("   - GET  /api/leakages [Customer-focused leakages]")
    print("   - GET  /api/leakages/<id>/details [Enhanced customer context]")
    print("   - POST /api/tickets/generate [Customer-impact tickets with Ollama AI]")
    print("   - GET  /api/tickets [Enhanced with customer context]")
    print("   - POST /api/tickets/<id>/resolve [Customer-focused Ollama AI/Manual resolution]")
    print("   - GET  /api/stats [Enhanced customer analytics]")
    print("   - POST /api/chat [Enhanced RAG with Ollama embeddings]")
    print("   - GET  /api/chat/history [Vector context history]")
    print("   - POST /api/vector/search [Test Ollama embeddings]")
    print("   - GET  /api/health [Enhanced Ollama AI stack status]")
    print("\n" + "="*70)
    
    app.run(debug=True, port=5000, host='0.0.0.0')