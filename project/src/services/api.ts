import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 45000, // Increased timeout for mxbai-embed-large processing
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export interface User {
  id: string;
  username: string;
  role: 'admin' | 'finance' | 'it';
  name: string;
}

export interface Leakage {
  id: string;
  dataset_id: string;
  sector: 'telecom' | 'healthcare' | 'banking';
  severity: 'low' | 'medium' | 'high' | 'critical';
  cause: string;
  root_cause: string;
  amount: number;
  status: 'detected' | 'ticket-generated' | 'resolved';
  confidence: number;
  category: string;
  department: string;
  customer_impact: string;
  sector_specific: string;
  embedding_score: number;
  detected_at: string;
}

export interface Ticket {
  id: string;
  leakage_id: string;
  assigned_to: 'finance' | 'it';
  status: 'open' | 'pending' | 'resolved';
  priority: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  root_cause: string;
  ai_suggestions: string[];
  resolution_method?: 'ai' | 'manual';
  customer_impact: string;
  estimated_timeline: string;
  created_at: string;
  resolved_at?: string;
  sector: string;
  amount: number;
  category: string;
  confidence: number;
}

export interface Stats {
  total_leakages: number;
  total_tickets: number;
  resolved_tickets: number;
  pending_tickets: number;
  ai_resolutions: number;
  manual_resolutions: number;
  severity_distribution: Record<string, number>;
  sector_distribution: Record<string, number>;
  category_distribution: Record<string, number>;
  revenue_impact_by_sector: Record<string, number>;
  avg_leakage_amount: number;
  total_revenue_impact: number;
  recovered_revenue: number;
  total_customers_analyzed: number;
  avg_ai_confidence: number;
  avg_embedding_score: number;
  embedding_model: string;
  vector_db: string;
}

export const authAPI = {
  login: async (username: string, password: string): Promise<{ success: boolean; user?: User; message?: string }> => {
    try {
      const response = await api.post('/auth/login', { username, password });
      return response.data;
    } catch (error: any) {
      return { success: false, message: error.response?.data?.message || 'Login failed' };
    }
  },
};

export const datasetAPI = {
  upload: async (filename: string, sector: string, uploadedBy: string, content?: string): Promise<{ 
    success: boolean; 
    dataset_id?: string; 
    message?: string; 
    customer_count?: number; 
  }> => {
    try {
      const response = await api.post('/datasets/upload', {
        filename,
        sector,
        uploaded_by: uploadedBy,
        content: content || ''
      });
      return response.data;
    } catch (error: any) {
      return { success: false, message: error.response?.data?.message || 'Upload failed' };
    }
  },

  process: async (datasetId: string): Promise<{ 
    success: boolean; 
    leakages_detected?: number; 
    leakages?: Leakage[]; 
    chunks_processed?: number;
    customers_analyzed?: number;
    embedding_model?: string;
    vector_db?: string;
    message?: string; 
  }> => {
    try {
      const response = await api.post(`/datasets/${datasetId}/process`);
      return response.data;
    } catch (error: any) {
      return { success: false, message: error.response?.data?.message || 'Enhanced AI processing failed' };
    }
  },
};

export const leakageAPI = {
  getAll: async (): Promise<Leakage[]> => {
    try {
      const response = await api.get('/leakages');
      return response.data.leakages;
    } catch (error) {
      console.error('Failed to fetch leakages:', error);
      return [];
    }
  },

  getDetails: async (leakageId: string): Promise<(Leakage & { 
    assigned_department: string; 
    assigned_to: string; 
    filename: string;
    customer_count: number;
    embedding_model: string;
  }) | null> => {
    try {
      const response = await api.get(`/leakages/${leakageId}/details`);
      return response.data.leakage;
    } catch (error) {
      console.error('Failed to fetch leakage details:', error);
      return null;
    }
  },
};

export const ticketAPI = {
  generate: async (leakageId: string): Promise<{ 
    success: boolean; 
    ticket_id?: string; 
    assigned_to?: string; 
    department?: string;
    customer_impact?: string;
    estimated_timeline?: string;
    message?: string; 
  }> => {
    try {
      const response = await api.post('/tickets/generate', { leakage_id: leakageId });
      return response.data;
    } catch (error: any) {
      return { success: false, message: error.response?.data?.message || 'Ticket generation failed' };
    }
  },

  getByRole: async (role: string): Promise<Ticket[]> => {
    try {
      const response = await api.get(`/tickets?role=${role}`);
      return response.data.tickets;
    } catch (error) {
      console.error('Failed to fetch tickets:', error);
      return [];
    }
  },

  resolve: async (ticketId: string, method: 'ai' | 'manual', solutions?: string[]): Promise<{ 
    success: boolean; 
    message?: string; 
    resolution_details?: string;
    customer_impact?: string; 
  }> => {
    try {
      const response = await api.post(`/tickets/${ticketId}/resolve`, {
        method,
        solutions,
      });
      return response.data;
    } catch (error: any) {
      return { success: false, message: error.response?.data?.message || 'Resolution failed' };
    }
  },
};

export const statsAPI = {
  get: async (): Promise<Stats | null> => {
    try {
      const response = await api.get('/stats');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch stats:', error);
      return null;
    }
  },
};

export const chatAPI = {
  ask: async (question: string, userId: string): Promise<{ 
    success: boolean; 
    response?: string; 
    chat_id?: string;
    vector_results_count?: number;
    embedding_model?: string;
    message?: string; 
  }> => {
    try {
      const response = await api.post('/chat', {
        question,
        user_id: userId
      });
      return response.data;
    } catch (error: any) {
      return { success: false, message: error.response?.data?.message || 'Enhanced RAG chatbot failed' };
    }
  },

  getHistory: async (userId: string): Promise<{ 
    question: string; 
    answer: string; 
    vector_context: any[];
    similarity_scores: number[];
    embedding_model: string;
    timestamp: string; 
  }[]> => {
    try {
      const response = await api.get(`/chat/history?user_id=${userId}`);
      return response.data.history || [];
    } catch (error) {
      console.error('Failed to fetch chat history:', error);
      return [];
    }
  },
};

export const healthAPI = {
  check: async (): Promise<{ 
    status: string; 
    ai_configured: boolean; 
    embedding_model: any;
    embedding_status: boolean;
    vector_db: any;
    crew_ai_ready: boolean;
    total_embeddings: number;
    timestamp: string; 
  }> => {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      return { 
        status: 'error', 
        ai_configured: false, 
        embedding_model: { status: 'failed' },
        embedding_status: false,
        vector_db: { status: false }, 
        crew_ai_ready: false,
        total_embeddings: 0,
        timestamp: new Date().toISOString() 
      };
    }
  },
};

export const vectorAPI = {
  search: async (query: string, k: number = 5): Promise<{
    success: boolean;
    results?: any[];
    embedding_model?: string;
    vector_db?: string;
    message?: string;
  }> => {
    try {
      const response = await api.post('/vector/search', { query, k });
      return response.data;
    } catch (error: any) {
      return { success: false, message: error.response?.data?.message || 'Vector search failed' };
    }
  },
};