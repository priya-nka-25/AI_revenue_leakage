import React, { createContext, useContext, useState, useEffect } from 'react';
import { leakageAPI, ticketAPI, statsAPI, datasetAPI, type Leakage, type Ticket, type Stats } from '../services/api';
import { useAuth } from './AuthContext';

interface DataContextType {
  leakages: Leakage[];
  tickets: Ticket[];
  stats: Stats | null;
  isLoading: boolean;
  uploadDataset: (filename: string, sector: string, content?: string) => Promise<{ 
    success: boolean; 
    dataset_id?: string; 
    customer_count?: number; 
  }>;
  processDataset: (datasetId: string) => Promise<{ 
    success: boolean; 
    leakages_detected?: number;
    customers_analyzed?: number;
    embedding_model?: string;
  }>;
  generateTicket: (leakageId: string) => Promise<{ 
    success: boolean; 
    ticket_id?: string; 
    assigned_to?: string; 
    department?: string;
    customer_impact?: string;
  }>;
  resolveTicket: (ticketId: string, method: 'ai' | 'manual', solutions?: string[]) => Promise<boolean>;
  getTicketsByRole: (role: 'finance' | 'it') => Ticket[];
  refreshData: () => Promise<void>;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export function DataProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const [leakages, setLeakages] = useState<Leakage[]>([]);
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (user) {
      refreshData();
    }
  }, [user]);

  const refreshData = async () => {
    try {
      const [leakagesData, statsData] = await Promise.all([
        leakageAPI.getAll(),
        statsAPI.get()
      ]);
      
      setLeakages(leakagesData);
      setStats(statsData);
      
      if (user && (user.role === 'finance' || user.role === 'it')) {
        const ticketsData = await ticketAPI.getByRole(user.role);
        setTickets(ticketsData);
      } else if (user?.role === 'admin') {
        const ticketsData = await ticketAPI.getByRole('all');
        setTickets(ticketsData);
      }
    } catch (error) {
      console.error('Failed to refresh data:', error);
    }
  };

  const uploadDataset = async (filename: string, sector: string, content?: string): Promise<{ 
    success: boolean; 
    dataset_id?: string; 
    customer_count?: number; 
  }> => {
    if (!user) return { success: false };
    
    setIsLoading(true);
    try {
      const result = await datasetAPI.upload(filename, sector, user.id, content);
      return {
        success: result.success,
        dataset_id: result.dataset_id,
        customer_count: result.customer_count
      };
    } catch (error) {
      console.error('Upload failed:', error);
      return { success: false };
    } finally {
      setIsLoading(false);
    }
  };

  const processDataset = async (datasetId: string): Promise<{ 
    success: boolean; 
    leakages_detected?: number;
    customers_analyzed?: number;
    embedding_model?: string;
  }> => {
    setIsLoading(true);
    try {
      const result = await datasetAPI.process(datasetId);
      if (result.success) {
        await refreshData();
      }
      return {
        success: result.success,
        leakages_detected: result.leakages_detected,
        customers_analyzed: result.customers_analyzed,
        embedding_model: result.embedding_model
      };
    } catch (error) {
      console.error('Processing failed:', error);
      return { success: false };
    } finally {
      setIsLoading(false);
    }
  };

  const generateTicket = async (leakageId: string): Promise<{ 
    success: boolean; 
    ticket_id?: string; 
    assigned_to?: string; 
    department?: string;
    customer_impact?: string;
  }> => {
    try {
      const result = await ticketAPI.generate(leakageId);
      if (result.success) {
        await refreshData();
      }
      return {
        success: result.success,
        ticket_id: result.ticket_id,
        assigned_to: result.assigned_to,
        department: result.department,
        customer_impact: result.customer_impact
      };
    } catch (error) {
      console.error('Ticket generation failed:', error);
      return { success: false };
    }
  };

  const resolveTicket = async (ticketId: string, method: 'ai' | 'manual', solutions?: string[]): Promise<boolean> => {
    try {
      const result = await ticketAPI.resolve(ticketId, method, solutions);
      if (result.success) {
        await refreshData();
      }
      return result.success;
    } catch (error) {
      console.error('Ticket resolution failed:', error);
      return false;
    }
  };

  const getTicketsByRole = (role: 'finance' | 'it') => {
    return tickets.filter(ticket => ticket.assigned_to === role);
  };

  return (
    <DataContext.Provider value={{
      leakages,
      tickets,
      stats,
      isLoading,
      uploadDataset,
      processDataset,
      generateTicket,
      resolveTicket,
      getTicketsByRole,
      refreshData
    }}>
      {children}
    </DataContext.Provider>
  );
}

export function useData() {
  const context = useContext(DataContext);
  if (context === undefined) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
}