import React, { useState } from 'react';
import { Header } from '../Layout/Header';
import { StatsCards } from './StatsCards';
import { DatasetUpload } from './DatasetUpload';
import { LeakageTable } from './LeakageTable';
import { ChartsSection } from './ChartsSection';
import { SuccessModal } from '../UI/SuccessModal';
import { RAGChatbot } from './RAGChatbot';

export function AdminDashboard() {
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');

  const handleTicketGenerated = (ticketId: string) => {
    setSuccessMessage(`✅ Ticket ${ticketId} Generated Successfully and assigned to the appropriate team!`);
    setShowSuccessModal(true);
  };

  return (
    <div className="min-h-screen bg-slate-900">
      <Header />
      
      <main className="p-6 space-y-8">
        {/* Statistics Cards */}
        <StatsCards />

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-4 gap-8">
          {/* Left Column - Dataset Upload */}
          <div className="lg:col-span-1">
            <DatasetUpload />
          </div>
          
          {/* Middle Column - Charts */}
          <div className="lg:col-span-2">
            <ChartsSection />
          </div>
          
          {/* Right Column - RAG Chatbot */}
          <div className="lg:col-span-1">
            <RAGChatbot />
          </div>
        </div>

        {/* Leakage Detection Results */}
        <LeakageTable onTicketGenerated={handleTicketGenerated} />
      </main>

      <SuccessModal
        isOpen={showSuccessModal}
        onClose={() => setShowSuccessModal(false)}
        message={successMessage}
      />
    </div>
  );
}