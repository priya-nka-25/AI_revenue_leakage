import React, { useState } from 'react';
import { AlertTriangle, DollarSign, FileText, ExternalLink, Eye, Brain, TrendingDown } from 'lucide-react';
import { useData } from '../../context/DataContext';
import { TicketDetailsModal } from './TicketDetailsModal';

interface LeakageTableProps {
  onTicketGenerated: (ticketId: string) => void;
}

export function LeakageTable({ onTicketGenerated }: LeakageTableProps) {
  const { leakages } = useData();
  const [sortBy, setSortBy] = useState<'severity' | 'amount' | 'date'>('severity');
  const [selectedLeakage, setSelectedLeakage] = React.useState<string | null>(null);
  const [showTicketModal, setShowTicketModal] = React.useState(false);

  const handleViewDetails = (leakageId: string) => {
    setSelectedLeakage(leakageId);
    setShowTicketModal(true);
  };

  const handleTicketGenerated = (ticketId: string) => {
    setShowTicketModal(false);
    setSelectedLeakage(null);
    onTicketGenerated(ticketId);
  };

  const sortedLeakages = [...leakages].sort((a, b) => {
    switch (sortBy) {
      case 'severity':
        const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
        return (severityOrder[b.severity as keyof typeof severityOrder] || 0) - 
               (severityOrder[a.severity as keyof typeof severityOrder] || 0);
      case 'amount':
        return b.amount - a.amount;
      case 'date':
        return new Date(b.detected_at).getTime() - new Date(a.detected_at).getTime();
      default:
        return 0;
    }
  });

  const totalLeakageAmount = leakages.reduce((sum, leakage) => sum + leakage.amount, 0);
  const criticalLeakages = leakages.filter(l => l.severity === 'critical').length;

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-500 bg-red-500/10';
      case 'high': return 'text-orange-500 bg-orange-500/10';
      case 'medium': return 'text-yellow-500 bg-yellow-500/10';
      case 'low': return 'text-green-500 bg-green-500/10';
      default: return 'text-slate-500 bg-slate-500/10';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'detected': return 'text-amber-500 bg-amber-500/10';
      case 'ticket-generated': return 'text-blue-500 bg-blue-500/10';
      case 'resolved': return 'text-emerald-500 bg-emerald-500/10';
      default: return 'text-slate-500 bg-slate-500/10';
    }
  };

  const getSectorEmoji = (sector: string) => {
    switch (sector) {
      case 'telecom': return '📱';
      case 'healthcare': return '🏥';
      case 'banking': return '🏦';
      default: return '📊';
    }
  };

  return (
    <>
      <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 overflow-hidden">
        <div className="p-6 border-b border-slate-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <TrendingDown className="w-6 h-6 text-red-500" />
              <h2 className="text-xl font-semibold text-white">AI Revenue Leakage Detection Results</h2>
              <span className="bg-red-500/20 text-red-400 px-3 py-1 rounded-lg text-sm font-medium">
                {leakages.length} Detected
              </span>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-slate-400 text-sm">Total Impact</p>
                <div className="flex items-center space-x-1">
                  <DollarSign className="w-4 h-4 text-emerald-500" />
                  <span className="text-white font-bold text-lg">
                    ${totalLeakageAmount.toLocaleString()}
                  </span>
                </div>
              </div>
              
              <div className="text-right">
                <p className="text-slate-400 text-sm">Critical Issues</p>
                <span className="text-red-400 font-bold text-lg">{criticalLeakages}</span>
              </div>
            </div>
          </div>
          
          {/* Sort Controls */}
          <div className="flex items-center space-x-4 mt-4">
            <span className="text-slate-400 text-sm">Sort by:</span>
            {(['severity', 'amount', 'date'] as const).map((option) => (
              <button
                key={option}
                onClick={() => setSortBy(option)}
                className={`px-3 py-1 rounded-lg text-sm font-medium transition-all ${
                  sortBy === option 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                {option === 'severity' ? 'Severity' : option === 'amount' ? 'Impact' : 'Date'}
              </button>
            ))}
          </div>
        </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-slate-700/50">
            <tr>
              <th className="text-left p-4 text-slate-300 font-medium">Leakage ID</th>
              <th className="text-left p-4 text-slate-300 font-medium">Sector</th>
              <th className="text-left p-4 text-slate-300 font-medium">Severity</th>
              <th className="text-left p-4 text-slate-300 font-medium">Issue Detected</th>
              <th className="text-left p-4 text-slate-300 font-medium">AI Root Cause</th>
              <th className="text-left p-4 text-slate-300 font-medium">Revenue Impact</th>
              <th className="text-left p-4 text-slate-300 font-medium">Status</th>
              <th className="text-left p-4 text-slate-300 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortedLeakages.map((leakage, index) => (
              <tr key={leakage.id} className="border-t border-slate-700 hover:bg-slate-700/20 transition-colors">
                <td className="p-4">
                  <div className="flex items-center space-x-2">
                    <Brain className="w-4 h-4 text-purple-500" />
                    <span className="text-slate-300 font-mono text-sm">#{leakage.id.slice(0, 8)}</span>
                  </div>
                </td>
                <td className="p-4">
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">{getSectorEmoji(leakage.sector)}</span>
                    <span className="text-slate-300 capitalize">{leakage.sector}</span>
                  </div>
                </td>
                <td className="p-4">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${getSeverityColor(leakage.severity)}`}>
                    🚨 {leakage.severity}
                  </span>
                </td>
                <td className="p-4">
                  <p className="text-slate-300 text-sm max-w-xs truncate" title={leakage.cause}>
                    {leakage.cause}
                  </p>
                </td>
                <td className="p-4">
                  <p className="text-slate-400 text-sm max-w-xs truncate" title={leakage.root_cause}>
                    {leakage.root_cause}
                  </p>
                </td>
                <td className="p-4">
                  <div className="flex items-center space-x-1">
                    <DollarSign className="w-4 h-4 text-emerald-500" />
                    <span className="text-white font-bold">
                      ${leakage.amount.toLocaleString()}
                    </span>
                  </div>
                </td>
                <td className="p-4">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${getStatusColor(leakage.status)}`}>
                    {leakage.status.replace('-', ' ')}
                  </span>
                </td>
                <td className="p-4">
                  {leakage.status === 'detected' ? (
                    <button
                      onClick={() => handleViewDetails(leakage.id)}
                      className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white text-sm font-bold px-4 py-2 rounded-lg transition-all transform hover:scale-105 shadow-lg"
                    >
                      <div className="flex items-center space-x-2">
                        <Eye className="w-4 h-4" />
                        <span>Analyze & Generate Ticket</span>
                      </div>
                    </button>
                  ) : leakage.status === 'ticket-generated' ? (
                    <span className="text-blue-400 text-sm flex items-center space-x-2">
                      <FileText className="w-4 h-4" />
                      <span>Ticket Generated</span>
                    </span>
                  ) : (
                    <span className="text-emerald-400 text-sm flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4" />
                      <span>Resolved</span>
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {leakages.length === 0 && (
          <div className="p-12 text-center">
            <div className="bg-slate-700/30 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-4">
              <Brain className="w-12 h-12 text-slate-500" />
            </div>
            <h3 className="text-xl font-semibold text-slate-400 mb-2">No Revenue Leakages Detected</h3>
            <p className="text-slate-500">Upload and process a dataset to start AI-powered leakage detection.</p>
          </div>
        )}
      </div>
    </div>

      {/* Ticket Details Modal */}
      <TicketDetailsModal
        isOpen={showTicketModal}
        onClose={() => setShowTicketModal(false)}
        leakageId={selectedLeakage}
        onTicketGenerated={handleTicketGenerated}
      />
    </>
  );
}