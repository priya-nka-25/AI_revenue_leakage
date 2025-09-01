import React, { useState, useEffect } from 'react';
import { X, FileText, DollarSign, AlertTriangle, Users, ArrowLeft, Loader2, Send, Brain, CheckCircle, Zap } from 'lucide-react';
import { leakageAPI } from '../../services/api';
import { useData } from '../../context/DataContext';
import type { Leakage } from '../../services/api';

interface TicketDetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  leakageId: string | null;
  onTicketGenerated: (ticketId: string) => void;
}

export function TicketDetailsModal({ isOpen, onClose, leakageId, onTicketGenerated }: TicketDetailsModalProps) {
  const [leakageDetails, setLeakageDetails] = useState<(Leakage & { assigned_department: string; assigned_to: string; filename: string; confidence: number; category: string }) | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { generateTicket } = useData();

  useEffect(() => {
    if (isOpen && leakageId) {
      fetchLeakageDetails();
      setShowConfirmation(false);
    }
  }, [isOpen, leakageId]);

  const fetchLeakageDetails = async () => {
    if (!leakageId) return;
    
    setIsLoading(true);
    try {
      const details = await leakageAPI.getDetails(leakageId);
      setLeakageDetails(details);
    } catch (error) {
      console.error('Failed to fetch leakage details:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleProceedToGenerate = () => {
    setShowConfirmation(true);
  };

  const handleBackToDetails = () => {
    setShowConfirmation(false);
  };

  const handleConfirmGenerate = async () => {
    if (!leakageId) return;
    
    setIsGenerating(true);
    try {
      const result = await generateTicket(leakageId);
      if (result.success && result.ticket_id) {
        onTicketGenerated(result.ticket_id);
      }
    } catch (error) {
      console.error('Failed to generate ticket:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-500 bg-red-500/10 border-red-500';
      case 'high': return 'text-orange-500 bg-orange-500/10 border-orange-500';
      case 'medium': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500';
      case 'low': return 'text-green-500 bg-green-500/10 border-green-500';
      default: return 'text-slate-500 bg-slate-500/10 border-slate-500';
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

  const getCategoryEmoji = (category: string) => {
    switch (category) {
      case 'billing': return '💳';
      case 'integration': return '🔗';
      case 'processing': return '⚙️';
      case 'recognition': return '📊';
      default: return '🔍';
    }
  };

  if (!isOpen) return null;

  if (isLoading) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" />
        <div className="relative bg-slate-800 border border-slate-700 rounded-xl p-8">
          <div className="flex items-center space-x-3">
            <Loader2 className="w-6 h-6 animate-spin text-purple-500" />
            <span className="text-white">🤖 Loading AI analysis results...</span>
          </div>
        </div>
      </div>
    );
  }

  if (!leakageDetails) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
        <div className="relative bg-slate-800 border border-slate-700 rounded-xl p-8">
          <div className="text-center">
            <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-white text-lg font-semibold mb-2">Error Loading AI Analysis</h3>
            <p className="text-slate-400 mb-4">Failed to load leakage details from AI system.</p>
            <button
              onClick={onClose}
              className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Confirmation Modal (Step 2) - Admin clicks OK to confirm
  if (showConfirmation) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
        
        <div className="relative bg-slate-800 border border-slate-700 rounded-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-slate-700">
            <div className="flex items-center space-x-3">
              <Send className="w-6 h-6 text-emerald-500" />
              <h2 className="text-xl font-semibold text-white">🎫 Confirm Ticket Generation</h2>
            </div>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            {/* AI Analysis Summary */}
            <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-lg p-6">
              <div className="flex items-center space-x-3 mb-4">
                <Brain className="w-6 h-6 text-purple-500" />
                <h3 className="text-lg font-semibold text-white">🤖 AI Analysis Summary</h3>
                <span className="bg-purple-500/20 text-purple-400 px-2 py-1 rounded text-xs">
                  Confidence: {Math.round((leakageDetails.confidence || 0.85) * 100)}%
                </span>
              </div>
              
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center">
                  <label className="block text-sm font-medium text-slate-400 mb-2">Sector Analysis</label>
                  <div className="flex items-center justify-center space-x-2">
                    <span className="text-2xl">{getSectorEmoji(leakageDetails.sector)}</span>
                    <span className="text-white capitalize font-semibold">{leakageDetails.sector}</span>
                  </div>
                  <p className="text-slate-400 text-xs mt-1">Source: {leakageDetails.filename}</p>
                </div>
                
                <div className="text-center">
                  <label className="block text-sm font-medium text-slate-400 mb-2">AI Severity Rating</label>
                  <span className={`inline-block px-4 py-2 rounded-full text-sm font-bold capitalize border ${getSeverityColor(leakageDetails.severity)}`}>
                    🚨 {leakageDetails.severity}
                  </span>
                  <p className="text-slate-400 text-xs mt-1">Category: {getCategoryEmoji(leakageDetails.category)} {leakageDetails.category}</p>
                </div>
                
                <div className="text-center">
                  <label className="block text-sm font-medium text-slate-400 mb-2">Revenue Impact</label>
                  <div className="flex items-center justify-center space-x-1">
                    <DollarSign className="w-5 h-5 text-emerald-500" />
                    <span className="text-white font-bold text-xl">
                      ${leakageDetails.amount.toLocaleString()}
                    </span>
                  </div>
                  <p className="text-slate-400 text-xs mt-1">Estimated Loss</p>
                </div>
              </div>
            </div>

            {/* Ticket Preview */}
            <div className="bg-gradient-to-r from-emerald-500/10 to-blue-500/10 border border-emerald-500/30 rounded-lg p-6">
              <div className="flex items-center space-x-3 mb-4">
                <CheckCircle className="w-6 h-6 text-emerald-500" />
                <h3 className="text-lg font-semibold text-white">✅ Ticket Ready for Generation</h3>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Will Be Assigned To</label>
                    <div className="flex items-center space-x-2">
                      <Users className="w-4 h-4 text-blue-500" />
                      <span className="text-white font-semibold">{leakageDetails.assigned_department}</span>
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Priority Level</label>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium capitalize border ${getSeverityColor(leakageDetails.severity)}`}>
                      🚨 {leakageDetails.severity}
                    </span>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Issue Category</label>
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getCategoryEmoji(leakageDetails.category)}</span>
                      <span className="text-white capitalize">{leakageDetails.category}</span>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Revenue Impact</label>
                    <div className="flex items-center space-x-1">
                      <DollarSign className="w-5 h-5 text-emerald-500" />
                      <span className="text-white font-bold text-xl">
                        ${leakageDetails.amount.toLocaleString()}
                      </span>
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Detection Date</label>
                    <p className="text-white">
                      {new Date(leakageDetails.detected_at).toLocaleDateString()}
                    </p>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">AI Confidence</label>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-slate-600 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full"
                          style={{ width: `${(leakageDetails.confidence || 0.85) * 100}%` }}
                        />
                      </div>
                      <span className="text-purple-400 font-bold text-sm">
                        {Math.round((leakageDetails.confidence || 0.85) * 100)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* AI Analysis Details */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-white font-medium mb-3 flex items-center space-x-2">
                  <AlertTriangle className="w-4 h-4 text-amber-500" />
                  <span>🤖 LLM Issue Detection</span>
                </h4>
                <p className="text-slate-300 text-sm leading-relaxed">{leakageDetails.cause}</p>
              </div>
              
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-white font-medium mb-3 flex items-center space-x-2">
                  <Brain className="w-4 h-4 text-purple-500" />
                  <span>🔍 Crew AI Root Cause Analysis</span>
                </h4>
                <p className="text-slate-300 text-sm leading-relaxed">{leakageDetails.root_cause}</p>
              </div>
            </div>

            {/* Confirmation Warning */}
            <div className="bg-amber-500/10 border border-amber-500 rounded-lg p-4">
              <div className="flex items-center space-x-3">
                <AlertTriangle className="w-5 h-5 text-amber-500" />
                <div>
                  <p className="text-amber-400 font-medium">⚠️ Confirm Ticket Generation</p>
                  <p className="text-slate-300 text-sm">
                    Clicking "OK - Generate Ticket" will create and send this ticket to {leakageDetails.assigned_department}. 
                    The ticket will include AI-generated resolution suggestions. This action cannot be undone.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Footer Actions */}
          <div className="flex space-x-3 p-6 border-t border-slate-700">
            <button
              onClick={handleBackToDetails}
              className="flex-1 bg-slate-700 hover:bg-slate-600 text-white font-medium py-3 px-4 rounded-lg transition-all transform hover:scale-105"
            >
              <div className="flex items-center justify-center space-x-2">
                <ArrowLeft className="w-5 h-5" />
                <span>Back to Analysis</span>
              </div>
            </button>
            
            <button
              onClick={handleConfirmGenerate}
              disabled={isGenerating}
              className="flex-2 bg-gradient-to-r from-emerald-600 to-blue-600 hover:from-emerald-700 hover:to-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-all transform hover:scale-105 disabled:opacity-50 disabled:transform-none"
            >
              {isGenerating ? (
                <div className="flex items-center justify-center space-x-2">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Generating Ticket...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <Send className="w-5 h-5" />
                  <span>✅ OK - Generate Ticket</span>
                </div>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Details Modal (Step 1) - Review AI Analysis
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      
      <div className="relative bg-slate-800 border border-slate-700 rounded-xl max-w-5xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center space-x-3">
            <FileText className="w-6 h-6 text-blue-500" />
            <h2 className="text-xl font-semibold text-white">🔍 AI Revenue Leakage Analysis</h2>
            <span className="bg-blue-500/20 text-blue-400 px-2 py-1 rounded text-xs">
              LLM + Crew AI
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Leakage Overview */}
          <div className="bg-gradient-to-r from-red-500/10 to-orange-500/10 border border-red-500/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-red-500" />
              <span>{getSectorEmoji(leakageDetails.sector)}</span>
              <span>Critical Revenue Leakage Detected by AI</span>
              <span className="bg-red-500/20 text-red-400 px-2 py-1 rounded text-xs">
                ID: #{leakageDetails.id.slice(0, 8)}
              </span>
            </h3>
            
            <div className="grid md:grid-cols-4 gap-6">
              <div className="text-center">
                <label className="block text-sm font-medium text-slate-400 mb-1">Sector</label>
                <div className="flex items-center justify-center space-x-2">
                  <span className="text-2xl">{getSectorEmoji(leakageDetails.sector)}</span>
                  <p className="text-white capitalize font-semibold">{leakageDetails.sector}</p>
                </div>
                <p className="text-slate-400 text-xs mt-1">Dataset: {leakageDetails.filename}</p>
              </div>
              
              <div className="text-center">
                <label className="block text-sm font-medium text-slate-400 mb-1">AI Severity Rating</label>
                <span className={`inline-block px-4 py-2 rounded-full text-sm font-bold capitalize border ${getSeverityColor(leakageDetails.severity)}`}>
                  🚨 {leakageDetails.severity}
                </span>
                <p className="text-slate-400 text-xs mt-1">AI Confidence: {Math.round((leakageDetails.confidence || 0.85) * 100)}%</p>
              </div>
              
              <div className="text-center">
                <label className="block text-sm font-medium text-slate-400 mb-1">Revenue Impact</label>
                <div className="flex items-center justify-center space-x-1">
                  <DollarSign className="w-4 h-4 text-emerald-500" />
                  <span className="text-white font-bold text-xl">
                    ${leakageDetails.amount.toLocaleString()}
                  </span>
                </div>
                <p className="text-slate-400 text-xs mt-1">Estimated Loss</p>
              </div>
              
              <div className="text-center">
                <label className="block text-sm font-medium text-slate-400 mb-1">Issue Category</label>
                <div className="flex items-center justify-center space-x-2">
                  <span className="text-lg">{getCategoryEmoji(leakageDetails.category)}</span>
                  <span className="text-white capitalize">{leakageDetails.category}</span>
                </div>
                <p className="text-slate-400 text-xs mt-1">AI Classification</p>
              </div>
            </div>
          </div>

          {/* AI Analysis Results */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-slate-700/50 rounded-lg p-6">
              <h4 className="text-white font-medium mb-4 flex items-center space-x-2">
                <AlertTriangle className="w-5 h-5 text-amber-500" />
                <span>🤖 LLM Issue Detection</span>
              </h4>
              <p className="text-white leading-relaxed">{leakageDetails.cause}</p>
            </div>

            <div className="bg-slate-700/50 rounded-lg p-6">
              <h4 className="text-white font-medium mb-4 flex items-center space-x-2">
                <Brain className="w-5 h-5 text-purple-500" />
                <span>🔍 Crew AI Root Cause Analysis</span>
              </h4>
              <p className="text-white leading-relaxed">{leakageDetails.root_cause}</p>
            </div>
          </div>

          {/* Smart Assignment Preview */}
          <div className="bg-blue-500/10 border border-blue-500 rounded-lg p-6">
            <div className="flex items-center space-x-3 mb-3">
              <Users className="w-5 h-5 text-blue-500" />
              <h4 className="text-blue-400 font-medium">🎯 AI Smart Department Assignment</h4>
            </div>
            <p className="text-slate-300 mb-3">
              AI recommends assigning this ticket to: <span className="font-semibold text-white">{leakageDetails.assigned_department}</span>
            </p>
            <p className="text-slate-400 text-sm">
              🤖 Assignment based on issue category ({leakageDetails.category}), sector expertise, and team workload analysis.
            </p>
          </div>

          {/* Important Notice */}
          <div className="bg-purple-500/10 border border-purple-500 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <Zap className="w-5 h-5 text-purple-500" />
              <div>
                <p className="text-purple-400 font-medium">📋 Manual Confirmation Required</p>
                <p className="text-slate-300 text-sm">
                  Review the AI analysis above. Click "Proceed to Generate Ticket" to see the ticket preview, 
                  then click "OK" to confirm and send it to the assigned department.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex space-x-3 p-6 border-t border-slate-700">
          <button
            onClick={onClose}
            className="flex-1 bg-slate-700 hover:bg-slate-600 text-white font-medium py-3 px-4 rounded-lg transition-all"
          >
            <div className="flex items-center justify-center space-x-2">
              <ArrowLeft className="w-5 h-5" />
              <span>Back to Dashboard</span>
            </div>
          </button>
          
          <button
            onClick={handleProceedToGenerate}
            className="flex-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold py-3 px-6 rounded-lg transition-all transform hover:scale-105"
          >
            <div className="flex items-center justify-center space-x-2">
              <FileText className="w-5 h-5" />
              <span>📋 Proceed to Generate Ticket</span>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}