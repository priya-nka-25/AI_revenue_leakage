import React, { useState } from 'react';
import { X, User, Send, Loader2 } from 'lucide-react';
import type { Ticket } from '../../services/api';

interface ManualResolutionModalProps {
  isOpen: boolean;
  onClose: () => void;
  ticketId: string | null;
  onSubmit: (ticketId: string, solutions: string[]) => void;
  ticket: Ticket | null;
}

export function ManualResolutionModal({ isOpen, onClose, ticketId, onSubmit, ticket }: ManualResolutionModalProps) {
  const [resolution, setResolution] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  React.useEffect(() => {
    if (isOpen && ticket) {
      // Pre-fill with AI suggestion
      const aiSuggestion = ticket.ai_suggestions && ticket.ai_suggestions.length > 0 
        ? ticket.ai_suggestions.join('\n\n') 
        : `Recommended resolution for ${ticket.title}:\n\n1. Investigate the root cause: ${ticket.root_cause}\n2. Implement corrective measures\n3. Monitor for similar issues`;
      setResolution(aiSuggestion);
    }
  }, [isOpen, ticket]);

  const handleSubmit = async () => {
    if (!ticketId || !resolution.trim()) return;
    
    setIsSubmitting(true);
    try {
      await onSubmit(ticketId, [resolution]);
      setResolution('');
      onClose();
    } catch (error) {
      console.error('Failed to submit resolution:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen || !ticket) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      
      <div className="relative bg-slate-800 border border-slate-700 rounded-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <h2 className="text-xl font-semibold text-white">Manual Resolution</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Ticket Details */}
          <div className="bg-slate-700/30 rounded-lg p-4">
            <h3 className="text-lg font-medium text-white mb-3">{ticket.title}</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-slate-400">Severity:</span>
                <span className={`ml-2 px-2 py-1 rounded text-xs font-medium ${
                  ticket.priority === 'critical' ? 'bg-red-500/20 text-red-400' :
                  ticket.priority === 'high' ? 'bg-orange-500/20 text-orange-400' :
                  ticket.priority === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-green-500/20 text-green-400'
                }`}>
                  {ticket.priority.toUpperCase()}
                </span>
              </div>
              <div>
                <span className="text-slate-400">Amount:</span>
                <span className="ml-2 text-red-400 font-semibold">
                  ${ticket.amount.toLocaleString()}
                </span>
              </div>
            </div>
            <div className="mt-3">
              <span className="text-slate-400">Root Cause:</span>
              <p className="text-white mt-1">{ticket.root_cause}</p>
            </div>
          </div>

          {/* Resolution Input */}
          <div className="space-y-3">
            <label className="block text-sm font-medium text-slate-300">
              Resolution Details (AI-suggested, editable)
            </label>
            <textarea
              value={resolution}
              onChange={(e) => setResolution(e.target.value)}
              className="w-full h-32 bg-slate-700/50 border border-slate-600 rounded-lg p-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              placeholder="Enter resolution details..."
            />
            <div className="flex items-center space-x-2 text-xs text-slate-400">
              <User className="w-4 h-4" />
              <span>You can edit the AI suggestion above or write your own resolution</span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end space-x-3 p-6 border-t border-slate-700">
          <button
            onClick={onClose}
            className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!resolution.trim() || isSubmitting}
            className="px-6 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Resolving...</span>
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                <span>Resolve Ticket</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}