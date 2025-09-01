import React, { useState } from 'react';
import { Clock, AlertTriangle, CheckCircle, Zap, User, Lightbulb, Brain, DollarSign, Calendar } from 'lucide-react';
import type { Ticket } from '../../services/api';

interface TicketCardProps {
  ticket: Ticket;
  onAIResolve: (ticketId: string) => void;
  onManualResolve: (ticketId: string) => void;
  isResolving: boolean;
  isResolved?: boolean;
}

export function TicketCard({ ticket, onAIResolve, onManualResolve, isResolving, isResolved = false }: TicketCardProps) {
  const [showSuggestions, setShowSuggestions] = useState(false);

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'border-red-500 bg-red-500/5';
      case 'high': return 'border-orange-500 bg-orange-500/5';
      case 'medium': return 'border-yellow-500 bg-yellow-500/5';
      case 'low': return 'border-green-500 bg-green-500/5';
      default: return 'border-slate-500 bg-slate-500/5';
    }
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'critical': return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case 'high': return <AlertTriangle className="w-5 h-5 text-orange-500" />;
      case 'medium': return <Clock className="w-5 h-5 text-yellow-500" />;
      case 'low': return <CheckCircle className="w-5 h-5 text-green-500" />;
      default: return <Clock className="w-5 h-5 text-slate-500" />;
    }
  };

  return (
    <div className={`bg-slate-800/50 backdrop-blur-xl rounded-xl border-2 p-6 ${getPriorityColor(ticket.priority)} ${isResolved ? 'opacity-75' : ''}`}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          {getPriorityIcon(ticket.priority)}
          <div>
            <h3 className="text-lg font-semibold text-white">{ticket.title}</h3>
            <div className="flex items-center space-x-4 text-sm text-slate-400">
              <span>Ticket #{ticket.id}</span>
              <div className="flex items-center space-x-1">
                <DollarSign className="w-3 h-3" />
                <span>${ticket.amount.toLocaleString()}</span>
              </div>
              <div className="flex items-center space-x-1">
                <Calendar className="w-3 h-3" />
                <span>{new Date(ticket.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className={`px-3 py-1 rounded-full text-xs font-medium capitalize ${
            ticket.priority === 'critical' ? 'text-red-400 bg-red-500/20' :
            ticket.priority === 'high' ? 'text-orange-400 bg-orange-500/20' :
            ticket.priority === 'medium' ? 'text-yellow-400 bg-yellow-500/20' :
            'text-green-400 bg-green-500/20'
          }`}>
            {ticket.priority}
          </span>
          
          {isResolved && (
            <span className="px-3 py-1 rounded-full text-xs font-medium text-emerald-400 bg-emerald-500/20">
              Resolved
            </span>
          )}
        </div>
      </div>

      <div className="space-y-4 mb-6">
        <div>
          <h4 className="text-sm font-medium text-slate-300 mb-2 flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            <span>Issue Description</span>
          </h4>
          <p className="text-slate-400 text-sm leading-relaxed">{ticket.description}</p>
        </div>

        <div>
          <h4 className="text-sm font-medium text-slate-300 mb-2 flex items-center space-x-2">
            <Brain className="w-4 h-4 text-purple-500" />
            <span>AI Root Cause Analysis</span>
          </h4>
          <p className="text-slate-400 text-sm leading-relaxed">{ticket.root_cause}</p>
        </div>

        {ticket.ai_suggestions && ticket.ai_suggestions.length > 0 && (
          <div>
            <button
              onClick={() => setShowSuggestions(!showSuggestions)}
              className="flex items-center space-x-2 text-purple-400 hover:text-purple-300 text-sm font-medium mb-3 bg-purple-500/10 px-3 py-2 rounded-lg transition-all"
            >
              <Brain className="w-4 h-4" />
              <span>🤖 AI Resolution Suggestions ({ticket.ai_suggestions.length})</span>
            </button>
            
            {showSuggestions && (
              <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
                {ticket.ai_suggestions.map((suggestion, index) => (
                  <div key={index} className="flex items-start space-x-3 bg-slate-700/50 rounded-lg p-3">
                    <div className="w-6 h-6 bg-purple-600 rounded-full flex items-center justify-center text-white text-xs font-bold mt-0.5">
                      {index + 1}
                    </div>
                    <span className="text-slate-200 text-sm leading-relaxed">{suggestion}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {!isResolved && (
        <div className="flex space-x-3">
          <button
            onClick={() => onAIResolve(ticket.id)}
            disabled={isResolving}
            className="flex-1 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-bold py-3 px-4 rounded-lg transition-all transform hover:scale-105 disabled:opacity-50 disabled:transform-none shadow-lg"
          >
            <div className="flex items-center justify-center space-x-2">
              <Brain className="w-5 h-5" />
              <span>{isResolving ? '🤖 AI Resolving...' : '🤖 AI Auto-Resolve'}</span>
            </div>
          </button>
          
          <button
            onClick={() => onManualResolve(ticket.id)}
            disabled={isResolving}
            className="flex-1 bg-slate-700 hover:bg-slate-600 text-white font-bold py-3 px-4 rounded-lg transition-all transform hover:scale-105 disabled:opacity-50 disabled:transform-none"
          >
            <div className="flex items-center justify-center space-x-2">
              <User className="w-5 h-5" />
              <span>👤 Manual Resolve</span>
            </div>
          </button>
        </div>
      )}

      {isResolved && ticket.resolved_at && (
        <div className="flex items-center justify-between text-sm text-slate-400 mt-4 pt-4 border-t border-slate-700">
          <span>✅ Resolved on {new Date(ticket.resolved_at).toLocaleDateString()}</span>
          <div className="flex items-center space-x-2">
            {ticket.resolution_method === 'ai' ? (
              <>
                <Brain className="w-4 h-4 text-purple-500" />
                <span>🤖 AI Resolution</span>
              </>
            ) : (
              <>
                <User className="w-4 h-4 text-blue-500" />
                <span>👤 Manual Resolution</span>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}