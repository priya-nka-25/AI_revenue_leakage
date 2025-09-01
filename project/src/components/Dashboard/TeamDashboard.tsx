import React, { useState } from 'react';
import { Header } from '../Layout/Header';
import { useAuth } from '../../context/AuthContext';
import { useData } from '../../context/DataContext';
import { TicketCard } from './TicketCard';
import { ManualResolutionModal } from './ManualResolutionModal';
import { Clock, CheckCircle, AlertTriangle, Zap, Brain, Users, TrendingUp } from 'lucide-react';

export function TeamDashboard() {
  const { user } = useAuth();
  const { getTicketsByRole, resolveTicket, stats } = useData();
  const [resolving, setResolving] = useState<string | null>(null);
  const [showManualModal, setShowManualModal] = useState(false);
  const [selectedTicket, setSelectedTicket] = useState<string | null>(null);

  const userRole = user?.role as 'finance' | 'it';
  const tickets = getTicketsByRole(userRole);
  const openTickets = tickets.filter(t => t.status === 'open');
  const resolvedTickets = tickets.filter(t => t.status === 'resolved');
  const criticalTickets = tickets.filter(t => t.priority === 'critical');
  const aiResolvedTickets = tickets.filter(t => t.resolution_method === 'ai');
  const teamEfficiency = tickets.length > 0 ? Math.round((resolvedTickets.length / tickets.length) * 100) : 0;

  const handleAIResolve = async (ticketId: string) => {
    setResolving(ticketId);
    try {
      await resolveTicket(ticketId, 'ai');
    } catch (error) {
      console.error('AI resolution failed:', error);
    } finally {
      setResolving(null);
    }
  };

  const handleManualResolve = (ticketId: string) => {
    setSelectedTicket(ticketId);
    setShowManualModal(true);
  };

  const handleManualResolutionSubmit = async (ticketId: string, solutions: string[]) => {
    setResolving(ticketId);
    try {
      await resolveTicket(ticketId, 'manual', solutions);
      setShowManualModal(false);
      setSelectedTicket(null);
    } catch (error) {
      console.error('Manual resolution failed:', error);
    } finally {
      setResolving(null);
    }
  };

  const handleCloseManualModal = () => {
    setShowManualModal(false);
    setSelectedTicket(null);
    setResolving(null);
  };

  const teamStats = [
    {
      title: 'Assigned Tickets',
      value: tickets.length,
      icon: Clock,
      color: 'text-blue-500',
      bgColor: 'bg-blue-500/10',
      subtitle: 'Total assigned'
    },
    {
      title: 'Open Tickets',
      value: openTickets.length,
      icon: AlertTriangle,
      color: 'text-amber-500',
      bgColor: 'bg-amber-500/10',
      subtitle: `${criticalTickets.length} critical`
    },
    {
      title: 'Resolved Tickets',
      value: resolvedTickets.length,
      icon: CheckCircle,
      color: 'text-emerald-500',
      bgColor: 'bg-emerald-500/10',
      subtitle: `${teamEfficiency}% efficiency`
    },
    {
      title: 'AI Resolutions',
      value: aiResolvedTickets.length,
      icon: Brain,
      color: 'text-purple-500',
      bgColor: 'bg-purple-500/10',
      subtitle: 'Automated fixes'
    }
  ];

  return (
    <>
      <div className="min-h-screen bg-slate-900">
      <Header />
      
      <main className="p-6 space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">
              {userRole === 'finance' ? '💰 Finance Team' : '🔧 IT Support'} Dashboard
            </h1>
            <p className="text-slate-400">AI-powered ticket management and resolution system</p>
          </div>
          
          <div className="text-right">
            <div className="text-2xl font-bold text-white">{teamEfficiency}%</div>
            <div className="text-slate-400 text-sm">Team Efficiency</div>
          </div>
        </div>

        {/* Team Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {teamStats.map((stat) => (
            <div key={stat.title} className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 p-6 hover:border-slate-600 transition-all hover:transform hover:scale-105">
              <div className="flex items-center space-x-4">
                <div className={`${stat.bgColor} p-3 rounded-lg`}>
                  <stat.icon className={`w-6 h-6 ${stat.color}`} />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">{stat.value}</p>
                  <p className="text-slate-400 text-xs">{stat.title}</p>
                  <p className="text-slate-500 text-xs">{stat.subtitle}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Open Tickets */}
        <div className="space-y-6">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-6 h-6 text-red-500" />
            <h2 className="text-xl font-semibold text-white">🚨 Priority Tickets</h2>
            <span className="bg-amber-500/20 text-amber-400 px-2 py-1 rounded-lg text-sm">
              {openTickets.length}
            </span>
            {criticalTickets.length > 0 && (
              <span className="bg-red-500/20 text-red-400 px-2 py-1 rounded-lg text-sm font-bold">
                {criticalTickets.length} Critical
              </span>
            )}
          </div>

          {openTickets.length === 0 ? (
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 p-8 text-center">
              <CheckCircle className="w-16 h-16 text-emerald-500 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">All Caught Up!</h3>
              <p className="text-slate-400">No open tickets assigned to your team.</p>
            </div>
          ) : (
            <div className="grid gap-6">
              {openTickets.map((ticket) => (
                <TicketCard
                  key={ticket.id}
                  ticket={ticket}
                  onAIResolve={handleAIResolve}
                  onManualResolve={handleManualResolve}
                  isResolving={resolving === ticket.id}
                />
              ))}
            </div>
          )}
        </div>

        {/* Resolved Tickets */}
        {resolvedTickets.length > 0 && (
          <div className="space-y-6">
            <div className="flex items-center space-x-3">
              <CheckCircle className="w-6 h-6 text-emerald-500" />
              <h2 className="text-xl font-semibold text-white">✅ Recently Resolved</h2>
              <span className="bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded-lg text-sm">
                {resolvedTickets.length}
              </span>
              <span className="bg-purple-500/20 text-purple-400 px-2 py-1 rounded-lg text-sm">
                {aiResolvedTickets.length} AI-resolved
              </span>
            </div>

            <div className="grid gap-4">
              {resolvedTickets.slice(0, 3).map((ticket) => (
                <TicketCard
                  key={ticket.id}
                  ticket={ticket}
                  onAIResolve={handleAIResolve}
                  onManualResolve={handleManualResolve}
                  isResolving={false}
                  isResolved={true}
                />
              ))}
            </div>
          </div>
        )}
      </main>
    </div>

      {/* Manual Resolution Modal */}
      <ManualResolutionModal
        isOpen={showManualModal}
        onClose={handleCloseManualModal}
        ticketId={selectedTicket}
        onSubmit={handleManualResolutionSubmit}
        ticket={selectedTicket ? tickets.find(t => t.id === selectedTicket) : null}
      />
    </>
  );
}