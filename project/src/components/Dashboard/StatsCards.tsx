import React from 'react';
import { TrendingDown, Ticket, CheckCircle, Clock, Zap, Users, DollarSign, Brain } from 'lucide-react';
import { useData } from '../../context/DataContext';

export function StatsCards() {
  const { stats, leakages } = useData();

  const totalRevenueLoss = leakages.reduce((sum, leakage) => sum + leakage.amount, 0);
  const criticalLeakages = leakages.filter(l => l.severity === 'critical').length;

  if (!stats) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8 gap-6">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 p-6 animate-pulse">
            <div className="h-12 bg-slate-700 rounded mb-4"></div>
            <div className="h-8 bg-slate-700 rounded mb-2"></div>
            <div className="h-4 bg-slate-700 rounded"></div>
          </div>
        ))}
      </div>
    );
  }

  const cards = [
    {
      title: 'Total Leakages',
      value: stats.total_leakages,
      icon: TrendingDown,
      color: 'text-red-500',
      bgColor: 'bg-red-500/10',
      change: `${criticalLeakages} Critical`,
      changeType: 'critical'
    },
    {
      title: 'Revenue Loss',
      value: `$${Math.round(totalRevenueLoss / 1000)}K`,
      icon: DollarSign,
      color: 'text-red-400',
      bgColor: 'bg-red-500/10',
      change: 'Total Impact',
      changeType: 'impact'
    },
    {
      title: 'Tickets Generated',
      value: stats.total_tickets,
      icon: Ticket,
      color: 'text-blue-500',
      bgColor: 'bg-blue-500/10',
      change: 'Auto-assigned',
      changeType: 'info'
    },
    {
      title: 'Resolved Tickets',
      value: stats.resolved_tickets,
      icon: CheckCircle,
      color: 'text-emerald-500',
      bgColor: 'bg-emerald-500/10',
      change: `${Math.round((stats.resolved_tickets / Math.max(stats.total_tickets, 1)) * 100)}%`,
      changeType: 'success'
    },
    {
      title: 'Pending Tickets',
      value: stats.pending_tickets,
      icon: Clock,
      color: 'text-amber-500',
      bgColor: 'bg-amber-500/10',
      change: 'In Progress',
      changeType: 'warning'
    },
    {
      title: 'AI Resolutions',
      value: stats.ai_resolutions,
      icon: Brain,
      color: 'text-purple-500',
      bgColor: 'bg-purple-500/10',
      change: 'Automated',
      changeType: 'ai'
    },
    {
      title: 'Manual Resolutions',
      value: stats.manual_resolutions,
      icon: Users,
      color: 'text-indigo-500',
      bgColor: 'bg-indigo-500/10',
      change: 'Human-led',
      changeType: 'manual'
    },
    {
      title: 'AI Efficiency',
      value: `${Math.round((stats.ai_resolutions / Math.max(stats.ai_resolutions + stats.manual_resolutions, 1)) * 100)}%`,
      icon: Zap,
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-500/10',
      change: 'Success Rate',
      changeType: 'efficiency'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8 gap-6">
      {cards.map((card) => (
        <div key={card.title} className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 p-6 hover:border-slate-600 transition-all hover:transform hover:scale-105">
          <div className="flex items-center justify-between mb-4">
            <div className={`${card.bgColor} p-3 rounded-lg`}>
              <card.icon className={`w-6 h-6 ${card.color}`} />
            </div>
            <span className={`text-xs px-2 py-1 rounded ${
              card.changeType === 'critical' ? 'text-red-400 bg-red-500/10' :
              card.changeType === 'success' ? 'text-emerald-400 bg-emerald-500/10' :
              card.changeType === 'warning' ? 'text-amber-400 bg-amber-500/10' :
              card.changeType === 'ai' ? 'text-purple-400 bg-purple-500/10' :
              card.changeType === 'efficiency' ? 'text-yellow-400 bg-yellow-500/10' :
              'text-blue-400 bg-blue-500/10'
            }`}>
              {card.change}
            </span>
          </div>
          <div>
            <p className="text-2xl font-bold text-white mb-1">{card.value}</p>
            <p className="text-slate-400 text-xs">{card.title}</p>
          </div>
        </div>
      ))}
    </div>
  );
}