import React from 'react';
import { BarChart3, PieChart, TrendingUp, DollarSign, AlertTriangle } from 'lucide-react';
import { useData } from '../../context/DataContext';

export function ChartsSection() {
  const { stats, leakages } = useData();

  if (!stats) {
    return (
      <div className="grid gap-6">
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 p-6 animate-pulse">
          <div className="h-6 bg-slate-700 rounded mb-4 w-1/3"></div>
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-4 bg-slate-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const severityData = stats.severity_distribution || {};
  const sectorData = stats.sector_distribution || {};
  
  // Calculate additional metrics
  const totalRevenueLoss = leakages.reduce((sum, leakage) => sum + leakage.amount, 0);
  const avgLeakageAmount = leakages.length > 0 ? totalRevenueLoss / leakages.length : 0;
  
  const SeverityChart = () => {
    const maxValue = Math.max(...Object.values(severityData));
    const colors = {
      critical: 'bg-red-500',
      high: 'bg-orange-500',
      medium: 'bg-yellow-500',
      low: 'bg-green-500',
    };
    const textColors = {
      critical: 'text-red-400',
      high: 'text-orange-400',
      medium: 'text-yellow-400',
      low: 'text-green-400',
    };

    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 text-red-500" />
          <span>AI Severity Analysis</span>
        </h3>
        <div className="space-y-3">
          {Object.entries(severityData).map(([severity, count]) => (
            <div key={severity} className="flex items-center space-x-3">
              <div className={`w-20 text-sm capitalize font-medium ${textColors[severity as keyof typeof textColors]}`}>
                {severity}
              </div>
              <div className="flex-1 bg-slate-700 rounded-full h-3 overflow-hidden">
                <div
                  className={`h-full ${colors[severity as keyof typeof colors]} transition-all duration-1000`}
                  style={{ width: maxValue > 0 ? `${(count / maxValue) * 100}%` : '0%' }}
                />
              </div>
              <div className="w-12 text-sm text-white font-bold">{count}</div>
              <div className="w-16 text-xs text-slate-400">
                {maxValue > 0 ? `${Math.round((count / maxValue) * 100)}%` : '0%'}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const SectorChart = () => {
    const total = Object.values(sectorData).reduce((sum, val) => sum + val, 0);
    const colors = {
      telecom: 'bg-blue-500',
      healthcare: 'bg-emerald-500',
      banking: 'bg-purple-500',
    };
    const textColors = {
      telecom: 'text-blue-400',
      healthcare: 'text-emerald-400',
      banking: 'text-purple-400',
    };

    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <PieChart className="w-5 h-5" />
          <span>Sector Impact Analysis</span>
        </h3>
        <div className="space-y-3">
          {Object.entries(sectorData).map(([sector, count]) => (
            <div key={sector} className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className={`w-4 h-4 rounded-full ${colors[sector as keyof typeof colors]}`} />
                <span className={`capitalize font-medium ${textColors[sector as keyof typeof textColors]}`}>
                  {sector === 'telecom' ? '📱 Telecom' : 
                   sector === 'healthcare' ? '🏥 Healthcare' : '🏦 Banking'}
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-white font-bold">{count}</span>
                <span className="text-slate-400 text-sm">
                  ({total > 0 ? Math.round((count / total) * 100) : 0}%)
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const TicketTrends = () => {
    const resolutionRate = stats.total_tickets > 0 ? 
      Math.round((stats.resolved_tickets / stats.total_tickets) * 100) : 0;
    const aiEfficiency = (stats.ai_resolutions + stats.manual_resolutions) > 0 ? 
      Math.round((stats.ai_resolutions / (stats.ai_resolutions + stats.manual_resolutions)) * 100) : 0;

    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <TrendingUp className="w-5 h-5" />
          <span>AI Resolution Analytics</span>
        </h3>
        
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-3 bg-slate-700/30 rounded-lg">
              <div className="text-2xl font-bold text-emerald-500">{stats.resolved_tickets}</div>
              <div className="text-sm text-slate-400">Resolved</div>
            </div>
            <div className="text-center p-3 bg-slate-700/30 rounded-lg">
              <div className="text-2xl font-bold text-amber-500">{stats.pending_tickets}</div>
              <div className="text-sm text-slate-400">Pending</div>
            </div>
          </div>
          
          <div className="bg-slate-700/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-300 text-sm">Resolution Rate</span>
              <span className="text-emerald-400 font-bold">{resolutionRate}%</span>
            </div>
            <div className="w-full bg-slate-600 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-emerald-500 to-blue-500 h-2 rounded-full transition-all duration-1000"
                style={{ width: `${resolutionRate}%` }}
              />
            </div>
          </div>
          
          <div className="bg-slate-700/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-300 text-sm">AI Efficiency</span>
              <span className="text-purple-400 font-bold">{aiEfficiency}%</span>
            </div>
            <div className="w-full bg-slate-600 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-1000"
                style={{ width: `${aiEfficiency}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    );
  };

  const RevenueImpactChart = () => {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <DollarSign className="w-5 h-5 text-emerald-500" />
          <span>Revenue Impact Analysis</span>
        </h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-700/30 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-red-400">
              ${totalRevenueLoss.toLocaleString()}
            </div>
            <div className="text-sm text-slate-400">Total Loss Detected</div>
          </div>
          
          <div className="bg-slate-700/30 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-amber-400">
              ${Math.round(avgLeakageAmount).toLocaleString()}
            </div>
            <div className="text-sm text-slate-400">Avg per Leakage</div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="grid gap-6">
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 p-6">
          <SeverityChart />
        </div>
        
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 p-6">
          <SectorChart />
        </div>
        
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 p-6">
          <RevenueImpactChart />
        </div>
      </div>
      
      <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 p-6">
        <TicketTrends />
      </div>
    </div>
  );
}