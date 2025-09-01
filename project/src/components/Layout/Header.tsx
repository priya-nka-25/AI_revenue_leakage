import React from 'react';
import { Shield, LogOut, Bell, Settings } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

export function Header() {
  const { user, logout } = useAuth();

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin': return 'bg-purple-500';
      case 'finance': return 'bg-emerald-500';
      case 'it': return 'bg-blue-500';
      default: return 'bg-slate-500';
    }
  };

  return (
    <header className="bg-slate-800/50 backdrop-blur-xl border-b border-slate-700 sticky top-0 z-50">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <Shield className="w-8 h-8 text-emerald-500" />
              <div>
                <h1 className="text-xl font-bold text-white">RevenueFortress</h1>
                <p className="text-xs text-slate-400">AI-Powered Leakage Detection</p>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <button className="p-2 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-slate-700">
              <Bell className="w-5 h-5" />
            </button>
            
            <button className="p-2 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-slate-700">
              <Settings className="w-5 h-5" />
            </button>

            <div className="flex items-center space-x-3 px-3 py-2 bg-slate-700 rounded-lg">
              <div className={`w-8 h-8 ${getRoleColor(user?.role || '')} rounded-full flex items-center justify-center`}>
                <span className="text-white font-bold text-sm">
                  {user?.name.split(' ').map(n => n[0]).join('')}
                </span>
              </div>
              <div className="text-sm">
                <div className="text-white font-medium">{user?.name}</div>
                <div className="text-slate-400 capitalize">{user?.role}</div>
              </div>
            </div>

            <button
              onClick={logout}
              className="p-2 text-slate-400 hover:text-red-400 transition-colors rounded-lg hover:bg-slate-700"
              title="Logout"
            >
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}