import React, { useState, useEffect } from 'react';
import { LoginPage } from './components/Auth/LoginPage';
import { AdminDashboard } from './components/Dashboard/AdminDashboard';
import { TeamDashboard } from './components/Dashboard/TeamDashboard';
import { AuthProvider, useAuth } from './context/AuthContext';
import { DataProvider } from './context/DataContext';

function AppContent() {
  const { user, isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  switch (user?.role) {
    case 'admin':
      return <AdminDashboard />;
    case 'finance':
    case 'it':
      return <TeamDashboard />;
    default:
      return <LoginPage />;
  }
}

function App() {
  return (
    <AuthProvider>
      <DataProvider>
        <div className="min-h-screen bg-slate-900">
          <AppContent />
        </div>
      </DataProvider>
    </AuthProvider>
  );
}

export default App;