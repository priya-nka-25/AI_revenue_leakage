import React, { useEffect } from 'react';
import { CheckCircle, X, Zap, Send } from 'lucide-react';

interface SuccessModalProps {
  isOpen: boolean;
  onClose: () => void;
  message: string;
}

export function SuccessModal({ isOpen, onClose, message }: SuccessModalProps) {
  useEffect(() => {
    if (isOpen) {
      const timer = setTimeout(() => {
        onClose();
      }, 4000);
      return () => clearTimeout(timer);
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      
      <div className="relative bg-gradient-to-br from-emerald-500/20 to-blue-500/20 border border-emerald-500/50 rounded-xl p-8 max-w-lg w-full mx-4 animate-in zoom-in-95 duration-200 backdrop-blur-xl">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
        
        <div className="flex flex-col items-center text-center">
          <div className="bg-emerald-500/30 p-6 rounded-full mb-6 animate-pulse">
            <div className="relative">
              <CheckCircle className="w-16 h-16 text-emerald-400" />
              <Zap className="w-6 h-6 text-yellow-400 absolute -top-1 -right-1 animate-bounce" />
            </div>
          </div>
          
          <div className="text-white text-2xl font-bold mb-3">
            🎉 Success!
          </div>
          
          <p className="text-slate-200 leading-relaxed text-lg font-medium">
            {message}
          </p>
          
          <div className="mt-6 flex items-center space-x-2 text-slate-400 text-sm">
            <Send className="w-4 h-4" />
            <span>Notification sent to assigned team</span>
          </div>
        </div>
      </div>
    </div>
  );
}