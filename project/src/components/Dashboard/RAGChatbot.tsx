import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, Send, Bot, User, Loader2, Brain, Zap, Database, AlertTriangle } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { chatAPI, healthAPI } from '../../services/api';

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
}

export function RAGChatbot() {
  const { user } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'bot',
      content: '🤖 **RAG AI Assistant Ready!**\n\nI can help you analyze:\n• 📊 Revenue leakages and patterns\n• 🎫 Ticket status and resolution metrics\n• 📁 Dataset processing results\n• 🏢 Sector-wise performance analysis\n• 🔍 AI pipeline insights\n\n💡 **Try asking:**\n- "Show me leakage summary"\n- "What\'s the ticket resolution rate?"\n- "Analyze banking sector performance"\n- "How many critical issues are pending?"',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [aiConfigured, setAiConfigured] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Check AI configuration on component mount
    checkAIStatus();
  }, []);

  const checkAIStatus = async () => {
    try {
      const health = await healthAPI.check();
      setAiConfigured(health.ai_configured);
    } catch (error) {
      console.error('Failed to check AI status:', error);
      setAiConfigured(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await chatAPI.ask(inputMessage, user?.id || 'admin-001');
      
      if (response.success) {
        const botMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: response.response || 'Sorry, I couldn\'t process your request.',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error(response.message || 'Chat API failed');
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: '❌ Sorry, I encountered an error. Please check if:\n• Backend server is running\n• OpenAI API key is configured\n• Internet connection is available',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const quickQuestions = [
    "Show me leakage summary",
    "What's the ticket status?",
    "Analyze sector performance", 
    "AI pipeline efficiency",
    "Critical issues pending",
    "Revenue impact by sector"
  ];

  return (
    <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700 h-[600px] flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-2 rounded-lg">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">🤖 RAG AI Assistant</h3>
              <p className="text-slate-400 text-sm">Vector DB + LLM powered Q&A</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {aiConfigured ? (
              <div className="flex items-center space-x-1 text-emerald-400 text-xs">
                <Database className="w-3 h-3" />
                <span>AI Ready</span>
              </div>
            ) : (
              <div className="flex items-center space-x-1 text-amber-400 text-xs">
                <AlertTriangle className="w-3 h-3" />
                <span>Fallback Mode</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-lg p-3 ${
              message.type === 'user' 
                ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white' 
                : 'bg-slate-700/50 text-slate-200 border border-slate-600'
            }`}>
              <div className="flex items-start space-x-2">
                {message.type === 'bot' && (
                  <div className="bg-purple-600 rounded-full p-1 mt-1">
                    <Bot className="w-3 h-3 text-white" />
                  </div>
                )}
                {message.type === 'user' && (
                  <div className="bg-blue-500 rounded-full p-1 mt-1">
                    <User className="w-3 h-3 text-white" />
                  </div>
                )}
                <div className="flex-1">
                  <div className="whitespace-pre-line text-sm leading-relaxed">
                    {message.content}
                  </div>
                  <div className="text-xs opacity-70 mt-2">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-3 max-w-[80%]">
              <div className="flex items-center space-x-2">
                <Loader2 className="w-4 h-4 animate-spin text-purple-400" />
                <span className="text-slate-300 text-sm">🤖 AI is analyzing vector database...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Questions */}
      <div className="p-3 border-t border-slate-700">
        <div className="flex flex-wrap gap-2 mb-3">
          {quickQuestions.map((question) => (
            <button
              key={question}
              onClick={() => setInputMessage(question)}
              className="text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 hover:text-white px-2 py-1 rounded transition-colors border border-slate-600"
            >
              {question}
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="p-4 border-t border-slate-700">
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about leakages, tickets, datasets, or AI insights..."
            className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white placeholder-slate-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent text-sm"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white p-2 rounded-lg transition-all transform hover:scale-105 disabled:opacity-50 disabled:transform-none"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        
        {!aiConfigured && (
          <div className="mt-2 text-xs text-amber-400 flex items-center space-x-1">
            <AlertTriangle className="w-3 h-3" />
            <span>Configure OpenAI API key in backend/.env for full AI features</span>
          </div>
        )}
      </div>
    </div>
  );
}