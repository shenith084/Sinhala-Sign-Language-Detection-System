import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import LiveDetectionPage from './pages/LiveDetectionPage';
import ExperimentsPage from './pages/ExperimentsPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#0a0a0f] text-white selection:bg-cyan-500/30">
        <div className="fixed inset-0 z-[-2] bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:50px_50px] pointer-events-none" />
        <div className="fixed top-[-100px] left-[-100px] w-[400px] h-[400px] bg-cyan-500/20 rounded-full blur-[100px] z-[-1] pointer-events-none" />
        <div className="fixed bottom-[-150px] right-[-100px] w-[500px] h-[500px] bg-purple-600/20 rounded-full blur-[100px] z-[-1] pointer-events-none" />
        
        <nav className="sticky top-0 z-50 bg-[#0a0a0f]/80 backdrop-blur-xl border-b border-white/10 px-6 py-4">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="font-black text-2xl tracking-tighter">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">SSL</span>400
            </div>
            <div className="flex gap-6 text-sm font-medium text-gray-400">
              <a href="/" className="hover:text-white transition-colors">Home</a>
              <a href="/live" className="hover:text-white transition-colors">Live App</a>
              <a href="/experiments" className="hover:text-white transition-colors">Experiments</a>
            </div>
          </div>
        </nav>

        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/live" element={<LiveDetectionPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
