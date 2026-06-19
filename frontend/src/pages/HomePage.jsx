import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, Beaker, Camera, FileText } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-16">
        <h1 className="text-5xl md:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 mb-6 tracking-tight">
          Sinhala Sign Language
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          An advanced real-time translation system powered by I3D Kinetics-400 Transfer Learning, recognizing 383 unique Sinhala signs.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
        <Link to="/live" className="group p-8 rounded-2xl bg-black/40 border border-white/10 hover:bg-white/5 hover:border-cyan-500/50 transition-all backdrop-blur-md">
          <div className="bg-cyan-500/20 w-16 h-16 rounded-xl flex items-center justify-center text-cyan-400 mb-6 group-hover:scale-110 transition-transform">
            <Camera size={32} />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Live Translation</h2>
          <p className="text-gray-400">Experience real-time translation from your webcam with on-device AI and automated Sinhala Text-to-Speech.</p>
        </Link>

        <Link to="/experiments" className="group p-8 rounded-2xl bg-black/40 border border-white/10 hover:bg-white/5 hover:border-purple-500/50 transition-all backdrop-blur-md">
          <div className="bg-purple-500/20 w-16 h-16 rounded-xl flex items-center justify-center text-purple-400 mb-6 group-hover:scale-110 transition-transform">
            <Beaker size={32} />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Research Experiments</h2>
          <p className="text-gray-400">Explore the 5 video enhancement pipelines (CLAHE, Bilateral Filtering, Unsharp Masking) and compare their performance.</p>
        </Link>

        <div className="group p-8 rounded-2xl bg-black/40 border border-white/10 opacity-70 backdrop-blur-md cursor-not-allowed">
          <div className="bg-amber-500/20 w-16 h-16 rounded-xl flex items-center justify-center text-amber-400 mb-6">
            <Activity size={32} />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Metrics Dashboard</h2>
          <p className="text-gray-400">View real-time inference statistics, confusion matrices, and detailed F1-scores. (Coming Soon)</p>
        </div>

        <div className="group p-8 rounded-2xl bg-black/40 border border-white/10 opacity-70 backdrop-blur-md cursor-not-allowed">
          <div className="bg-emerald-500/20 w-16 h-16 rounded-xl flex items-center justify-center text-emerald-400 mb-6">
            <FileText size={32} />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Documentation</h2>
          <p className="text-gray-400">Read the methodology, architecture diagrams, and usage instructions for the SSL400 dataset. (Coming Soon)</p>
        </div>
      </div>
    </div>
  );
}
