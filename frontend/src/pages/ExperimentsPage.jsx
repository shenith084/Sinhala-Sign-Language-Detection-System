import React, { useEffect, useState } from 'react';
import { fetchExperiments } from '../services/api';
import { Settings, Image as ImageIcon } from 'lucide-react';

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState({});

  useEffect(() => {
    const loadExp = async () => {
      const data = await fetchExperiments();
      setExperiments(data);
    };
    loadExp();
  }, []);

  return (
    <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <div className="mb-12">
        <h1 className="text-3xl font-bold text-white mb-2">Research Experiments</h1>
        <p className="text-gray-400">Configuration details for the 5 video enhancement pipelines.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Object.entries(experiments).map(([id, exp]) => (
          <div key={id} className="bg-black/40 border border-white/10 p-6 rounded-2xl backdrop-blur-md">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center">
                <Settings size={20} />
              </div>
              <h2 className="text-xl font-bold text-white">EXP {id}</h2>
            </div>
            
            <h3 className="text-lg font-medium text-cyan-400 mb-4">{exp.name}</h3>
            
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-500 uppercase tracking-wider mb-1">Technique</p>
                <p className="text-gray-300 bg-white/5 px-3 py-2 rounded-lg">{exp.technique}</p>
              </div>
              
              {exp.params && Object.keys(exp.params).length > 0 && (
                <div>
                  <p className="text-sm text-gray-500 uppercase tracking-wider mb-1">Hyperparameters</p>
                  <div className="bg-black/50 border border-white/5 rounded-lg p-3">
                    <pre className="text-xs text-emerald-400 font-mono">
                      {JSON.stringify(exp.params, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
