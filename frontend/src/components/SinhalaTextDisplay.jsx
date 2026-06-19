import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function SinhalaTextDisplay({ predictions }) {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-gray-500">
        <p className="text-xl">Waiting for motion...</p>
      </div>
    );
  }

  const topPred = predictions[0];
  const confPct = (topPred.confidence * 100).toFixed(1);

  // Colors based on confidence
  let colorClass = "text-red-500";
  let barColor = "bg-red-500";
  if (topPred.confidence > 0.7) {
    colorClass = "text-emerald-400";
    barColor = "bg-emerald-400";
  } else if (topPred.confidence > 0.4) {
    colorClass = "text-amber-400";
    barColor = "bg-amber-400";
  }

  return (
    <div className="flex flex-col items-center text-center p-6 bg-black/40 rounded-xl border border-white/10 backdrop-blur-md">
      <h3 className="text-sm tracking-[0.2em] text-gray-400 uppercase mb-4">Top Prediction</h3>
      
      <AnimatePresence mode="wait">
        <motion.div
          key={topPred.sinhala_word}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="mb-6"
        >
          <h1 className={`text-6xl md:text-8xl font-black mb-2 drop-shadow-2xl ${colorClass}`} style={{ fontFamily: '"Iskoola Pota", serif' }}>
            {topPred.sinhala_word}
          </h1>
          <p className="text-gray-400 text-lg">Class ID: {topPred.class_id}</p>
        </motion.div>
      </AnimatePresence>

      <div className="w-full max-w-md">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-gray-400">Confidence</span>
          <span className="text-white font-mono">{confPct}%</span>
        </div>
        <div className="h-2 w-full bg-white/10 rounded-full overflow-hidden">
          <motion.div 
            className={`h-full ${barColor}`}
            initial={{ width: 0 }}
            animate={{ width: `${confPct}%` }}
            transition={{ type: "spring", bounce: 0, duration: 0.5 }}
          />
        </div>
      </div>

      {predictions.length > 1 && (
        <div className="mt-8 w-full">
          <h4 className="text-xs tracking-widest text-gray-500 uppercase mb-3 text-left">Alternatives</h4>
          <div className="flex flex-col gap-2">
            {predictions.slice(1).map((p, i) => (
              <div key={i} className="flex justify-between items-center bg-white/5 p-3 rounded-lg">
                <span className="text-xl" style={{ fontFamily: '"Iskoola Pota", serif' }}>{p.sinhala_word}</span>
                <span className="text-cyan-400 font-mono text-sm">{(p.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
