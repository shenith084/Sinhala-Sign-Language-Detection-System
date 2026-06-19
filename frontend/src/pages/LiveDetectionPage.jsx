import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Volume2, VolumeX } from 'lucide-react';
import WebcamFeed from '../components/WebcamFeed';
import SinhalaTextDisplay from '../components/SinhalaTextDisplay';
import { sendFramesForPrediction, getTtsAudioUrl } from '../services/api';

export default function LiveDetectionPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [predictions, setPredictions] = useState([]);
  
  const audioRef = useRef(new Audio());
  const stabilityCount = useRef(0);
  const currentTopClass = useRef(null);
  const lastSpokenWord = useRef("");

  const handleBufferReady = async (framesBase64) => {
    const results = await sendFramesForPrediction(framesBase64);
    if (results && results.length > 0) {
      setPredictions(results);
      handleTtsLogic(results[0]);
    }
  };

  const handleTtsLogic = (topPred) => {
    if (!ttsEnabled) return;

    if (topPred.confidence > 0.7) {
      if (currentTopClass.current === topPred.class_id) {
        stabilityCount.current += 1;
      } else {
        currentTopClass.current = topPred.class_id;
        stabilityCount.current = 1;
      }

      // 2 consecutive hits (~0.5s stability)
      if (stabilityCount.current >= 2 && topPred.sinhala_word !== lastSpokenWord.current) {
        if (audioRef.current.paused) {
          lastSpokenWord.current = topPred.sinhala_word;
          audioRef.current.src = getTtsAudioUrl(topPred.sinhala_word);
          audioRef.current.play().catch(e => console.error("TTS Play error", e));
        }
        stabilityCount.current = 0;
      }
    } else {
      stabilityCount.current = 0;
    }
  };

  return (
    <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
      <div className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Live Translation</h1>
          <p className="text-gray-400 mt-1">Real-time Sinhala Sign Language detection</p>
        </div>
        
        <div className="flex gap-4">
          <button
            onClick={() => setTtsEnabled(!ttsEnabled)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              ttsEnabled ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50' : 'bg-gray-800 text-gray-400 border border-gray-700'
            }`}
          >
            {ttsEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            Auto TTS
          </button>

          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center gap-2 px-6 py-2 rounded-lg font-bold transition-all shadow-lg hover:shadow-xl hover:-translate-y-0.5 ${
              isRunning ? 'bg-red-500 hover:bg-red-600 text-white shadow-red-500/20' : 'bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white shadow-cyan-500/20'
            }`}
          >
            {isRunning ? <Square size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}
            {isRunning ? 'Stop Camera' : 'Start Camera'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
        <div className="lg:col-span-3">
          <WebcamFeed isRunning={isRunning} onBufferReady={handleBufferReady} />
        </div>
        <div className="lg:col-span-2">
          <SinhalaTextDisplay predictions={predictions} />
        </div>
      </div>
    </div>
  );
}
