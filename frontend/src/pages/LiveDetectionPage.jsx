import React, { useState, useEffect } from 'react';
import { Box, Typography, Button, FormControl, Select, MenuItem, Grid, Paper } from '@mui/material';
import VideocamOutlinedIcon from '@mui/icons-material/VideocamOutlined';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlined';
import CircleIcon from '@mui/icons-material/Circle';
import WebcamFeed from '../components/WebcamFeed';
import { predictFrames } from '../services/api';

const LiveDetectionPage = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [expId, setExpId] = useState(1); // Default to EXP 1
  const [currentWord, setCurrentWord] = useState('-');
  const [confidence, setConfidence] = useState(0);
  const [sentence, setSentence] = useState('');
  const [recentSigns, setRecentSigns] = useState([]);
  const [lastWordTime, setLastWordTime] = useState(Date.now());
  const [isProcessing, setIsProcessing] = useState(false);
  const [fps, setFps] = useState(0);
  const [detectionTime, setDetectionTime] = useState(0);

  // Auto clear sentence after 5 seconds of inactivity
  useEffect(() => {
    const timer = setInterval(() => {
      if (Date.now() - lastWordTime > 5000 && sentence !== '') {
        // We don't auto-clear sentence to match standard UX, just word
        setCurrentWord('-');
        setConfidence(0);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [lastWordTime, sentence]);

  const handleFramesCaptured = async (frames) => {
    if (isProcessing) return;
    
    const startTime = performance.now();
    setIsProcessing(true);
    try {
      const result = await predictFrames(frames, expId);
      const endTime = performance.now();
      
      setDetectionTime(Math.round(endTime - startTime));
      setFps(Math.round(1000 / (endTime - startTime) * 10) / 10);
      
      if (result.confidence > 0.05) {
        setCurrentWord(result.word_sinhala);
        setConfidence(result.confidence);
        setLastWordTime(Date.now());
        
        setSentence(prev => {
          const words = prev.split(' ').filter(w => w.length > 0);
          const lastWord = words[words.length - 1];
          if (lastWord !== result.word_sinhala) {
            const newWords = [...words, result.word_sinhala];
            
            // Update recent signs array
            setRecentSigns(prevSigns => {
              const updated = [...prevSigns, result.word_sinhala];
              if (updated.length > 4) updated.shift();
              return updated;
            });
            
            if (newWords.length > 10) newWords.shift();
            return newWords.join(' ');
          }
          return prev;
        });
      } else {
        setConfidence(result.confidence);
      }
    } catch (error) {
      console.error("Prediction failed", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClear = () => {
    setSentence('');
    setCurrentWord('-');
    setConfidence(0);
    setRecentSigns([]);
  };

  return (
    <Box sx={{ maxWidth: 1200, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Top Bar */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
            Live <span style={{ color: 'var(--text-muted)' }}>Sign Language Detection</span>
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
            <CircleIcon sx={{ fontSize: 12, color: isDetecting ? '#10b981' : '#ef4444', mr: 1 }} />
            <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>
              {isDetecting ? 'Camera Active' : 'Camera Inactive'}
            </Typography>
          </Box>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button 
            variant="contained" 
            onClick={() => setIsDetecting(!isDetecting)}
            startIcon={<VideocamOutlinedIcon />}
            sx={{ 
              bgcolor: isDetecting ? 'var(--bg-surface-light)' : 'var(--accent-purple)', 
              color: isDetecting ? 'var(--text-main)' : '#fff',
              '&:hover': { bgcolor: isDetecting ? 'var(--border-color)' : 'var(--accent-purple-hover)' }
            }}
          >
            {isDetecting ? "Stop Detection" : "Start Detection"}
          </Button>
          <Button 
            variant="outlined" 
            onClick={handleClear}
            startIcon={<DeleteOutlineIcon />}
            sx={{ borderColor: 'var(--border-color)', color: 'var(--text-main)' }}
          >
            Clear Text
          </Button>
        </Box>
      </Box>

      {/* Main Content */}
      <Grid container spacing={3} sx={{ flexGrow: 1 }}>
        {/* Left Column: Camera & Controls */}
        <Grid item xs={12} md={7} sx={{ display: 'flex', flexDirection: 'column' }}>
          <Paper className="glass-panel" sx={{ p: 0, overflow: 'hidden', flexGrow: 1, minHeight: 400, display: 'flex', justifyContent: 'center', alignItems: 'center', bgcolor: '#000' }}>
             <WebcamFeed onFramesCaptured={handleFramesCaptured} isDetecting={isDetecting} />
          </Paper>

          {/* Bottom Info Bar */}
          <Paper className="glass-panel" sx={{ p: 2, mt: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box>
                <Typography variant="caption" sx={{ color: 'var(--text-muted)', display: 'block', mb: 0.5 }}>Model</Typography>
                <FormControl size="small" sx={{ minWidth: 180 }}>
                  <Select
                    value={expId}
                    onChange={(e) => setExpId(e.target.value)}
                    disabled={isDetecting}
                    sx={{ 
                      color: '#fff', 
                      bgcolor: 'var(--bg-dark)',
                      '& .MuiOutlinedInput-notchedOutline': { borderColor: 'var(--border-color)' }
                    }}
                  >
                    <MenuItem value={1}>EXP 1: Baseline</MenuItem>
                    <MenuItem value={2}>EXP 2: CLAHE + Gamma</MenuItem>
                    <MenuItem value={3}>EXP 3: Bilateral Filter</MenuItem>
                    <MenuItem value={4}>EXP 4: Unsharp Masking</MenuItem>
                    <MenuItem value={5}>EXP 5: Hybrid (Best)</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              {expId === 5 && (
                <Box sx={{ bgcolor: 'rgba(16, 185, 129, 0.1)', color: '#10b981', px: 1.5, py: 0.5, borderRadius: 4, fontSize: '0.75rem', fontWeight: 'bold', mt: 2 }}>
                  Best Performance
                </Box>
              )}
            </Box>

            <Box sx={{ display: 'flex', gap: 4 }}>
              <Box>
                <Typography variant="caption" sx={{ color: 'var(--text-muted)' }}>FPS</Typography>
                <Typography variant="h6" sx={{ color: '#10b981', fontWeight: 'bold' }}>{fps || '--'}</Typography>
              </Box>
              <Box>
                <Typography variant="caption" sx={{ color: 'var(--text-muted)' }}>Detection Time</Typography>
                <Typography variant="h6" sx={{ fontWeight: 'bold' }}>{detectionTime ? `${detectionTime} ms` : '-- ms'}</Typography>
              </Box>
              <Box>
                <Typography variant="caption" sx={{ color: 'var(--text-muted)' }}>Status</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                  <CircleIcon sx={{ fontSize: 10, color: isProcessing ? '#10b981' : 'var(--text-muted)' }} />
                  <Typography variant="body2" sx={{ fontWeight: 'bold', color: isProcessing ? '#10b981' : 'var(--text-muted)' }}>
                    {isProcessing ? 'Detecting...' : 'Idle'}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Right Column: Output */}
        <Grid item xs={12} md={5} sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* Current Sign */}
          <Paper className="glass-panel" sx={{ p: 3 }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 2 }}>Current Sign</Typography>
            <Box sx={{ height: 120, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <Typography variant="h1" sx={{ color: 'var(--accent-purple)', fontWeight: 'bold', fontSize: '5rem' }}>
                {currentWord}
              </Typography>
            </Box>
            <Box sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>Confidence</Typography>
                <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{(confidence * 100).toFixed(2)}%</Typography>
              </Box>
              <Box sx={{ width: '100%', height: 6, bgcolor: 'var(--bg-dark)', borderRadius: 3, overflow: 'hidden' }}>
                <Box sx={{ width: `${Math.min(confidence * 100, 100)}%`, height: '100%', bgcolor: 'var(--accent-purple)', transition: 'width 0.3s ease' }} />
              </Box>
            </Box>
          </Paper>

          {/* Constructed Sentence */}
          <Paper className="glass-panel" sx={{ p: 3, flexGrow: 1 }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 2 }}>Constructed Sentence</Typography>
            <Typography variant="h4" sx={{ color: 'var(--accent-purple)', fontWeight: 'bold', lineHeight: 1.4 }}>
              {sentence || '...'}
            </Typography>
          </Paper>

          {/* Recent Signs */}
          <Paper className="glass-panel" sx={{ p: 3 }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 2 }}>Recent Signs</Typography>
            <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
              {recentSigns.length === 0 ? (
                <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>No signs detected yet</Typography>
              ) : (
                recentSigns.map((sign, index) => (
                  <Box key={index} sx={{ 
                    bgcolor: 'var(--bg-surface-light)', 
                    color: '#fff', 
                    px: 2, py: 1, 
                    borderRadius: 2, 
                    fontWeight: 600,
                    opacity: 0.5 + (index / recentSigns.length) * 0.5 // Fade out older signs
                  }}>
                    {sign}
                  </Box>
                ))
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default LiveDetectionPage;
