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

            if (newWords.length > 4) newWords.shift();
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
      // Add a 3.5 second cooldown so you have time to prepare the next sign!
      setTimeout(() => {
        setIsProcessing(false);
      }, 5000);
    }
  };

  const handleClear = () => {
    setSentence('');
    setCurrentWord('-');
    setConfidence(0);
    setRecentSigns([]);
  };

  return (
    <Box sx={{ width: '100%', height: { xs: 'auto', md: '100%' }, display: 'flex', flexDirection: 'column', overflow: { xs: 'visible', md: 'hidden' }, pb: 1 }}>
      {/* Top Bar */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, justifyContent: 'space-between', alignItems: { xs: 'flex-start', md: 'center' }, gap: 2, mb: 2, flexShrink: 0 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
            Live <span style={{ color: 'var(--text-muted)' }}>Sign Language Detection</span>
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
            <CircleIcon sx={{ fontSize: 12, color: isDetecting ? '#10b981' : '#ef4444', mr: 1 }} />
            <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>
              {isDetecting ? 'Camera Active' : 'Camera Inactive'}
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', gap: 2, width: { xs: '100%', md: 'auto' } }}>
          <Button
            variant="contained"
            onClick={() => setIsDetecting(!isDetecting)}
            startIcon={<VideocamOutlinedIcon />}
            sx={{
              flex: { xs: 1, md: 'none' },
              bgcolor: isDetecting ? 'var(--bg-surface-light)' : 'var(--accent-purple)',
              color: isDetecting ? 'var(--text-main)' : '#fff',
              px: 3, py: 1,
              '&:hover': { bgcolor: isDetecting ? 'var(--border-color)' : 'var(--accent-purple-hover)' }
            }}
          >
            {isDetecting ? "Stop Detection" : "Start Detection"}
          </Button>
          <Button
            variant="outlined"
            onClick={handleClear}
            startIcon={<DeleteOutlineIcon />}
            sx={{ flex: { xs: 1, md: 'none' }, borderColor: 'var(--border-color)', color: 'var(--text-main)', px: 3, py: 1 }}
          >
            Clear Text
          </Button>
        </Box>
      </Box>

      {/* Main Content (Camera + Output Cards) */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 2, flexGrow: 1, minHeight: 0 }}>

        {/* Left: Camera */}
        <Paper className="glass-panel" sx={{ flex: { xs: 'none', md: 1.2 }, minHeight: { xs: 250, md: 0 }, p: 0, overflow: 'hidden', display: 'flex', justifyContent: 'center', alignItems: 'center', bgcolor: '#000', borderRadius: 3 }}>
          <WebcamFeed onFramesCaptured={handleFramesCaptured} isDetecting={isDetecting} />
        </Paper>

        {/* Right: Output Grid */}
        <Box sx={{ flex: { xs: 'none', md: 1.5 }, display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) minmax(0, 1fr)' }, gridTemplateRows: { xs: 'auto', md: '1fr 1fr' }, gap: 2, minHeight: 0 }}>

          {/* Current Sign (Tall) */}
          <Paper className="glass-panel" sx={{ gridRow: { xs: 'auto', md: '1 / 3' }, p: 3, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 2 }}>Current Sign</Typography>
            <Box sx={{ flexGrow: 1, minHeight: { xs: 120, md: 0 }, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <Typography variant="h1" sx={{ color: 'var(--accent-purple)', fontWeight: 'bold', fontSize: { xs: '3rem', md: '4rem' }, textAlign: 'center', wordBreak: 'break-word' }}>
                {currentWord}
              </Typography>
            </Box>
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 1 }}>Confidence</Typography>
              <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1, mb: 1 }}>
                <Typography variant="h5" sx={{ fontWeight: 'bold', lineHeight: 1 }}>{(confidence * 100).toFixed(2)}%</Typography>
              </Box>
              <Box sx={{ width: '100%', height: 6, bgcolor: 'var(--bg-dark)', borderRadius: 3, overflow: 'hidden' }}>
                <Box sx={{ width: `${Math.min(confidence * 100, 100)}%`, height: '100%', bgcolor: 'var(--accent-purple)', transition: 'width 0.3s ease' }} />
              </Box>
            </Box>
          </Paper>

          {/* Constructed Sentence (Top Right) */}
          <Paper className="glass-panel" sx={{ p: 3, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 2 }}>Constructed Sentence</Typography>
            <Typography variant="body1" sx={{
              color: 'var(--accent-purple)',
              fontWeight: 'bold',
              lineHeight: 1.6,
              fontSize: { xs: '1rem', md: '1.2rem' }
            }}>
              {sentence || '...'}
            </Typography>
          </Paper>

          {/* Recent Signs (Bottom Right) */}
          <Paper className="glass-panel" sx={{ p: 3, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 2 }}>Recent Signs</Typography>
            <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
              {recentSigns.length === 0 ? (
                <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>No signs detected yet</Typography>
              ) : (
                <>
                  {recentSigns.map((sign, index) => (
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
                  ))}
                  <Typography sx={{ color: 'var(--text-muted)', ml: 1, fontSize: '1.2rem', display: { xs: 'none', md: 'block' } }}>&gt;</Typography>
                </>
              )}
            </Box>
          </Paper>

        </Box>
      </Box>

      {/* Bottom Bar (Full Width) */}
      <Paper className="glass-panel" sx={{ width: '100%', p: 2, mt: 2, display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, alignItems: { xs: 'flex-start', sm: 'center' }, justifyContent: 'space-between', gap: 2, flexShrink: 0 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: { xs: '100%', sm: 'auto' }, justifyContent: 'space-between' }}>
          <Box sx={{ flex: { xs: 1, sm: 'none' } }}>
            <Typography variant="caption" sx={{ color: 'var(--text-muted)', display: 'block', mb: 0.5 }}>Model</Typography>
            <FormControl size="small" sx={{ width: { xs: '100%', sm: 180 } }}>
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
            <Box sx={{ display: { xs: 'none', md: 'block' }, bgcolor: 'rgba(16, 185, 129, 0.1)', color: '#10b981', px: 1.5, py: 0.5, borderRadius: 4, fontSize: '0.75rem', fontWeight: 'bold', mt: 2 }}>
              Best Performance
            </Box>
          )}
        </Box>

        <Box sx={{ display: 'flex', gap: { xs: 3, md: 6 }, flexWrap: 'wrap', width: { xs: '100%', sm: 'auto' } }}>
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
    </Box>
  );
};

export default LiveDetectionPage;
