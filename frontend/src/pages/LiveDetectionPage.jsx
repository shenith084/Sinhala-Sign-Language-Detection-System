import React, { useState, useEffect } from 'react';
import { Box, Typography, Button, FormControl, Select, MenuItem, Grid, Paper } from '@mui/material';
import VideocamOutlinedIcon from '@mui/icons-material/VideocamOutlined';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlined';
import CircleIcon from '@mui/icons-material/Circle';
import WebcamFeed from '../components/WebcamFeed';
import { predictFrames } from '../services/api';

const LiveDetectionPage = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const expId = 2; // Hardcode to EXP 2
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

      // Lowered threshold to 5% because the model's confidence scores are currently around 15%
      if (result.confidence > 0.05) {
        setCurrentWord(result.word_sinhala);
        setConfidence(result.confidence);
        setLastWordTime(Date.now());
      } else {
        // If confidence is low, don't show a word (just show idle state)
        setConfidence(result.confidence);
        setCurrentWord('-');
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

        {/* Right: Output Grid (Simplified for single word) */}
        <Box sx={{ flex: { xs: 'none', md: 1.5 }, display: 'flex', flexDirection: 'column', gap: 2, minHeight: 0 }}>

          {/* Current Sign */}
          <Paper className="glass-panel" sx={{ flexGrow: 1, p: 4, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
            <Typography variant="h5" sx={{ color: 'var(--text-muted)', mb: 4 }}>Detected Sign</Typography>
            
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexGrow: 1 }}>
              <Typography variant="h1" sx={{ color: 'var(--accent-purple)', fontWeight: 'bold', fontSize: { xs: '4rem', md: '6rem' }, textAlign: 'center', wordBreak: 'break-word', textShadow: '0 0 20px var(--accent-purple-glow)' }}>
                {currentWord}
              </Typography>
            </Box>

            <Box sx={{ mt: 4, width: '100%', maxWidth: '400px' }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)' }}>Confidence</Typography>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: confidence > 0.5 ? '#10b981' : 'var(--accent-purple)' }}>
                  {(confidence * 100).toFixed(2)}%
                </Typography>
              </Box>
              <Box sx={{ width: '100%', height: 8, bgcolor: 'var(--bg-dark)', borderRadius: 4, overflow: 'hidden' }}>
                <Box sx={{ width: `${Math.min(confidence * 100, 100)}%`, height: '100%', bgcolor: confidence > 0.5 ? '#10b981' : 'var(--accent-purple)', transition: 'width 0.3s ease, background-color 0.3s ease' }} />
              </Box>
            </Box>
          </Paper>

        </Box>
      </Box>

      {/* Bottom Bar (Full Width) */}
      <Paper className="glass-panel" sx={{ width: '100%', p: 2, mt: 2, display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, alignItems: { xs: 'flex-start', sm: 'center' }, justifyContent: 'flex-end', gap: 2, flexShrink: 0 }}>

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
