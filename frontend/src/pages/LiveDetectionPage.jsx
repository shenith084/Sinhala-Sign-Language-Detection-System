import React, { useState, useEffect } from 'react';
import { Container, Grid, Box, Typography, Button, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import WebcamFeed from '../components/WebcamFeed';
import SinhalaTextDisplay from '../components/SinhalaTextDisplay';
import { predictFrames } from '../services/api';

const LiveDetectionPage = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [expId, setExpId] = useState(1); // Default to Baseline
  const [currentWord, setCurrentWord] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [sentence, setSentence] = useState('');
  const [lastWordTime, setLastWordTime] = useState(Date.now());
  const [isProcessing, setIsProcessing] = useState(false);

  // Auto clear sentence after 3 seconds of inactivity
  useEffect(() => {
    const timer = setInterval(() => {
      if (Date.now() - lastWordTime > 3000 && sentence !== '') {
        setSentence('');
        setCurrentWord('');
        setConfidence(0);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [lastWordTime, sentence]);

  const handleFramesCaptured = async (frames) => {
    if (isProcessing) return; // Prevent concurrent requests
    
    setIsProcessing(true);
    try {
      const result = await predictFrames(frames, expId);
      
      if (result.confidence > 0.20) { // Lowered threshold so they can see *something*!
        setCurrentWord(result.word_sinhala);
        setConfidence(result.confidence);
        setLastWordTime(Date.now());
        
        // Add to sentence if different from last word (basic duplicate suppression)
        setSentence(prev => {
          const words = prev.split(' ').filter(w => w.length > 0);
          const lastWord = words[words.length - 1];
          
          if (lastWord !== result.word_sinhala) {
            const newWords = [...words, result.word_sinhala];
            if (newWords.length > 10) newWords.shift(); // Max 10 words
            return newWords.join(' ');
          }
          return prev;
        });
      } else {
        setCurrentWord('Too uncertain...');
        setConfidence(result.confidence);
      }
    } catch (error) {
      console.error("Prediction failed", error);
      setCurrentWord('Error');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>Live Sign Language Detection</Typography>
      
      <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center' }}>
        <Button 
          variant="contained" 
          color={isDetecting ? "error" : "primary"}
          onClick={() => setIsDetecting(!isDetecting)}
          size="large"
        >
          {isDetecting ? "Stop Detection" : "Start Detection"}
        </Button>
        
        <Button variant="outlined" onClick={() => { setSentence(''); setCurrentWord(''); }}>
          Clear Text
        </Button>

        <FormControl sx={{ minWidth: 200, ml: 'auto' }}>
          <InputLabel>Active Model</InputLabel>
          <Select
            value={expId}
            label="Active Model"
            onChange={(e) => setExpId(e.target.value)}
            disabled={isDetecting}
          >
            <MenuItem value={1}>EXP 1: Baseline</MenuItem>
            <MenuItem value={2}>EXP 2: CLAHE + Gamma</MenuItem>
            <MenuItem value={3}>EXP 3: Bilateral Filter</MenuItem>
            <MenuItem value={4}>EXP 4: Unsharp Masking</MenuItem>
            <MenuItem value={5}>EXP 5: Hybrid (Best)</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Grid container spacing={4}>
        <Grid item xs={12} md={7}>
          <WebcamFeed onFramesCaptured={handleFramesCaptured} isDetecting={isDetecting} />
        </Grid>
        <Grid item xs={12} md={5}>
          <SinhalaTextDisplay sentence={sentence} currentWord={currentWord} confidence={confidence} isProcessing={isProcessing} />
        </Grid>
      </Grid>
    </Container>
  );
};

export default LiveDetectionPage;
