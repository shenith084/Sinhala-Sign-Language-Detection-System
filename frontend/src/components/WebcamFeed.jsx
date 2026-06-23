import React, { useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Box, Button, Typography, LinearProgress, Paper } from '@mui/material';

const WebcamFeed = ({ onFramesCaptured, isDetecting }) => {
  const webcamRef = useRef(null);
  const [frames, setFrames] = useState([]);
  
  // Capture frame every 50ms (20 FPS)
  React.useEffect(() => {
    if (!isDetecting) {
      setFrames([]);
      return;
    }

    const captureInterval = setInterval(() => {
      if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          setFrames(prev => {
            const newFrames = [...prev, imageSrc];
            // If we hit 60 frames (3 seconds at 20fps), send them
            if (newFrames.length >= 60) {
              onFramesCaptured(newFrames);
              return []; // Reset buffer
            }
            return newFrames;
          });
        }
      }
    }, 50); // 50ms = 20 fps

    return () => clearInterval(captureInterval);
  }, [isDetecting, onFramesCaptured]);

  const progress = (frames.length / 60) * 100;

  return (
    <Paper elevation={3} sx={{ overflow: 'hidden', borderRadius: 2, position: 'relative' }}>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width="100%"
        height="auto"
        videoConstraints={{ facingMode: "user" }}
        style={{ display: 'block' }}
      />
      {isDetecting && (
        <Box sx={{ width: '100%', position: 'absolute', bottom: 0 }}>
          <LinearProgress variant="determinate" value={progress} color="secondary" />
        </Box>
      )}
    </Paper>
  );
};

export default WebcamFeed;
