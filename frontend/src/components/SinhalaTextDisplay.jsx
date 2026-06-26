import React from 'react';
import { Box, Typography, Paper, CircularProgress } from '@mui/material';

const SinhalaTextDisplay = ({ sentence, currentWord, confidence, isProcessing }) => {
  return (
    <Box sx={{ mt: 2 }}>
      <Paper elevation={3} sx={{ p: 2, mb: 2, bgcolor: '#1e1e1e', color: '#fff', position: 'relative' }}>
        <Typography variant="overline" color="text.secondary">Current Sign</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', minHeight: '56px' }}>
          {isProcessing ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
               <CircularProgress size={24} color="secondary" />
               <Typography variant="body1" color="text.secondary">Analyzing frames...</Typography>
            </Box>
          ) : (
            <>
              <Typography variant="h3" sx={{ fontFamily: '"Iskoola Pota", sans-serif', color: '#ffd700' }}>
                {currentWord || '...'}
              </Typography>
              {confidence > 0 && (
                <Typography variant="h6" color={confidence > 0.65 ? "success.main" : "warning.main"}>
                  {(confidence * 100).toFixed(1)}%
                </Typography>
              )}
            </>
          )}
        </Box>
      </Paper>

      <Paper elevation={3} sx={{ p: 2, minHeight: '100px', bgcolor: '#f5f5f5' }}>
        <Typography variant="overline" color="text.secondary">Constructed Sentence</Typography>
        <Typography variant="h4" sx={{ fontFamily: '"Iskoola Pota", sans-serif', mt: 1, color: '#333' }}>
          {sentence || 'Waiting for signs...'}
        </Typography>
      </Paper>
    </Box>
  );
};

export default SinhalaTextDisplay;
