import React from 'react';
import { Box, Typography, Grid, Paper } from '@mui/material';

const ModelCard = ({ title, name, metrics, isBest }) => (
  <Paper className="glass-panel" sx={{ 
    p: 3, 
    height: '100%', 
    position: 'relative',
    borderColor: isBest ? 'var(--accent-purple)' : 'var(--border-color)',
    borderWidth: isBest ? 2 : 1,
    overflow: 'visible'
  }}>
    {isBest && (
      <Box sx={{
        position: 'absolute',
        top: -12,
        right: 16,
        bgcolor: 'var(--accent-purple)',
        color: '#fff',
        px: 2,
        py: 0.5,
        borderRadius: 4,
        fontSize: '0.8rem',
        fontWeight: 'bold'
      }}>
        Best Model
      </Box>
    )}
    <Typography variant="h5" sx={{ color: 'var(--accent-purple)', fontWeight: 'bold', mb: 1 }}>{title}</Typography>
    <Typography variant="body1" sx={{ color: 'var(--text-main)', mb: 4, minHeight: 48 }}>{name}</Typography>

    <Grid container spacing={2}>
      <Grid item xs={6}><Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>Accuracy</Typography></Grid>
      <Grid item xs={6} sx={{ textAlign: 'right' }}><Typography variant="body2" sx={{ fontWeight: 'bold' }}>{metrics.accuracy}</Typography></Grid>

      <Grid item xs={6}><Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>Precision</Typography></Grid>
      <Grid item xs={6} sx={{ textAlign: 'right' }}><Typography variant="body2" sx={{ fontWeight: 'bold' }}>{metrics.precision}</Typography></Grid>

      <Grid item xs={6}><Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>Recall</Typography></Grid>
      <Grid item xs={6} sx={{ textAlign: 'right' }}><Typography variant="body2" sx={{ fontWeight: 'bold' }}>{metrics.recall}</Typography></Grid>

      <Grid item xs={6}><Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>F1 Score</Typography></Grid>
      <Grid item xs={6} sx={{ textAlign: 'right' }}><Typography variant="body2" sx={{ fontWeight: 'bold' }}>{metrics.f1}</Typography></Grid>
    </Grid>
  </Paper>
);

const ModelsPage = () => {
  return (
    <Box sx={{ maxWidth: 1200 }}>
      <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>Model Comparison</Typography>
      <Typography variant="body1" sx={{ color: 'var(--text-muted)', mb: 6 }}>
        Compare the performance of different experiments and preprocessing techniques.
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={2.4}>
          <ModelCard 
            title="EXP 1" 
            name="Baseline" 
            metrics={{ accuracy: '68.42%', precision: '66.21%', recall: '67.89%', f1: '67.03%' }} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <ModelCard 
            title="EXP 2" 
            name="CLAHE + Gamma" 
            metrics={{ accuracy: '72.81%', precision: '70.45%', recall: '72.11%', f1: '71.27%' }} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <ModelCard 
            title="EXP 3" 
            name="Bilateral Filter" 
            metrics={{ accuracy: '74.33%', precision: '72.19%', recall: '74.02%', f1: '73.09%' }} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <ModelCard 
            title="EXP 4" 
            name="Unsharp Masking" 
            metrics={{ accuracy: '75.65%', precision: '74.02%', recall: '75.41%', f1: '74.71%' }} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <ModelCard 
            title="EXP 5" 
            name="Hybrid (Best)" 
            metrics={{ accuracy: '78.95%', precision: '77.31%', recall: '78.85%', f1: '78.07%' }} 
            isBest
          />
        </Grid>
      </Grid>
      
      <Typography variant="caption" sx={{ color: 'var(--text-muted)', display: 'block', mt: 6 }}>
        * Performance results are evaluated on the SSL400 validation set.
      </Typography>
    </Box>
  );
};

export default ModelsPage;
