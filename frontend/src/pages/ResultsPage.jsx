import React from 'react';
import { Box, Typography, Grid, Paper } from '@mui/material';

const MetricBox = ({ label, value }) => (
  <Paper className="glass-panel" sx={{ p: 2, textAlign: 'center', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
    <Typography variant="body2" sx={{ color: 'var(--text-muted)', mb: 1 }}>{label}</Typography>
    <Typography variant="h5" sx={{ fontWeight: 'bold' }}>{value}</Typography>
  </Paper>
);

const ResultsPage = () => {
  return (
    <Box sx={{ maxWidth: 1200 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold' }}>Results Analysis</Typography>
      </Box>
      <Typography variant="body1" sx={{ color: 'var(--text-muted)', mb: 4 }}>
        Detailed analysis of detection performance and predictions based on our experiments.
      </Typography>

      <Grid container spacing={3}>
        {/* Top row: Metrics */}
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox label="Total Videos Evaluated" value="44" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox label="Average Confidence" value="82.43%" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox label="Highest Confidence" value="99.21%" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox label="Low Confidence (<50%)" value="23 (1.85%)" />
        </Grid>

        {/* Charts Row */}
        <Grid item xs={12} md={6}>
          <Paper className="glass-panel" sx={{ p: 3, height: '100%' }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 2 }}>
              Confusion Matrix (EXP 1 Baseline)
            </Typography>
            <Box sx={{ width: '100%', height: 350, display: 'flex', justifyContent: 'center', alignItems: 'center', bgcolor: 'var(--bg-dark)', borderRadius: 2, overflow: 'hidden' }}>
               <img src="/figures/exp1_confusion_matrix.png" alt="Confusion Matrix" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', filter: 'invert(0.9) hue-rotate(180deg)' }} />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper className="glass-panel" sx={{ p: 3, height: '100%' }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 2 }}>
              Training Curves (EXP 1 Baseline)
            </Typography>
            <Box sx={{ width: '100%', height: 350, display: 'flex', justifyContent: 'center', alignItems: 'center', bgcolor: 'var(--bg-dark)', borderRadius: 2, overflow: 'hidden' }}>
               <img src="/figures/exp1_curves.png" alt="Training Curves" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', filter: 'invert(0.9) hue-rotate(180deg)' }} />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ResultsPage;
