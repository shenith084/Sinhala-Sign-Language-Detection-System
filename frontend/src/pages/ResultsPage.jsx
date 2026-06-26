import React, { useState } from 'react';
import { Box, Typography, Grid, Paper, FormControl, Select, MenuItem } from '@mui/material';

const MetricBox = ({ label, value }) => (
  <Paper className="glass-panel" sx={{ p: 2, textAlign: 'center', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
    <Typography variant="body2" sx={{ color: 'var(--text-muted)', mb: 1 }}>{label}</Typography>
    <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'var(--accent-purple)' }}>{value}</Typography>
  </Paper>
);

const experimentData = {
  1: {
    name: "EXP 1: Baseline",
    samples: 100,
    accuracy: "23.00%",
    f1: "15.71%",
    latency: "169.9 ms",
    cmImage: "/figures/exp1_confusion_matrix.png",
    curvesImage: "/figures/exp1_curves.png"
  },
  2: {
    name: "EXP 2: CLAHE + Gamma",
    samples: 100,
    accuracy: "25.00%",
    f1: "18.49%",
    latency: "183.2 ms",
    cmImage: "/figures/exp2_confusion_matrix.png",
    curvesImage: "/figures/exp2_curves.png"
  }
};

const ResultsPage = () => {
  const [expId, setExpId] = useState(1);
  const data = experimentData[expId];

  return (
    <Box sx={{ maxWidth: 1200 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold' }}>Results Analysis</Typography>
        <FormControl size="small" sx={{ minWidth: 200 }}>
          <Select
            value={expId}
            onChange={(e) => setExpId(e.target.value)}
            sx={{ 
              color: '#fff', 
              bgcolor: 'var(--bg-dark)',
              fontWeight: 'bold',
              '& .MuiOutlinedInput-notchedOutline': { borderColor: 'var(--accent-purple)' }
            }}
          >
            <MenuItem value={1}>EXP 1: Baseline</MenuItem>
            <MenuItem value={2}>EXP 2: CLAHE + Gamma</MenuItem>
          </Select>
        </FormControl>
      </Box>
      <Typography variant="body1" sx={{ color: 'var(--text-muted)', mb: 4 }}>
        Detailed analysis of detection performance and training curves based on our experiments.
      </Typography>

      <Grid container spacing={3}>
        {/* Top row: Metrics */}
        <Grid xs={12} sm={6} md={3}>
          <MetricBox label="Total Samples Evaluated" value={data.samples} />
        </Grid>
        <Grid xs={12} sm={6} md={3}>
          <MetricBox label="Top-1 Accuracy" value={data.accuracy} />
        </Grid>
        <Grid xs={12} sm={6} md={3}>
          <MetricBox label="Macro F1 Score" value={data.f1} />
        </Grid>
        <Grid xs={12} sm={6} md={3}>
          <MetricBox label="Inference Latency" value={data.latency} />
        </Grid>

        {/* Charts Row */}
        <Grid xs={12} md={6}>
          <Paper className="glass-panel" sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 2 }}>
              Confusion Matrix ({data.name})
            </Typography>
            <Box sx={{ width: '100%', flexGrow: 1, minHeight: 350, display: 'flex', justifyContent: 'center', alignItems: 'center', bgcolor: 'var(--bg-dark)', borderRadius: 2, overflow: 'hidden' }}>
               <img src={data.cmImage} alt="Confusion Matrix" style={{ maxWidth: '100%', maxHeight: '350px', objectFit: 'contain', filter: 'invert(0.9) hue-rotate(180deg)' }} />
            </Box>
          </Paper>
        </Grid>

        <Grid xs={12} md={6}>
          <Paper className="glass-panel" sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--text-muted)', mb: 2 }}>
              Training Curves ({data.name})
            </Typography>
            <Box sx={{ width: '100%', flexGrow: 1, minHeight: 350, display: 'flex', justifyContent: 'center', alignItems: 'center', bgcolor: 'var(--bg-dark)', borderRadius: 2, overflow: 'hidden' }}>
               <img src={data.curvesImage} alt="Training Curves" style={{ maxWidth: '100%', maxHeight: '350px', objectFit: 'contain', filter: 'invert(0.9) hue-rotate(180deg)' }} />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ResultsPage;
