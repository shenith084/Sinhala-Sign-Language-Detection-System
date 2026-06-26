import React from 'react';
import { Box, Typography, Grid, Paper } from '@mui/material';
import DataObjectIcon from '@mui/icons-material/DataObject';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import ScienceIcon from '@mui/icons-material/Science';

const AboutPage = () => {
  return (
    <Box sx={{ maxWidth: 1200 }}>
      <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 4 }}>About SSL400 System</Typography>

      <Grid container spacing={4}>
        <Grid xs={12} md={6}>
          {/* Project Overview */}
          <Paper className="glass-panel" sx={{ p: 4, mb: 4 }}>
            <Typography variant="h6" sx={{ color: 'var(--text-main)', mb: 2, fontWeight: 'bold' }}>
              Project Overview
            </Typography>
            <Typography variant="body2" sx={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>
              SSL400 System is a real-time sign language detection web application built using deep learning. It recognizes 400 different signs from the SSL400 dataset with high accuracy using the MoViNet-A2 backbone and various preprocessing techniques.
            </Typography>
          </Paper>

          {/* Technologies Used */}
          <Paper className="glass-panel" sx={{ p: 4 }}>
            <Typography variant="h6" sx={{ color: 'var(--text-main)', mb: 3, fontWeight: 'bold' }}>
              Technologies Used
            </Typography>
            <Box sx={{ display: 'flex', gap: 4, justifyContent: 'space-around' }}>
              <Box sx={{ textAlign: 'center' }}>
                <DataObjectIcon sx={{ fontSize: 40, color: '#61dafb', mb: 1 }} />
                <Typography variant="body2">React</Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <DataObjectIcon sx={{ fontSize: 40, color: '#ffcc00', mb: 1 }} />
                <Typography variant="body2">Python</Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <DataObjectIcon sx={{ fontSize: 40, color: '#ff6f00', mb: 1 }} />
                <Typography variant="body2">TensorFlow</Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <DataObjectIcon sx={{ fontSize: 40, color: '#38bdf8', mb: 1 }} />
                <Typography variant="body2">Tailwind / MUI</Typography>
              </Box>
            </Box>
          </Paper>
        </Grid>

        <Grid xs={12} md={6}>
          {/* Dataset */}
          <Paper className="glass-panel" sx={{ p: 3, mb: 3, display: 'flex', alignItems: 'center', gap: 3 }}>
            <Box sx={{ bgcolor: 'var(--bg-surface-light)', p: 2, borderRadius: 2 }}>
              <AutoGraphIcon sx={{ color: '#0ea5e9', fontSize: 32 }} />
            </Box>
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>SSL400 Dataset</Typography>
              <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>400 classes - 1279+ videos - Diverse signers</Typography>
            </Box>
          </Paper>

          {/* Model Architecture */}
          <Paper className="glass-panel" sx={{ p: 3, mb: 3, display: 'flex', alignItems: 'center', gap: 3 }}>
            <Box sx={{ bgcolor: 'var(--bg-surface-light)', p: 2, borderRadius: 2 }}>
              <ScienceIcon sx={{ color: 'var(--accent-purple)', fontSize: 32 }} />
            </Box>
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Model Architecture</Typography>
              <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>MoViNet-A2<br/>Pre-trained on Kinetics-600</Typography>
            </Box>
          </Paper>

          {/* Developed For */}
          <Paper className="glass-panel" sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 3 }}>
            <Box sx={{ bgcolor: 'var(--bg-surface-light)', p: 2, borderRadius: 2 }}>
              <AutoGraphIcon sx={{ color: '#10b981', fontSize: 32 }} />
            </Box>
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Research & Academic Purpose</Typography>
              <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>Final Year Software Engineering Project</Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
      
      <Typography variant="body2" sx={{ color: 'var(--text-muted)', mt: 8, textAlign: 'left' }}>
        © 2026 SSL400 System. All rights reserved.
      </Typography>
    </Box>
  );
};

export default AboutPage;
