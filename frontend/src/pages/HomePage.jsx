import React from 'react';
import { Container, Typography, Box, Grid, Paper, Button } from '@mui/material';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <Container maxWidth="lg" sx={{ mt: 8 }}>
      <Box sx={{ textAlign: 'center', mb: 8 }}>
        <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          SSL400 Research Project
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          Performance Enhancement of Sinhala Sign Language Detection Systems Using Image Enhancement Techniques
        </Typography>
        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center', gap: 2 }}>
          <Button variant="contained" color="primary" size="large" component={Link} to="/live">
            Live Detection
          </Button>
          <Button variant="outlined" color="primary" size="large" component={Link} to="/experiments">
            View Experiments
          </Button>
        </Box>
      </Box>

      <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
          <Paper elevation={2} sx={{ p: 4, height: '100%', borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom>150 Classes</Typography>
            <Typography color="text.secondary">
              Trained on a low-resource dataset of dynamic Sinhala Sign Language word gestures.
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper elevation={2} sx={{ p: 4, height: '100%', borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom>5 Experimental Enhancements</Typography>
            <Typography color="text.secondary">
              Comparing Baseline, CLAHE, Bilateral Filtering, Unsharp Masking, and Hybrid approaches.
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper elevation={2} sx={{ p: 4, height: '100%', borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom>Real-time Inference</Typography>
            <Typography color="text.secondary">
              Utilizing a custom native TimeDistributed MobileNetV3 + LSTM architecture for low-latency live translation.
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default HomePage;
