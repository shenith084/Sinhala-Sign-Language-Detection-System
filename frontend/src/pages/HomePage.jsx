import React from 'react';
import { Box, Typography, Button, Grid, Paper } from '@mui/material';
import { Link } from 'react-router-dom';
import VideocamOutlinedIcon from '@mui/icons-material/VideocamOutlined';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutlined';
import FrontHandIcon from '@mui/icons-material/FrontHand';

const FeatureItem = ({ text }) => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
    <CheckCircleOutlineIcon sx={{ color: 'var(--accent-purple)', fontSize: 20 }} />
    <Typography variant="body1" sx={{ color: 'var(--text-muted)' }}>{text}</Typography>
  </Box>
);

const StatBox = ({ title, subtitle }) => (
  <Paper className="glass-panel" sx={{ p: 2, display: 'flex', flexDirection: 'column', justifyContent: 'center', height: '100%' }}>
    <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'var(--accent-purple)', mb: 0.5, textAlign: 'center' }}>
      {title}
    </Typography>
    <Typography variant="body2" sx={{ color: 'var(--text-muted)', textAlign: 'center' }}>
      {subtitle}
    </Typography>
  </Paper>
);

const HomePage = () => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', pt: 4 }}>
      <Grid container spacing={6} sx={{ flexGrow: 1, alignItems: 'center' }}>
        
        {/* Left Side: Hero Text */}
        <Grid item xs={12} md={6}>
          <Typography variant="h2" sx={{ fontWeight: 800, mb: 3, letterSpacing: '-0.5px' }}>
            Smart Sign Language <br />
            Detection System
          </Typography>
          
          <Typography variant="h6" sx={{ color: 'var(--text-muted)', fontWeight: 400, mb: 5, lineHeight: 1.6 }}>
            Real-time sign language recognition using advanced deep learning models and intelligent preprocessing techniques.
          </Typography>

          <Grid container spacing={2} sx={{ mb: 6 }}>
            <Grid item xs={6}><FeatureItem text="Real-time Detection" /></Grid>
            <Grid item xs={6}><FeatureItem text="Multiple Models" /></Grid>
            <Grid item xs={6}><FeatureItem text="High Accuracy" /></Grid>
            <Grid item xs={6}><FeatureItem text="No Account Required" /></Grid>
          </Grid>

          <Box sx={{ display: 'flex', gap: 3 }}>
            <Button 
              component={Link} 
              to="/live" 
              variant="contained" 
              startIcon={<VideocamOutlinedIcon />}
              sx={{ 
                bgcolor: 'var(--accent-purple)', 
                px: 4, py: 1.5, 
                fontSize: '1.1rem',
                '&:hover': { bgcolor: 'var(--accent-purple-hover)' }
              }}
            >
              Start Live Detection
            </Button>
            <Button 
              component={Link} 
              to="/models" 
              variant="outlined" 
              sx={{ 
                borderColor: 'var(--border-color)', 
                color: 'var(--text-main)', 
                px: 4, py: 1.5, 
                fontSize: '1.1rem',
                '&:hover': { borderColor: 'var(--accent-purple)' }
              }}
            >
              View Models
            </Button>
          </Box>
        </Grid>

        {/* Right Side: Graphic */}
        <Grid item xs={12} md={6} sx={{ display: 'flex', justifyContent: 'center' }}>
          <Box sx={{ 
            position: 'relative', 
            width: 400, height: 400, 
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            '&::before': {
              content: '""',
              position: 'absolute',
              width: '100%', height: '100%',
              background: 'radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, transparent 70%)',
              zIndex: 0
            }
          }}>
            <FrontHandIcon sx={{ fontSize: 200, color: 'var(--accent-purple)', zIndex: 1, filter: 'drop-shadow(0 0 20px rgba(139, 92, 246, 0.6))' }} />
            {/* Corner brackets styling to match image */}
            <Box sx={{ position: 'absolute', top: 20, left: 20, width: 40, height: 40, borderTop: '3px solid var(--accent-purple)', borderLeft: '3px solid var(--accent-purple)', borderRadius: '8px 0 0 0' }} />
            <Box sx={{ position: 'absolute', top: 20, right: 20, width: 40, height: 40, borderTop: '3px solid var(--accent-purple)', borderRight: '3px solid var(--accent-purple)', borderRadius: '0 8px 0 0' }} />
            <Box sx={{ position: 'absolute', bottom: 20, left: 20, width: 40, height: 40, borderBottom: '3px solid var(--accent-purple)', borderLeft: '3px solid var(--accent-purple)', borderRadius: '0 0 0 8px' }} />
            <Box sx={{ position: 'absolute', bottom: 20, right: 20, width: 40, height: 40, borderBottom: '3px solid var(--accent-purple)', borderRight: '3px solid var(--accent-purple)', borderRadius: '0 0 8px 0' }} />
          </Box>
        </Grid>

      </Grid>

      {/* Bottom Row */}
      <Box sx={{ mt: 8, display: 'flex', gap: 4, alignItems: 'center' }}>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 1 }}>About Sign Language</Typography>
          <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>
            Sign language is a visual language that uses hand movements, facial expressions, and body postures to convey meaning. Our system helps to bridge the communication gap using AI-powered real-time detection.
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2, flex: 1.5 }}>
          <Grid container spacing={2}>
            <Grid item xs={3}><StatBox title="400" subtitle="Classes" /></Grid>
            <Grid item xs={3}><StatBox title="1279+" subtitle="Training Videos" /></Grid>
            <Grid item xs={3}><StatBox title="Deep Learning" subtitle="MoViNet-A2" /></Grid>
            <Grid item xs={3}><StatBox title="Real-time" subtitle="Prediction" /></Grid>
          </Grid>
        </Box>
      </Box>

    </Box>
  );
};

export default HomePage;
