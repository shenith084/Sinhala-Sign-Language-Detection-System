import React from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';
import { Link } from 'react-router-dom';
import VideocamOutlinedIcon from '@mui/icons-material/VideocamOutlined';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutlined';

const FeatureItem = ({ text }) => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
    <CheckCircleOutlineIcon sx={{ color: 'var(--accent-purple)', fontSize: 18 }} />
    <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>{text}</Typography>
  </Box>
);

const StatBox = ({ title, subtitle }) => (
  <Paper className="glass-panel" sx={{ p: 1.5, display: 'flex', flexDirection: 'column', justifyContent: 'center', height: '100%' }}>
    <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'var(--accent-purple)', mb: 0, textAlign: 'center' }}>
      {title}
    </Typography>
    <Typography variant="caption" sx={{ color: 'var(--text-muted)', textAlign: 'center', fontSize: '0.75rem' }}>
      {subtitle}
    </Typography>
  </Paper>
);

const HomePage = () => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'center', pb: 2 }}>
      
      {/* Top Section: Flex Container */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, flexGrow: 1, alignItems: 'center' }}>
        
        {/* Left Side: Hero Text */}
        <Box sx={{ flex: 1, minWidth: '350px' }}>
          <Typography variant="h3" sx={{ fontWeight: 800, mb: 2, letterSpacing: '-0.5px' }}>
            Smart Sign Language <br />
            Detection System
          </Typography>
          
          <Typography variant="body1" sx={{ color: 'var(--text-muted)', fontWeight: 400, mb: 3, lineHeight: 1.5 }}>
            Real-time sign language recognition using advanced deep learning models and intelligent preprocessing techniques.
          </Typography>

          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1.5, mb: 4 }}>
            <FeatureItem text="Real-time Detection" />
            <FeatureItem text="Multiple Models" />
            <FeatureItem text="High Accuracy" />
            <FeatureItem text="No Account Required" />
          </Box>

          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button 
              component={Link} 
              to="/live" 
              variant="contained" 
              startIcon={<VideocamOutlinedIcon />}
              sx={{ 
                bgcolor: 'var(--accent-purple)', 
                px: 3, py: 1, 
                fontSize: '1rem',
                '&:hover': { bgcolor: 'var(--accent-purple-hover)' }
              }}
            >
              Start Live Detection
            </Button>
          </Box>
        </Box>

        {/* Right Side: Graphic */}
        <Box sx={{ flex: 1, minWidth: '300px', display: 'flex', justifyContent: 'center' }}>
          <Box 
            component="img" 
            src="/assets/3d_glowing_hand.png" 
            alt="3D Glowing Hand"
            sx={{ 
              width: '100%', 
              maxWidth: 380, 
              maxHeight: '50vh',
              objectFit: 'contain',
              filter: 'drop-shadow(0 0 30px rgba(139, 92, 246, 0.3))' 
            }} 
          />
        </Box>

      </Box>

      {/* Bottom Row */}
      <Box sx={{ 
        mt: 4, 
        p: 2.5, 
        bgcolor: '#13141f', 
        borderRadius: 4, 
        display: 'flex', 
        flexWrap: 'wrap',
        gap: 3, 
        alignItems: 'center',
        border: '1px solid #1f2130' 
      }}>
        <Box sx={{ flex: 1, minWidth: { xs: '100%', sm: '250px' }, mb: { xs: 2, sm: 0 } }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 0.5 }}>About Sign Language</Typography>
          <Typography variant="caption" sx={{ color: 'var(--text-muted)', display: 'block', lineHeight: 1.4, fontSize: '0.8rem' }}>
            Sign language is a visual language that uses hand movements, facial expressions, and body postures to convey meaning. Our system helps to bridge the communication gap using AI-powered real-time detection.
          </Typography>
        </Box>
        
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', sm: 'repeat(3, 1fr)' }, gap: 1.5, flex: 1.5, minWidth: 0 }}>
          <StatBox title="8" subtitle="Classes" />
          <StatBox title="98%" subtitle="Top Accuracy" />
          <StatBox title="EfficientNet" subtitle="Architecture" />
        </Box>
      </Box>

    </Box>
  );
};

export default HomePage;
