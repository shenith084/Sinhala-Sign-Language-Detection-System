import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import WaterDropOutlinedIcon from '@mui/icons-material/WaterDropOutlined';
import SettingsBrightnessOutlinedIcon from '@mui/icons-material/SettingsBrightnessOutlined';
import FilterBAndWOutlinedIcon from '@mui/icons-material/FilterBAndWOutlined';
import BlurCircularOutlinedIcon from '@mui/icons-material/BlurCircularOutlined';
import GpsFixedOutlinedIcon from '@mui/icons-material/GpsFixedOutlined';

const ModelCard = ({ title, name, metrics, isBest, icon: Icon, comingSoon }) => (
  <Paper className="glass-panel" sx={{ 
    p: 3, 
    height: '100%', 
    position: 'relative',
    borderColor: isBest ? 'var(--accent-purple)' : 'var(--border-color)',
    borderWidth: isBest ? 2 : 1,
    overflow: 'visible',
    display: 'flex',
    flexDirection: 'column'
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
    
    <Box sx={{ mb: 2 }}>
      {Icon && <Icon sx={{ fontSize: 32, color: 'var(--accent-purple)' }} />}
    </Box>

    <Typography variant="h5" sx={{ fontWeight: 'bold', mb: 0.5 }}>{title}</Typography>
    <Typography variant="body2" sx={{ color: 'var(--text-muted)', mb: 4, minHeight: 40 }}>{name}</Typography>

    {comingSoon ? (
      <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="body1" sx={{ color: 'var(--text-muted)', fontStyle: 'italic', opacity: 0.7 }}>
          Coming Soon
        </Typography>
      </Box>
    ) : (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>Accuracy</Typography>
          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{metrics.accuracy}</Typography>
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>Precision</Typography>
          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{metrics.precision}</Typography>
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>Recall</Typography>
          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{metrics.recall}</Typography>
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" sx={{ color: 'var(--text-muted)' }}>F1 Score</Typography>
          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{metrics.f1}</Typography>
        </Box>
      </Box>
    )}
  </Paper>
);

const ModelsPage = () => {
  return (
    <Box sx={{ maxWidth: 1200 }}>
      <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>Model Comparison</Typography>
      <Typography variant="body1" sx={{ color: 'var(--text-muted)', mb: 5 }}>
        Compare the performance of different experiments and preprocessing techniques.
      </Typography>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(5, 1fr)' }, gap: 3 }}>
        <ModelCard 
          title="EXP 1" 
          name="Baseline" 
          icon={WaterDropOutlinedIcon}
          metrics={{ accuracy: '94.00%', precision: '94.00%', recall: '94.00%', f1: '93.00%' }} 
          isBest
        />
        <ModelCard 
          title="EXP 2" 
          name="CLAHE + Gamma" 
          icon={SettingsBrightnessOutlinedIcon}
          comingSoon
        />
        <ModelCard 
          title="EXP 3" 
          name="Bilateral Filter" 
          icon={FilterBAndWOutlinedIcon}
          comingSoon
        />
        <ModelCard 
          title="EXP 4" 
          name="Unsharp Masking" 
          icon={BlurCircularOutlinedIcon}
          comingSoon
        />
        <ModelCard 
          title="EXP 5" 
          name="Hybrid (Best)" 
          icon={GpsFixedOutlinedIcon}
          comingSoon
        />
      </Box>
      
      <Typography variant="caption" sx={{ color: 'var(--text-muted)', display: 'block', mt: 6 }}>
        * Performance results are evaluated on the SSL400 validation set.
      </Typography>
    </Box>
  );
};

export default ModelsPage;
