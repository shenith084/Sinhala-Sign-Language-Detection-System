import React from 'react';
import { Paper, BottomNavigation, BottomNavigationAction } from '@mui/material';
import { useLocation, useNavigate } from 'react-router-dom';
import HomeOutlinedIcon from '@mui/icons-material/HomeOutlined';
import VideocamOutlinedIcon from '@mui/icons-material/VideocamOutlined';

const MobileNav = () => {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <Paper 
      sx={{ 
        position: 'fixed', 
        bottom: 0, 
        left: 0, 
        right: 0, 
        display: { xs: 'block', md: 'none' },
        zIndex: 1000,
        borderTop: '1px solid var(--border-color)',
        bgcolor: 'var(--bg-dark)'
      }} 
      elevation={3}
    >
      <BottomNavigation
        showLabels
        value={location.pathname}
        onChange={(event, newValue) => {
          navigate(newValue);
        }}
        sx={{ 
          bgcolor: 'transparent',
          '& .Mui-selected': {
            color: 'var(--accent-purple)'
          },
          '& .MuiBottomNavigationAction-root': {
            color: 'var(--text-muted)'
          }
        }}
      >
        <BottomNavigationAction label="Home" value="/" icon={<HomeOutlinedIcon />} />
        <BottomNavigationAction label="Live" value="/live" icon={<VideocamOutlinedIcon />} />
      </BottomNavigation>
    </Paper>
  );
};

export default MobileNav;
