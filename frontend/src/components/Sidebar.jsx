import React from 'react';
import { Box, Typography, List, ListItem, ListItemIcon, ListItemText, ListItemButton } from '@mui/material';
import { useLocation, useNavigate } from 'react-router-dom';
import HomeOutlinedIcon from '@mui/icons-material/HomeOutlined';
import VideocamOutlinedIcon from '@mui/icons-material/VideocamOutlined';
import AutoGraphOutlinedIcon from '@mui/icons-material/AutoGraphOutlined';
import TimelineOutlinedIcon from '@mui/icons-material/TimelineOutlined';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import PanToolIcon from '@mui/icons-material/PanTool'; // For the hand logo

const Sidebar = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const menuItems = [
    { text: 'Home', icon: <HomeOutlinedIcon />, path: '/' },
    { text: 'Live Detection', icon: <VideocamOutlinedIcon />, path: '/live' },
    { text: 'Models', icon: <AutoGraphOutlinedIcon />, path: '/models' },
    { text: 'Results', icon: <TimelineOutlinedIcon />, path: '/results' },
    { text: 'About', icon: <InfoOutlinedIcon />, path: '/about' }
  ];

  return (
    <Box
      sx={{
        width: 250,
        flexShrink: 0,
        borderRight: '1px solid var(--border-color)',
        bgcolor: 'var(--bg-dark)',
        display: { xs: 'none', md: 'flex' },
        flexDirection: 'column',
        p: 2
      }}
    >
      {/* Logo Area */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 4, mt: 1, pl: 1 }}>
        <PanToolIcon sx={{ color: 'var(--accent-purple)', fontSize: 32, mr: 1.5 }} />
        <Typography variant="h6" sx={{ fontWeight: 600, color: '#fff', fontSize: '1.1rem' }}>
          SSL400 System
        </Typography>
      </Box>

      {/* Navigation List */}
      <List sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <ListItem disablePadding key={item.text}>
              <ListItemButton
                onClick={() => navigate(item.path)}
                sx={{
                  borderRadius: 2,
                  py: 1.2,
                  bgcolor: isActive ? 'var(--accent-purple)' : 'transparent',
                  color: isActive ? '#fff' : 'var(--text-muted)',
                  '&:hover': {
                    bgcolor: isActive ? 'var(--accent-purple-hover)' : 'var(--bg-surface)',
                    color: '#fff'
                  }
                }}
              >
                <ListItemIcon 
                  sx={{ 
                    color: isActive ? '#fff' : 'var(--text-muted)',
                    minWidth: 40 
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText 
                  primary={item.text} 
                  slotProps={{ 
                    primary: {
                      sx: {
                        fontWeight: isActive ? 600 : 500,
                        fontSize: '0.95rem'
                      }
                    }
                  }} 
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
    </Box>
  );
};

export default Sidebar;
