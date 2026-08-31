import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, Box } from '@mui/material';

import Sidebar from './components/Sidebar';
import MobileNav from './components/MobileNav';
import HomePage from './pages/HomePage';
import LiveDetectionPage from './pages/LiveDetectionPage';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#8b5cf6' }, // Neon purple
    secondary: { main: '#c084fc' },
    background: { default: '#0a0b10', paper: '#13141f' },
    text: { primary: '#f8fafc', secondary: '#94a3b8' }
  },
  typography: {
    fontFamily: '"Inter", system-ui, -apple-system, sans-serif',
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: '8px',
          fontWeight: 600,
        }
      }
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none', // Remove MUI's default elevation overlay
          border: '1px solid #2e303a'
        }
      }
    }
  }
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', width: '100vw', minHeight: '100vh', flexDirection: { xs: 'column', md: 'row' } }}>
          {/* Persistent Sidebar (Hidden on mobile) */}
          <Sidebar />

          {/* Main Content Area */}
          <Box component="main" sx={{ flexGrow: 1, p: { xs: 2, md: 4 }, overflowY: 'auto', pb: { xs: 10, md: 4 } }}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/live" element={<LiveDetectionPage />} />
            </Routes>
          </Box>
          
          {/* Mobile Bottom Navigation (Hidden on desktop) */}
          <MobileNav />
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
