import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const getExperiments = async () => {
  const response = await api.get('/experiments');
  return response.data;
};

export const getMetrics = async () => {
  const response = await api.get('/metrics');
  return response.data;
};

export const predictFrames = async (frames, exp_id = 5) => {
  const response = await api.post('/predict', {
    frames,
    exp_id
  });
  return response.data;
};

export default api;
