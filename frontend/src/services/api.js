import axios from 'axios';

const API_BASE = 'http://localhost:5000/api';

export const checkStatus = async () => {
    try {
        const response = await axios.get(`${API_BASE}/metrics/status`);
        return response.data.status === 'online';
    } catch (error) {
        return false;
    }
};

export const fetchExperiments = async () => {
    try {
        const response = await axios.get(`${API_BASE}/experiments/`);
        return response.data.experiments;
    } catch (error) {
        console.error("Failed to fetch experiments:", error);
        return {};
    }
};

export const sendFramesForPrediction = async (framesBase64) => {
    try {
        const response = await axios.post(`${API_BASE}/predict`, { frames: framesBase64 });
        return response.data.predictions;
    } catch (error) {
        console.error("Prediction API error:", error);
        return [];
    }
};

export const getTtsAudioUrl = (text) => {
    return `${API_BASE}/tts?text=${encodeURIComponent(text)}`;
};
