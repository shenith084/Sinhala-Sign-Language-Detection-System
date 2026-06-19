import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera, CameraOff } from 'lucide-react';

const TARGET_FRAMES = 32;
const FPS_CAPTURE = 15; // Target capture frame rate

export default function WebcamFeed({ isRunning, onBufferReady }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [hasPermission, setHasPermission] = useState(false);
  const [frameCount, setFrameCount] = useState(0);

  const frameBufferRef = useRef([]);
  const lastCaptureTimeRef = useRef(0);
  const animationFrameRef = useRef();

  useEffect(() => {
    if (isRunning) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [isRunning]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480, facingMode: "user" } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setHasPermission(true);
      captureLoop(0);
    } catch (err) {
      console.error("Camera access denied:", err);
      setHasPermission(false);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    cancelAnimationFrame(animationFrameRef.current);
    frameBufferRef.current = [];
    setFrameCount(0);
  };

  const captureLoop = useCallback((timestamp) => {
    if (!isRunning || !videoRef.current || !canvasRef.current) return;

    const timeDiff = timestamp - lastCaptureTimeRef.current;
    if (timeDiff > (1000 / FPS_CAPTURE)) {
      lastCaptureTimeRef.current = timestamp;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      const size = Math.min(video.videoWidth, video.videoHeight);
      if (size > 0) {
        const startX = (video.videoWidth - size) / 2;
        const startY = (video.videoHeight - size) / 2;

        ctx.drawImage(video, startX, startY, size, size, 0, 0, 224, 224);
        const base64Frame = canvas.toDataURL('image/jpeg', 0.7);

        frameBufferRef.current.push(base64Frame);
        if (frameBufferRef.current.length > TARGET_FRAMES) {
          frameBufferRef.current.shift();
        }

        setFrameCount(frameBufferRef.current.length);

        // Every 8 frames once buffer is full, send to parent
        if (frameBufferRef.current.length === TARGET_FRAMES) {
          onBufferReady([...frameBufferRef.current]);
        }
      }
    }
    animationFrameRef.current = requestAnimationFrame(captureLoop);
  }, [isRunning, onBufferReady]);

  return (
    <div className="relative w-full aspect-video bg-black/40 rounded-xl overflow-hidden border border-white/10 shadow-2xl">
      <video
        ref={videoRef}
        className="w-full h-full object-cover scale-x-[-1]"
        playsInline
        muted
      />
      <canvas ref={canvasRef} width="224" height="224" className="hidden" />

      {!isRunning && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400 bg-black/60 backdrop-blur-sm">
          <CameraOff size={48} className="mb-4 opacity-50" />
          <p className="text-lg font-medium tracking-wide">Camera is currently inactive</p>
        </div>
      )}

      {isRunning && (
        <div className="absolute bottom-4 left-4 right-4 flex justify-between">
          <div className="bg-black/50 backdrop-blur px-3 py-1.5 rounded-lg border border-white/10 text-sm">
            <span className="text-cyan-400 font-bold">{frameCount}</span>
            <span className="text-gray-400"> / {TARGET_FRAMES} buffer</span>
          </div>
          <div className="bg-black/50 backdrop-blur px-3 py-1.5 rounded-lg border border-white/10 text-sm flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
            <span className="text-gray-300">REC</span>
          </div>
        </div>
      )}
    </div>
  );
}
