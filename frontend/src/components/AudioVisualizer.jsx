import React, { useEffect, useRef } from 'react';

const AudioVisualizer = ({ isRecording }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);
  const sourceRef = useRef(null);

  useEffect(() => {
    if (isRecording) {
      startVisualizing();
    } else {
      stopVisualizing();
    }
    return () => stopVisualizing();
  }, [isRecording]);

  const startVisualizing = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      sourceRef.current.connect(analyserRef.current);

      analyserRef.current.fftSize = 256;
      const bufferLength = analyserRef.current.frequencyBinCount;
      dataArrayRef.current = new Uint8Array(bufferLength);

      draw();
    } catch (err) {
      console.error("Error accessing microphone for visualization:", err);
    }
  };

  const stopVisualizing = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  };

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    animationRef.current = requestAnimationFrame(draw);
    analyserRef.current.getByteFrequencyData(dataArrayRef.current);

    ctx.clearRect(0, 0, width, height);
    
    const barWidth = (width / dataArrayRef.current.length) * 2.5;
    let barHeight;
    let x = 0;

    for (let i = 0; i < dataArrayRef.current.length; i++) {
      barHeight = (dataArrayRef.current[i] / 255) * height;

      const r = 16 + (i * 2);
      const g = 185;
      const b = 129;

      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${dataArrayRef.current[i] / 255})`;
      
      // Draw rounded bars
      const barX = x;
      const barY = height - barHeight;
      const radius = 2;
      
      ctx.beginPath();
      ctx.roundRect(barX, barY, barWidth - 1, barHeight, radius);
      ctx.fill();

      x += barWidth + 1;
    }
  };

  return (
    <canvas 
      ref={canvasRef} 
      width={400} 
      height={80}
      className="w-full h-16 opacity-80"
    />
  );
};

export default AudioVisualizer;
