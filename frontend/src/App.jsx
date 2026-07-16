import React, { useEffect, useRef, useState } from 'react';
import LandingPage from './components/LandingPage';
import HistorySidebar from './components/HistorySidebar';
import HelpTooltip from './components/HelpTooltip';
import './App.css';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAppStarted, setIsAppStarted] = useState(false);
  const [formula, setFormula] = useState("");
  const [history, setHistory] = useState([]);
  const [isVisionSelected, setIsVisionSelected] = useState(false);
  const [showSavedToast, setShowSavedToast] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const processedImgRef = useRef(null);
  const wsRef = useRef(null);
  const isStreamingRef = useRef(false);
  const lastSavedRef = useRef("");

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
      stopStreaming();
    };
  }, []);

  const connectWebSocket = () => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws/video');
    
    wsRef.current.onopen = () => {
      setIsConnected(true);
      console.log('Connected to AI Backend');
    };
    
    wsRef.current.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.error) {
          console.error("Backend error:", payload.error);
        } else {
          if (processedImgRef.current) {
            processedImgRef.current.src = `data:image/jpeg;base64,${payload.image}`;
          }
          
          let displayFormula = payload.equation;
          if (payload.result) {
            displayFormula += ` ${payload.result}`;
          }
          setFormula(displayFormula);
          
          if (payload.save && displayFormula && displayFormula !== lastSavedRef.current) {
            lastSavedRef.current = displayFormula;
            setHistory(prev => [...prev, displayFormula]);
            setShowSavedToast(true);
            setTimeout(() => setShowSavedToast(false), 2000);
          }
        }
      } catch(e) {
        if (processedImgRef.current && event.data !== "error") {
          processedImgRef.current.src = `data:image/jpeg;base64,${event.data}`;
        }
      }
      
      if (isStreamingRef.current) {
        requestAnimationFrame(captureAndSendFrame);
      }
    };
    
    wsRef.current.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from AI Backend');
      setTimeout(connectWebSocket, 3000);
    };
  };

  const handleStart = () => {
    setIsAppStarted(true);
    startStreaming();
  };

  const handleStop = () => {
    setIsAppStarted(false);
    stopStreaming();
  };

  const startStreaming = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480, facingMode: 'user' } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setIsStreaming(true);
          isStreamingRef.current = true;
          requestAnimationFrame(captureAndSendFrame);
        };
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      alert("Could not access webcam. Please ensure permissions are granted in your browser.");
      setIsAppStarted(false);
    }
  };

  const stopStreaming = () => {
    isStreamingRef.current = false;
    setIsStreaming(false);
    
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }
  };

  const clearBoard = () => {
    setHistory([]);
    lastSavedRef.current = "";
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "clear" }));
    }
  };

  const toggleDebug = () => {
    setIsVisionSelected(prev => !prev);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "toggle_debug" }));
    }
  };

  const captureAndSendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    
    if (wsRef.current.readyState === WebSocket.OPEN) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
      wsRef.current.send(dataUrl);
    }
  };

  return (
    <div className={`app-container ${isAppStarted ? 'demo-mode' : ''}`}>
      {!isAppStarted ? (
        <LandingPage isConnected={isConnected} onStart={handleStart} />
      ) : (
        <div className="demo-container fade-in">
          <div className="top-controls-container" style={{ position: 'absolute', top: '20px', left: '20px', zIndex: 100, display: 'flex', gap: '10px' }}>
            <button className="btn back-btn" onClick={handleStop} style={{ position: 'relative', top: '0', left: '0' }}>
              ← Back
            </button>
            <button 
              className="btn debug-btn" 
              onClick={toggleDebug} 
              style={{ 
                padding: isVisionSelected ? '12px 24px' : '12px 16px', 
                fontSize: '1.2rem', 
                fontWeight: 'bold', 
                background: isVisionSelected ? '#00a8ff' : 'var(--card-bg)', 
                border: '2px solid rgba(255, 255, 255, 0.1)', 
                color: 'white', 
                borderRadius: '12px', 
                cursor: 'pointer', 
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)', 
                boxShadow: isVisionSelected ? '0 0 20px rgba(0, 168, 255, 0.6)' : '0 4px 15px rgba(0, 0, 0, 0.3)',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                overflow: 'hidden',
                whiteSpace: 'nowrap'
              }} 
              title="Toggle AI Vision Mode"
            >
              <span>👁️</span>
              {isVisionSelected && <span>AI Vision</span>}
            </button>
          </div>
          
          <HelpTooltip />
          
          <div className="video-wrapper">
            <div className="status-badge">
              <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
              {isConnected ? 'Backend Connected' : 'Connecting...'}
            </div>

            <video ref={videoRef} className="hidden-webcam" playsInline muted></video>
            <canvas ref={canvasRef} className="hidden-webcam"></canvas>

            <img 
              ref={processedImgRef} 
              className="video-feed-full" 
              alt="AI Feed"
              style={{ display: isStreaming ? 'block' : 'none' }}
            />

            {isStreaming && formula && (
              <div className="formula-bar-full">
                {formula}
              </div>
            )}

            {!isStreaming && (
              <div className="waking-up">
                Waking up AI model...
              </div>
            )}
          </div>

          <HistorySidebar history={history} onClearBoard={clearBoard} />
          
          {showSavedToast && (
            <div style={{ 
              position: 'absolute', 
              bottom: '20px', 
              right: '30px', 
              background: '#00c853', 
              color: 'white', 
              padding: '12px 24px', 
              borderRadius: '12px', 
              fontWeight: 'bold', 
              fontSize: '1.2rem', 
              boxShadow: '0 4px 15px rgba(0, 0, 0, 0.3)', 
              zIndex: 1000,
              animation: 'fadeIn 0.3s ease-out'
            }}>
              ✓ Saved!
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
