import React, { useEffect, useRef, useState } from 'react';
import './App.css';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAppStarted, setIsAppStarted] = useState(false);
  const [formula, setFormula] = useState("");
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const processedImgRef = useRef(null);
  const wsRef = useRef(null);
  const isStreamingRef = useRef(false);

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
        <>
          <div className="header">
            <h1 className="title">AI Vision Math Solver</h1>
            <p className="subtitle">Draw math in the air. Let me solve it.</p>
            
            <button className="btn primary massive-btn" onClick={handleStart} style={{ marginTop: '2.5rem' }}>
              {isConnected ? "Try it out" : "Connecting to AI..."}
            </button>
          </div>

          <div className="home-page">
            <div className="instructions">
              <div className="instruction-card">
                <h3>✍️ Draw</h3>
                <p>Hold up your <strong>right index finger</strong> to draw numbers and math operators.</p>
              </div>
              <div className="instruction-card">
                <h3>🤏 Drag & Drop</h3>
                <p><strong>Pinch your right thumb and index finger</strong> together to grab a number and move it.</p>
              </div>
              <div className="instruction-card">
                <h3>🧽 Erase & Clear</h3>
                <div style={{ textAlign: 'left', marginTop: '0.8rem', fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-secondary)' }}>
                  <strong>• Index Finger:</strong> Precision brush<br/>
                  <strong>• Closed Fist:</strong> Large eraser<br/>
                  <strong>• Hand Swipe:</strong> Clear all
                </div>
              </div>
              <div className="instruction-card">
                <h3>✋ Neutral Position</h3>
                <p>For both hands, an <strong>open hand</strong> is the neutral position. The AI won't draw or erase.</p>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="demo-container fade-in">
          <button className="btn back-btn" onClick={handleStop}>
            ← Back
          </button>
          
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
        </div>
      )}
    </div>
  );
}

export default App;
