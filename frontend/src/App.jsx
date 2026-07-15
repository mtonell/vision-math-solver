import React, { useEffect, useRef, useState } from 'react';
import './App.css';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [formula, setFormula] = useState("");
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const processedImgRef = useRef(null);
  const wsRef = useRef(null);
  const isStreamingRef = useRef(false);

  // Initialize WebSocket connection to FastAPI
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
        // Fallback if not JSON
        if (processedImgRef.current && event.data !== "error") {
          processedImgRef.current.src = `data:image/jpeg;base64,${event.data}`;
        }
      }
      
      // Ping-pong streaming to prevent queueing lag!
      if (isStreamingRef.current) {
        requestAnimationFrame(captureAndSendFrame);
      }
    };
    
    wsRef.current.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from AI Backend');
      setTimeout(connectWebSocket, 3000); // Try to reconnect
    };
  };

  const startStreaming = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480, facingMode: 'user' } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      
      setIsStreaming(true);
      isStreamingRef.current = true;
      
      // Kick off the very first frame. The rest will ping-pong automatically.
      requestAnimationFrame(captureAndSendFrame);
      
    } catch (err) {
      console.error("Error accessing webcam:", err);
      alert("Could not access webcam. Please ensure permissions are granted in your browser.");
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
    
    // Only send frames if WebSocket is actually connected
    if (wsRef.current.readyState === WebSocket.OPEN) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      
      // Draw hidden video to hidden canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Extract as JPEG (quality 0.5 to drastically reduce payload size)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
      
      // Send to FastAPI
      wsRef.current.send(dataUrl);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1 className="title">AI Vision Math Solver</h1>
        <p className="subtitle">Draw math in the air. Let AI solve it.</p>
      </div>

      <div className="glass-panel">
        <div className="video-container">
          <div className="status-badge">
            <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
            {isConnected ? 'Backend Connected' : 'Connecting...'}
          </div>

          {/* Hidden elements for capturing frames */}
          <video ref={videoRef} className="hidden-webcam" playsInline muted></video>
          <canvas ref={canvasRef} className="hidden-webcam"></canvas>

          {/* The AI processed image stream from the backend */}
          <img 
            ref={processedImgRef} 
            className="video-feed" 
            alt="AI Feed"
            style={{ display: isStreaming ? 'block' : 'none' }}
          />

          {/* Formula Bar */}
          {isStreaming && formula && (
            <div className="formula-bar">
              {formula}
            </div>
          )}

          {!isStreaming && (
            <div style={{
              width: '100%', height: '100%', display: 'flex', 
              justifyContent: 'center', alignItems: 'center', color: '#8a93a6',
              fontSize: '1.2rem'
            }}>
              Click Start to access webcam
            </div>
          )}
        </div>

        <div className="controls">
          {!isStreaming ? (
            <button className="btn primary" onClick={startStreaming}>
              Start Camera
            </button>
          ) : (
            <button className="btn" onClick={stopStreaming}>
              Stop Camera
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
