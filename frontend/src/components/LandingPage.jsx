import React from 'react';

const LandingPage = ({ isConnected, onStart }) => {
  return (
    <>
      <div className="header">
        <h1 className="title">AI Vision Math Solver</h1>
        <p className="subtitle">Draw math in the air. Let me solve it.</p>
        
        <button className="btn primary massive-btn" onClick={onStart} style={{ marginTop: '2.5rem' }}>
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
  );
};

export default LandingPage;
