import React from 'react';

const HelpTooltip = () => {
  return (
    <div className="help-btn-container">
      <button className="btn help-btn">?</button>
      <div className="help-tooltip">
        <div className="help-section">
          <div className="help-title r-hand">Right ✋</div>
          <div className="emoji-row"><span>Draw</span> <span>👆</span></div>
          <div className="emoji-row"><span>Save</span> <span>👍</span></div>
          <div className="emoji-row"><span>Drag</span> <span>🤏</span></div>
        </div>
        <div className="help-section">
          <div className="help-title l-hand">Left 🤚</div>
          <div className="emoji-row"><span>Erase</span> <span>👆 / ✊</span></div>
          <div className="emoji-row"><span>Clear</span> <span>👋</span></div>
        </div>
        <div className="help-section">
          <div className="help-title n-hand">Neutral ⏸️</div>
          <div className="emoji-row"><span>Open Hand</span> <span>🖐️</span></div>
        </div>
      </div>
    </div>
  );
};

export default HelpTooltip;
