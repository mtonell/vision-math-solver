import React from 'react';

const HistorySidebar = ({ history, onClearBoard }) => {
  return (
    <div className="demo-sidebar">
      <button className="empty-btn" onClick={onClearBoard}>
        🗑️ Empty
      </button>
      <h3>History</h3>
      <div className="history-list">
        {history.length === 0 ? (
          <p className="empty-history">Give a 👍 to save an equation!</p>
        ) : (
          history.map((eq, i) => <div key={i} className="history-item">{eq}</div>)
        )}
      </div>
    </div>
  );
};

export default HistorySidebar;
