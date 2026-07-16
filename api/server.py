import sys
import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Ensure we can import the core module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.hand_tracker import HandTracker

app = FastAPI(title="Vision Math Solver API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the global tracker instance
tracker = HandTracker()

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("🚀 VISION MATH SOLVER IS SUCCESSFULLY RUNNING!")
    print("🌐 Open your browser and navigate to: http://localhost:5173")
    print("="*60 + "\n")

@app.get("/")
def read_root():
    return {"status": "Backend is running!"}

@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("React Client connected to video stream!")
    try:
        while True:
            try:
                # Receive base64 jpeg from React frontend
                data = await websocket.receive_text()
                
                # Check for UI Commands (like Clear Board)
                if data.startswith("{"):
                    import json
                    cmd = json.loads(data)
                    if cmd.get("action") == "clear":
                        if hasattr(tracker, 'canvas') and tracker.canvas is not None:
                            tracker.canvas.fill(0)
                        if hasattr(tracker, 'pts'):
                            tracker.pts.clear()
                        tracker.prediction_cache.clear()
                    elif cmd.get("action") == "toggle_debug":
                        tracker.debug = not tracker.debug
                    continue
                
                if "," in data:
                    base64_data = data.split(",")[1]
                else:
                    base64_data = data
                    
                # Decode base64 to OpenCV image
                img_bytes = base64.b64decode(base64_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    import json
                    # Run the exact same AI logic from our prototype!
                    result = tracker.process_frame(img)
                    
                    # Handle both old and new return types gracefully
                    if isinstance(result, tuple):
                        if len(result) == 4:
                            processed_img, eq_str, res_str, save_trig = result
                        elif len(result) == 3:
                            processed_img, eq_str, res_str = result
                            save_trig = False
                    else:
                        processed_img, eq_str, res_str, save_trig = result, "", "", False
                        
                    # Encode the processed frame back to base64
                    _, buffer = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    b64_img = base64.b64encode(buffer).decode('utf-8')
                    
                    # Package it in JSON
                    payload = {
                        "image": b64_img,
                        "equation": eq_str.replace('*', 'x'),
                        "result": res_str,
                        "save": save_trig
                    }
                    
                    # Send it back to React
                    await websocket.send_text(json.dumps(payload))
                else:
                    import json
                    # If decoding fails, send original back to prevent ping-pong freeze
                    await websocket.send_text(json.dumps({"image": base64_data, "equation": "", "result": ""}))
            
            except Exception as e:
                print(f"Frame Processing Error: {e}")
                # Ensure the loop continues and ping-pong doesn't break
                import json
                await websocket.send_text(json.dumps({"error": str(e)}))
                
    except WebSocketDisconnect:
        print("React Client disconnected.")
