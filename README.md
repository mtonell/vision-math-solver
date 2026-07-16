# Vision Math Solver 🚀

> **[Insert Main Project Demo GIF/Video Here]**  
> *(Tip: Record a 15-second clip of you drawing an equation, zooming in on the result, and grabbing a number to drag it. Replace this text with `![Demo](path_to_video.gif)`)*

Vision Math Solver is an interactive, real-time Computer Vision application that allows you to draw mathematical equations in the air using your hand, parse them using a custom Convolutional Neural Network (CNN), and solve them instantly on your screen.

## ✨ Features
* **Air Drawing:** Hold your index finger up to draw math equations directly onto your camera feed.
* **Spatial Parsing & CNN Inference:** A custom-trained CNN categorizes your hand-drawn digits and mathematical symbols, intelligently grouping them using 2D spatial bounding box algorithms.
* **Pinch-to-Drag:** Grab parsed numbers with your thumb and index finger to move them around the virtual canvas!
* **Full-Stack Architecture:** 
  * 🎨 **Frontend:** React (Vite) interface running on a highly optimized Nginx web server.
  * ⚙️ **Backend:** FastAPI handling real-time WebSockets, OpenCV image processing, and SymPy math evaluation.

---

## 👁️ AI Vision Mode

Want to see exactly how the AI interprets your hand-drawn math? 

Toggle the **👁️ AI Vision** button in the web UI to instantly reveal the underlying Neural Network bounding boxes, confidence score percentages, and the 3D MediaPipe hand tracking skeleton layered directly over the camera feed.

> **[Insert AI Vision Demo GIF/Video Here]**  
> *(Tip: Record a 10-second clip of you clicking the AI Vision button and drawing a number so the green bounding boxes appear. Replace this text with `![AI Vision](path_to_video.gif)`)*

---

## ✋ Gestures
* **Right Index Finger (👆):** Draw ink on the canvas.
* **Right Thumbs Up (👍):** Save the current equation to your history sidebar (Hold for 1 second).
* **Right Pinch (🤏):** Grab a parsed bounding box and drag it around.
* **Left Index Finger (👆):** Precision Eraser.
* **Left Closed Fist (✊):** Large Area Eraser.
* **Left Hand Swipe (👋):** Clear the entire canvas.
* **Open Hand (🖐️):** Neutral position (No drawing/erasing).

---

## 🛠️ Tech Stack
* **Computer Vision:** OpenCV, MediaPipe (Hand Tracking)
* **AI / Deep Learning:** ONNX Runtime, PyTorch (Custom CNN Model)
* **Backend:** Python, FastAPI, Uvicorn, WebSockets, SymPy
* **Frontend:** React, JavaScript, Vite, CSS
* **Deployment:** Docker, Docker Compose, Nginx (Multi-Stage Builds)

---

## 🚀 Quick Start (Docker)
The easiest way to run the application is using the provided Docker containers. The backend has been optimized for CPU-only ONNX inference, and the frontend uses an ultra-lightweight multi-stage Nginx build.

1. Ensure you have Docker Desktop installed.
2. Clone this repository.
3. Run the following command in the root directory:
```bash
docker compose up --build
```
4. Open your browser and navigate to `http://localhost:5173`.

---

## 💻 Manual Setup (Standalone OpenCV Demo)
If you prefer not to use Docker or React, you can run the standalone OpenCV native demo script (`main.py`) which draws the UI natively over your webcam feed.

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Standalone Demo
python main.py
```
*(Press `q` to quit the OpenCV window)*

---

## 🧠 Training the CNN (Advanced)
By default, the application runs a highly optimized ONNX binary (`cnn_model.onnx`) for blazing-fast inference, which completely removes the need for PyTorch in production. 

If you wish to re-train the neural network yourself or modify the architecture, you will need to install the heavy Machine Learning libraries manually:

```bash
# 1. Install ML Engineering dependencies
pip install torch torchvision onnx onnxscript

# 2. Train the model (Outputs cnn_model.pth)
python training/train_model.py

# 3. Export the newly trained PyTorch model back to ONNX for production
python convert_to_onnx.py
```

---

## 🔮 Future Works
While the current version of Vision Math Solver is highly capable, there are several exciting avenues planned for future development:
* **Advanced Neural Architecture:** Upgrade the custom CNN to a deeper, more robust architecture for improved handwriting generalization.
* **Expanded Symbol Library:** Add support for complex mathematical symbols such as parentheses `()`, equals `=`, exponents `^`, and square roots `√`.
* **Complex Equations:** Enhance the 2D spatial parser to understand multi-line equations, fractions, and variables.
* **Vision Transformers (ViT):** Transition from a character-by-character bounding box approach to an end-to-end Vision Transformer (ViT) capable of reading entire mathematical formulas directly into LaTeX.
