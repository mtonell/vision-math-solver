import cv2
import argparse
import time
from core.hand_tracker import HandTracker

def main():
    parser = argparse.ArgumentParser(description="Vision Math Solver")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (show skeleton, print stats)')
    args = parser.parse_args()

    # Initialize the hand tracker module
    tracker = HandTracker(debug=args.debug)
    
    # Initialize video capture (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting webcam... Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()
    saved_equations = []

    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process the frame to detect hands and draw on the canvas
        img, equation_str, res_str, save_triggered = tracker.process_frame(img)
        
        display_text = f"{equation_str} {res_str}".strip()
        
        if save_triggered and display_text:
            if display_text not in saved_equations:
                saved_equations.append(display_text)

        h, w = img.shape[:2]
        
        # Draw History Sidebar
        if saved_equations:
            sidebar_w = 300
            overlay = img.copy()
            cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (20, 14, 11), cv2.FILLED)
            cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
            
            cv2.putText(img, "History", (w - sidebar_w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.line(img, (w - sidebar_w + 20, 70), (w - 20, 70), (100, 100, 100), 2)
            
            for i, eq in enumerate(saved_equations[-15:]): # Show last 15
                cv2.putText(img, eq, (w - sidebar_w + 20, 120 + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Draw Bottom Equation Bar
        bar_h = 80
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 14, 11), cv2.FILLED)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        if not display_text:
            display_text = "Draw an equation..."
            color = (150, 150, 150)
        else:
            color = (255, 255, 255)
            
        cv2.putText(img, display_text, (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        frame_count += 1

        # Show the combined image
        cv2.imshow("Vision Math Solver Demo", img)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    total_time = end_time - start_time
    
    # Print debug stats when exiting
    if total_time > 0:
        mean_fps = frame_count / total_time
        print("\n--- Debug Stats ---")
        print(f"Total Frames: {frame_count}")
        print(f"Total Time:   {total_time:.2f} seconds")
        print(f"Mean Speed:   {mean_fps:.2f} FPS")
        print("-------------------")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
