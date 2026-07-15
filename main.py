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

    print("Starting webcam... Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process the frame to detect hands and draw on the canvas
        img = tracker.process_frame(img)

        frame_count += 1

        # Show the combined image
        cv2.imshow("Vision Math Solver", img)

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
