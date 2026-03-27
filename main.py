import threading
import time
import cv2
import numpy as np
import screeninfo
from collections import deque
import argparse
import os

# Global variables (to be initialized in main)
frame_buffer = None
frame_buffer_lock = None
frame_rate = None
buffer_duration = None

def get_screen_resolution():
    screen_info = screeninfo.get_monitors()[0]
    return screen_info.width, screen_info.height

def save_last_x_seconds_of_video():
    with frame_buffer_lock:
        local_frame_buffer = list(frame_buffer)  # Copy buffer to avoid locking for too long

    if not local_frame_buffer:
        print("No frames to save.")
        return

    # Use the size of the first frame in the buffer
    frame_height, frame_width = local_frame_buffer[0].shape[:2]
    # Ensure output directory exists
    os.makedirs('./video', exist_ok=True)
    video_name = f'./video/video_{int(time.time())}.avi'

    try:
        # Check frame size
        if frame_width == 0 or frame_height == 0:
            print("Invalid frame size for video writer.")
            return
        # Use XVID codec for better compatibility
        video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))
        if not video_writer.isOpened():
            print(f"Failed to open video writer for {video_name}.")
            return

        for frame in local_frame_buffer:
            video_writer.write(frame)
    except Exception as e:
        print(f"Error while saving video: {e}")
    finally:
        if 'video_writer' in locals():
            video_writer.release()
        print("Video saved successfully")

# Add a global variable to track if dimensions have been printed
frame_dimensions_printed = False

# Global variable for the frame height and width
frame_height = 0
frame_width = 0

def capture_image(cap, target_width, target_height):
    global frame_dimensions_printed  # Access the global variable
    global frame_height, frame_width  # Access the global variables
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return None

    # Print the dimensions of the frame once
    if not frame_dimensions_printed:
        # print(f"Captured frame dimensions: Width = {frame.shape[1]}, Height = {frame.shape[0]}")
        frame_dimensions_printed = True  # Set the flag to True after printing
        


    # Resize frame while maintaining aspect ratio
    h, w = frame.shape[:2]
    aspect_ratio = w / h

    if target_width / target_height > aspect_ratio:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    
    frame_height = new_height
    frame_width = new_width
    

    return cv2.resize(frame, (new_width, new_height))





def save_video_thread():
    thread = threading.Thread(target=save_last_x_seconds_of_video)
    thread.daemon = True  # Daemon thread to ensure it doesn't block program exit
    thread.start()

def main(frame_rate_arg, buffer_duration_arg):
    global frame_buffer, frame_buffer_lock, frame_rate, buffer_duration

    frame_rate = frame_rate_arg
    buffer_duration = buffer_duration_arg
    frame_buffer = deque(maxlen=frame_rate * buffer_duration)  # Buffer to store last X seconds' worth of frames
    frame_buffer_lock = threading.Lock()  # Lock for accessing the frame buffer

    cap = cv2.VideoCapture(0)  # Open the default camera (change the index if you have multiple cameras)
    if not cap.isOpened():
        print("Failed to open the camera.")
        return
    # Print camera FPS
    print("Camera FPS:", cap.get(cv2.CAP_PROP_FPS))

    screen_width, screen_height = get_screen_resolution()
    real_fps_count = 0
    last_fps_update_time = time.time()

    try:
        # Make the window resizable
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        window_resized = False
        while True:
            local_start = time.time()
            frame = capture_image(cap, screen_width, screen_height)
            if frame is not None:
                with frame_buffer_lock:
                    frame_buffer.append(frame)  # Append current frame to buffer

                # Display the oldest frame in the buffer
                with frame_buffer_lock:
                    oldest_frame = frame_buffer[0] if frame_buffer else frame

                # Update real FPS counter
                real_fps_count += 1
                if time.time() - last_fps_update_time >= 1.0:
                    print(f"Current Real FPS: {real_fps_count}")
                    last_fps_update_time = time.time()
                    real_fps_count = 0

                # Define the width of the rectangle you want to add
                rectangle_width = int(frame_width/3)   # Adjust this value as needed

                # Create a new image with the wider width
                new_width = frame_width + rectangle_width
                new_height = frame_height

                # Create a blank image (black) with the new dimensions
                wider_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

                # Place the original frame on the right side of the new image
                wider_image[:, rectangle_width:] = oldest_frame

                # Draw a black rectangle on the left side of the new image
                cv2.rectangle(wider_image, (0, 0), (rectangle_width, new_height), (0, 0, 0), -1)  # Draw a black rectangle

                # Overlay the rectangle on the left side
                cv2.addWeighted(wider_image, 0.5, wider_image, 0.5, 0, wider_image)  # Optional: adjust visibility of the rectangle

                # Display Real and Target FPS on the new wider image
                cv2.putText(wider_image, f'Target FPS: {frame_rate}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(wider_image, f'Buffer Duration: {buffer_duration} sec', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display controls for the user
                control_text = (
                    # "+ : Augmenter FPS\n"
                    # "- : Diminuer FPS\n"
                    "+ : Augmenter la duree du delai\n"
                    "- : Diminuer la duree du delai\n"
                    "s : sauvegarder la video\n"
                    "q : Quitter"
                )

                # Display each control on a new line
                for i, line in enumerate(control_text.splitlines()):
                    cv2.putText(wider_image, line, (10, 170 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Set the window size to match the image only once
                if not window_resized:
                    cv2.resizeWindow("Frame", new_width, new_height)
                    window_resized = True
                # Show the new wider image
                cv2.imshow("Frame", wider_image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):
                save_video_thread()
            elif key & 0xFF == ord('+'):  # Increase buffer duration
                buffer_duration += 1
                frame_buffer = deque(maxlen=frame_rate * buffer_duration)  # Adjust buffer size
            elif key & 0xFF == ord('-'):  # Decrease buffer duration
                if buffer_duration > 1:  # Prevent buffer duration from going below 1
                    buffer_duration -= 1
                    frame_buffer = deque(maxlen=frame_rate * buffer_duration)  # Adjust buffer size

            # Maintain the target frame rate
            elapsed_time = time.time() - local_start
            time_to_wait = max(1 / frame_rate - elapsed_time, 0)
            time.sleep(time_to_wait)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture and save video frames.")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate for video capture.")
    parser.add_argument("--buffer_duration", type=int, default=5, help="Duration of the frame buffer in seconds.")

    args = parser.parse_args()

    main(args.frame_rate, args.buffer_duration)
