import cv2
import os

# Set up the video path and output folder
video_path = "hardecore.mp4"
output_folder = "screenshots"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize variables
cap = cv2.VideoCapture(video_path)
frame_count = 0
roi_list = []
drawing = False
ix, iy = -1, -1
frame_resized = None
resize_ratio_x = 1
resize_ratio_y = 1


# Mouse callback function for drawing rectangles
def select_roi(event, x, y, flags, param):
    global ix, iy, drawing, frame, frame_resized, roi_list

    # Map the mouse coordinates back to the original frame size
    x_orig = int(x / resize_ratio_x)
    y_orig = int(y / resize_ratio_y)

    # Start drawing a rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_orig, y_orig

    # Update the rectangle during mouse movement
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Make a copy of the resized frame to display the rectangle without erasing previous ones
            img_copy = frame_resized.copy()
            cv2.rectangle(img_copy, (int(ix * resize_ratio_x), int(iy * resize_ratio_y)), (x, y), (0, 255, 0), 2)
            cv2.imshow('Frame', img_copy)

    # Finalize the rectangle on mouse release
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (ix, iy, x_orig, y_orig)  # Save the top-left and bottom-right coordinates in original scale

        # Ensure correct coordinates order (top-left to bottom-right)
        x1, y1, x2, y2 = min(ix, x_orig), min(iy, y_orig), max(ix, x_orig), max(iy, y_orig)

        # Ensure the coordinates are within frame bounds
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        y1, y2 = max(0, y1), min(frame.shape[0], y2)

        roi_list.append((x1, y1, x2, y2))

        # Draw the final rectangle on the resized frame
        cv2.rectangle(frame_resized, (int(x1 * resize_ratio_x), int(y1 * resize_ratio_y)),
                      (int(x2 * resize_ratio_x), int(y2 * resize_ratio_y)), (0, 255, 0), 2)
        cv2.imshow('Frame', frame_resized)


# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  # Allow window to be resizable
cv2.setMouseCallback('Frame', select_roi)

# Set the desired display size (e.g., 800x600)
display_width = 800
display_height = 600

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Finished processing video.")
        break

    roi_list = []  # Reset list of regions of interest (ROIs) for the new frame

    # Calculate the resize ratio between original frame and display size
    h, w, _ = frame.shape
    resize_ratio_x = display_width / w
    resize_ratio_y = display_height / h

    # Resize the frame to fit within the display window
    frame_resized = cv2.resize(frame, (display_width, display_height))

    while True:
        cv2.imshow('Frame', frame_resized)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Press 'c' to confirm selection(s)
            for i, roi in enumerate(roi_list):
                x1, y1, x2, y2 = roi
                cropped_img = frame[y1:y2, x1:x2]  # Crop from the original frame

                # Validate the cropped region (shouldn't be empty)
                if cropped_img.size > 0:
                    screenshot_filename = os.path.join(output_folder, f'frame_{frame_count}_roi_{i}.png')
                    cv2.imwrite(screenshot_filename, cropped_img)
                    print(f"Saved {screenshot_filename}")
                else:
                    print(f"Invalid ROI at frame {frame_count} skipped.")
            break  # Move to the next frame after confirming the selections

        elif key == ord('n'):  # Press 'n' to skip to the next frame without saving
            print("Skipped frame without saving.")
            break

        elif key == ord('q'):  # Press 'q' to quit early
            cap.release()
            cv2.destroyAllWindows()
            exit()

    frame_count += 1

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

print(f"Total frames processed: {frame_count}")
