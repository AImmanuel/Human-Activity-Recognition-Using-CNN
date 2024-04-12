import cv2

# Function to perform frame differencing
def frame_difference(prev_frame, current_frame):
    # Convert frames to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between frames
    frame_diff = cv2.absdiff(prev_frame_gray, current_frame_gray)

    # Apply threshold to highlight the differences
    _, thresholded_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    return thresholded_diff

# Open webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, prev_frame = cap.read()

while True:
    # Read the current frame
    ret, current_frame = cap.read()

    # Perform frame differencing
    diff_frame = frame_difference(prev_frame, current_frame)

    # Display the frames
    cv2.imshow('Original Frame', current_frame)
    cv2.imshow('Frame Difference', diff_frame)

    # Update the previous frame for the next iteration
    prev_frame = current_frame.copy()

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()