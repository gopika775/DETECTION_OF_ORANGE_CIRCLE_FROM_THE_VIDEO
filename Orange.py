import cv2
import numpy as np

# Step 1: Load the video and create a VideoCapture object
video_path = r'C:\Users\PC\Desktop\Orange_Circle\rumbleverse all posiitives Good triggers.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Step 2: Read each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Step 3: Preprocess the frame (e.g., resize, convert to appropriate color space)

    # Step 4: Color Filtering (Convert to HSV and threshold on orange color range)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    # Step 5: Apply bitwise AND to extract only the orange regions
    orange_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Step 6: Detect the orange circle (e.g., using Hough Circle Transform)
    # Replace the values with appropriate parameters for your video
    image_gray = cv2.cvtColor(orange_frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=100)  # Increased maxRadius to 100

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # Convert the coordinates to integers
        
        # Step 7: Select the circle with the largest radius and draw a yellow bounding box around it
        max_circle = max(circles, key=lambda x: x[2])
        center = (max_circle[0], max_circle[1])
        radius = max_circle[2]
        x, y = center[0] - radius, center[1] - radius
        w, h = 2 * radius, 2 * radius
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)  # Yellow bounding box
        cv2.putText(frame, 'Circle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

    # Step 8: Display the processed frame
    cv2.imshow('Orange Circle Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the display window
cap.release()
cv2.destroyAllWindows()


