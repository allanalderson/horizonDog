import cv2
import numpy as np


smoothing_window = 5  # Moving average. Typ: 5
lower_blue = np.array([80, 10, 10]) # color in HSV
upper_blue = np.array([120, 255, 255]) # color in HSV
potential_edge = 200
strong_edge = 200
angle_buffer = []
smoothed_average_angle = 0
desired_width = 300
def rotate_image(image, angle):
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1)
	rotated_image = cv2.warpAffine(image, M, (w, h))
	return rotated_image


# Load the video capture
cap = cv2.VideoCapture(0)
while True:
	ret, original_frame = cap.read()
	# Resize the frame to half its size
	desired_height = int(desired_width * 9 / 16)

	small_frame = cv2.resize(original_frame, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
	if not ret:
		break
	# Convert BGR to HSV
	hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
	# Threshold the HSV frame to get only blue colors
	mask = cv2.inRange(hsv, lower_blue, upper_blue)
	# Bitwise-AND mask and original frame
	blue_areas = cv2.bitwise_and(small_frame, small_frame, mask=mask)
	# Convert frame to grayscale
	gray = cv2.cvtColor(blue_areas, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, potential_edge, strong_edge)
	# cv2.imshow('Edges Window', edges)
	# Perform Hough Transform to detect lines
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=30, maxLineGap=150)
	# Visualize edge detection and Hough Transform
	if lines is not None:
		hough_frame = small_frame.copy()  # Create a copy of the original frame for visualization
		hough_frame[:] = 0  # Set all pixels to black
		for line in lines:
			x1, y1, x2, y2 = line[0]
			cv2.line(hough_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Draw detected lines on the copy of the frame
		cv2.imshow('Hough Transform Window', hough_frame)


	# Avarage the detected lines
	if lines is not None:
		total_angle = 0
		num_lines = 0
		total_x1, total_y1, total_x2, total_y2 = 0, 0, 0, 0
		num_lines = len(lines)
		for line in lines:
			x1, y1, x2, y2 = line[0]
			total_x1 += x1
			total_y1 += y1
			total_x2 += x2
			total_y2 += y2
		avg_x1 = total_x1 / num_lines
		avg_y1 = total_y1 / num_lines
		avg_x2 = total_x2 / num_lines
		avg_y2 = total_y2 / num_lines

	# Check if avg line is approximately horizontal
		if abs(avg_y2 - avg_y1) < 180:
			angle = np.arctan2(avg_y2 - avg_y1, avg_x2 - avg_x1) * 180 / np.pi
			angle_buffer.append(angle)
			if len(angle_buffer) > smoothing_window:
				angle_buffer.pop(0)
			# Calculate the average angle from the buffer
			smoothed_average_angle = sum(angle_buffer) / len(angle_buffer)
	else:
		print("No lines ")
		smoothed_average_angle = 0

	rotated_frame = rotate_image(original_frame, smoothed_average_angle)
	# Display the rotated frame
	cv2.imshow('Result', rotated_frame)
	# Break the loop if 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
