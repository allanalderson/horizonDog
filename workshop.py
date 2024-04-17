import cv2
import numpy as np

scale = .7 # image size
smoothing_window = 5  # Moving average. Typ: 5
lower_blue = np.array([90, 20, 20]) # color in HSV
upper_blue = np.array([110, 255, 255]) # color in HSV
potential_edge = 5
strong_edge = 95
angle_buffer = []
smoothed_average_angle = 0
computational_window_width = 600 # Default 600
def rotate_image(image, angle):
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated_image = cv2.warpAffine(image, M, (w, h))
	return rotated_image


# Load the video capture
cap = cv2.VideoCapture(0)
while True:
	ret, original_frame = cap.read()
	# Resize the frame to half its size
	desired_height = int(computational_window_width * 9 / 16)
	small_frame = cv2.resize(original_frame, (computational_window_width, desired_height), interpolation=cv2.INTER_LINEAR)
	if not ret:
		break
	# Convert BGR to HSV
	hsv_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
	mask1 = cv2.inRange(hsv_frame, lower_blue, upper_blue) # Threshold the HSV frame to get only blue colors
	duplicated_mask1 = mask1.copy()
	rows, cols = duplicated_mask1.shape
	M = np.float32([[1, 0, 0], [0, 1, -10]])  # Translation matrix for 10 pixels down
	translated_mask1 = cv2.warpAffine(duplicated_mask1, M, (cols, rows))
	# Add the translated mask to the original mask
	mask2 = cv2.bitwise_and(mask1, translated_mask1)
	# Bitwise-AND mask and original frame
	blue_areas = cv2.bitwise_and(small_frame, small_frame, mask=mask2)
	# Convert frame to grayscale
	gray = cv2.cvtColor(blue_areas, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, potential_edge, strong_edge)
	cv2.imshow('Edges Window', mask2)
	# Perform Hough Transform to detect lines
	hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 65, minLineLength=100, maxLineGap=100)
	# Visualize edge detection and Hough Transform
	if hough_lines is not None:
		hough_frame = small_frame.copy()  # Create a copy of the original frame for visualization
		# hough_frame[:] = 0  # Set all pixels to black
		for line in hough_lines:
			x1, y1, x2, y2 = line[0]
			cv2.line(hough_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Draw detected lines on the copy of the frame
		cv2.imshow('Hough Transform Window', hough_frame)
		# calculate an average line from the lines
		total_angle = 0
		num_lines = 0
		total_x1, total_y1, total_x2, total_y2 = 0, 0, 0, 0
		num_lines = len(hough_lines)
		for line in hough_lines:
			x1, y1, x2, y2 = line[0]
			total_x1 += x1
			total_y1 += y1
			total_x2 += x2
			total_y2 += y2
		avg_x1 = total_x1 / num_lines
		avg_y1 = total_y1 / num_lines
		avg_x2 = total_x2 / num_lines
		avg_y2 = total_y2 / num_lines
		angle = np.arctan2(avg_y2 - avg_y1, avg_x2 - avg_x1) * 180 / np.pi
		angle_buffer.append(angle)
		if len(angle_buffer) > smoothing_window:
			angle_buffer.pop(0)
		# Calculate the average angle from the buffer
		smoothed_average_angle = sum(angle_buffer) / len(angle_buffer)
	else:
		# print("No lines ")
		smoothed_average_angle = 0
	# rotate the frame
	rotated_frame = rotate_image(original_frame, smoothed_average_angle)

	prev_frame = None
	prev_points = None


	# Convert the frame to grayscale
	gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

	# Perform motion estimation between current and previous frame
	if prev_frame is not None:
		flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		flow = -flow

		# Apply the motion to the frame
		h, w = flow.shape[:2]
		flow_map = np.column_stack((flow[:, :, 0] + np.arange(w), flow[:, :, 1] + np.arange(h)[:, np.newaxis])).astype(np.float32)
		stabilized_frame = cv2.remap(rotated_frame, flow_map, None, cv2.INTER_LINEAR)
	else:
		stabilized_frame = rotated_frame



	cv2.imshow('Result', stabilized_frame)

	# Update variables for next iteration
	prev_frame = gray




	# Break the loop if 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
