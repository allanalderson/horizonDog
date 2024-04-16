#
# To stabilize a shaky video using OpenCV in Python, you can follow these steps:
#
#         1.      Import necessary libraries:
#
# import cv2
# import numpy as np
#
#         2.      Read the input video:
#
# cap = cv2.VideoCapture('input_video.mp4')
#
#         3.      Create a VideoWriter object to save the stabilized video:
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output_video.avi', fourcc, 30, (int(cap.get(3)), int(cap.get(4))))
#
#         4.      Initialize variables for motion estimation:
#
# prev_frame = None
# prev_points = None
#
#         5.      Loop through each frame of the video:
#


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform motion estimation between current and previous frame
    if prev_frame is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = -flow

        # Apply the motion to the frame
        h, w = flow.shape[:2]
        flow_map = np.column_stack((flow[:,:,0] + np.arange(w), flow[:,:,1] + np.arange(h)[:,np.newaxis])).astype(np.float32)
        stabilized_frame = cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)
    else:
        stabilized_frame = frame

    # Write the stabilized frame to the output video
    out.write(stabilized_frame)

    # Display the stabilized frame
    cv2.imshow('Stabilized Video', stabilized_frame)

    # Update variables for next iteration
    prev_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        6.      Release resources:

cap.release()
out.release()
cv2.destroyAllWindows()


#
# This code calculates optical flow between consecutive frames using the Farneback method and applies the motion to stabilize the video. Adjust parameters as needed for better results.
#    - Allan.
#
#       Tamworth, Australia, Earth.
#        040 999 7870
#        __________________________
