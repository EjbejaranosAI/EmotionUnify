import cv2
import numpy as np

def cut_face(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Define the classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    # Loop through each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Crop the frame with the rectangle shape around the detected face
            for (x, y, w, h) in faces:
                face_frame = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Write the modified frame to the output video
                out.write(face_frame)

            # Display the modified frame
            cv2.imshow('frame', face_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


cut_face("01_Dataset_generation/dataset/bad/1_woman_23.mp4")