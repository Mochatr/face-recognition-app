import cv2 # OpenCV library

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Access the webcam
cap = cv2.VideoCapture(0)

# check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # check if the frame was not captured correctly, skip the iteration
    if not ret:
        print("Error: Can't receive frame.")
        continue

    # Convert the frame to grayscale for face detection 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(300, 300))

    # Draw a green rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break