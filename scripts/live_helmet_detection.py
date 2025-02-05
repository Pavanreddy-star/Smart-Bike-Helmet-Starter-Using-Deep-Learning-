import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('C:/Users/PranithaReddy/Desktop/helmet_detection/models/helmet_model.h5')

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to the target size
    img_resized = cv2.resize(frame, (224, 224))
    
    # Convert the image to array and normalize
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    label = "Helmet detected" if prediction[0][0] > 0.5 else "No helmet detected"

    # Display the result on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('Helmet Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
