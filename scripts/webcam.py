import cv2
import tensorflow as tf
import numpy as np

def load_model():
    return tf.keras.models.load_model('C:/Users/PranithaReddy/Desktop/helmet_detection/models/helmet_model.h5')

def predict_frame(model, frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    prediction = model.predict(frame_expanded)
    return np.argmax(prediction), prediction

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, confidence = predict_frame(model, frame)
        text = f"Helmet: {confidence[0][1]:.2f}, No Helmet: {confidence[0][0]:.2f}"
        color = (0, 255, 0) if label == 1 else (0, 0, 255)

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Helmet Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
