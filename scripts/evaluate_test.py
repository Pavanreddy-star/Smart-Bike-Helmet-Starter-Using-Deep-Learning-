from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('models/helmet_detection_finetuned.h5')

# Test on a single image
img_path = 'dataset/test_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
print("Helmet detected" if prediction[0][0] > 0.5 else "No helmet detected")
