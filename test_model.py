import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model

# Check if TensorFlow is using GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Enable memory growth (prevents TensorFlow from using all GPU memory)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is enabled and memory growth is set.")
    except RuntimeError as e:
        print(e)

# Define class labels
CLASS_LABELS = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sadness']

# Load the trained model
model = load_model("emotion_predictor.h5")

# Function to preprocess the image (resize to 48x48)
def preprocess_image(image_path, img_height=48, img_width=48):
    img = cv2.imread(image_path)  # Load image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (img_width, img_height))  # Resize to 48x48
    img = img / 255.0  # Normalize pixel values (0-1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load and preprocess the sample image
image_path = "6.jpg"  # Replace with your image path
processed_img = preprocess_image(image_path)

# Make a prediction
predictions = model.predict(processed_img)
predicted_class = np.argmax(predictions)  # Get class index
confidence = np.max(predictions)  # Get confidence score

# Display the result
print(f"Predicted Emotion: {CLASS_LABELS[predicted_class]} (Confidence: {confidence:.2f})")
