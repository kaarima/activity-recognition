import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('activity_recognition_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to 64x64
    image = cv2.resize(image, (64, 64))
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize the image
    image = image.astype('float32') / 255.0
    # Reshape for the model
    image = image.reshape(-1, 64, 64, 1)
    return image

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Make a prediction
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions[0])

    # Display the predicted activity on the frame
    cv2.putText(frame, f'Predicted Activity: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with prediction
    cv2.imshow('Activity Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
