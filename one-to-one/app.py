# ===========================================
# RNN Practical
# Handwritten Digit Classification
# One-to-One RNN
# Dataset : mnist_train.csv
# ===========================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import streamlit as st
except Exception:
    st = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def is_streamlit_runtime():
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

try:

    from tensorflow.keras.models import (
        Sequential,
        load_model
    )

    from tensorflow.keras.layers import (
        SimpleRNN,
        Dense
    )

    from tensorflow.keras.utils import to_categorical

    TENSORFLOW_AVAILABLE = True

except:

    TENSORFLOW_AVAILABLE = False


# ===========================================
# Configuration
# ===========================================

MODEL = os.path.join(BASE_DIR, "digit_model.keras")

IMAGE_SIZE = 28

NUM_CLASSES = 10


# ===========================================
# Train Model
# ===========================================

def train_model():

    print("Loading Dataset...")

    df = pd.read_csv(os.path.join(BASE_DIR, "mnist_train.csv"))

    print(df.head())

    # --------------------------
    # Features
    # --------------------------

    X = df.iloc[:,1:].values

    y = df.iloc[:,0].values

    # Normalize

    X = X / 255.0

    # Convert

    X = X.reshape((-1,28,28))

    # One Hot Encoding

    y = to_categorical(y,NUM_CLASSES)

    print("X Shape :",X.shape)

    print("Y Shape :",y.shape)

    # --------------------------
    # Train Test Split
    # --------------------------

    x_train,x_test,y_train,y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42

    )

    # --------------------------
    # Model
    # --------------------------

    model = Sequential()

    model.add(

        SimpleRNN(

            128,

            activation="tanh",

            input_shape=(28,28)

        )

    )

    model.add(

        Dense(

            64,

            activation="relu"

        )

    )

    model.add(

        Dense(

            NUM_CLASSES,

            activation="softmax"

        )

    )

    model.compile(

        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    model.summary()

    # --------------------------
    # Train
    # --------------------------

    history = model.fit(

        x_train,

        y_train,

        validation_split=0.2,

        epochs=10,

        batch_size=64,

        verbose=1

    )
        # --------------------------
    # Save Model
    # --------------------------

    model.save(MODEL)

    print("\nModel Saved Successfully!")

    # --------------------------
    # Evaluate Model
    # --------------------------

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    print("\nTest Accuracy :", accuracy)

    print("Test Loss :", loss)

    # --------------------------
    # Prediction
    # --------------------------

    predictions = model.predict(x_test)

    predicted_labels = np.argmax(predictions, axis=1)

    actual_labels = np.argmax(y_test, axis=1)

    # --------------------------
    # Classification Report
    # --------------------------

    print("\nClassification Report\n")

    print(
        classification_report(
            actual_labels,
            predicted_labels
        )
    )

    # --------------------------
    # Confusion Matrix
    # --------------------------

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            actual_labels,
            predicted_labels
        )
    )

    # --------------------------
    # Display Sample Prediction
    # --------------------------

    sample = x_test[0]

    prediction = model.predict(
        sample.reshape(1,28,28),
        verbose=0
    )

    predicted_digit = np.argmax(prediction)

    actual_digit = actual_labels[0]

    print("\nActual Digit :", actual_digit)

    print("Predicted Digit :", predicted_digit)

    plt.figure(figsize=(4,4))

    plt.imshow(
        sample,
        cmap="gray"
    )

    plt.title(
        f"Predicted : {predicted_digit}"
    )

    plt.axis("off")

    plt.show()

    # --------------------------
    # Accuracy Graph
    # --------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("Model Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.show()

    # --------------------------
    # Loss Graph
    # --------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Model Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.show()

    print("\nTraining Completed Successfully!")
    # ===========================================
# Predict Digit
# ===========================================

def predict_digit(image):

    model = load_model(MODEL)

    image = np.array(image)

    image = image.reshape(28,28)

    image = image / 255.0

    image = image.reshape(1,28,28)

    prediction = model.predict(image, verbose=0)

    digit = np.argmax(prediction)

    confidence = np.max(prediction)

    return digit, confidence


# ===========================================
# Auto Train Model
# ===========================================

def app():
    if not os.path.exists(MODEL):
        if TENSORFLOW_AVAILABLE:
            train_model()
        else:
            print("TensorFlow is not installed.")
            print("Install TensorFlow and rerun the program.")

    if is_streamlit_runtime():

        st.set_page_config(
            page_title="Digit Classifier",
            page_icon="🔢"
        )

        st.title("Handwritten Digit Classification")

        st.write("### One-to-One RNN Example")

        uploaded_file = st.file_uploader(
            "Upload a 28×28 grayscale digit image",
            type=["png","jpg","jpeg"]
        )

        if uploaded_file is not None:

            from PIL import Image

            image = Image.open(uploaded_file)

            image = image.convert("L")

            image = image.resize((28,28))

            st.image(
                image,
                caption="Uploaded Image",
                width=200
            )

            image_array = np.array(image)

            if st.button("Predict"):

                digit, confidence = predict_digit(image_array)

                st.success(f"Predicted Digit : {digit}")

                st.write(
                    "Confidence :",
                    round(confidence*100,2),
                    "%"
                )

    else:

        print("\nStreamlit is not installed.")

        print("Model is ready.")

        print("Call predict_digit(image_array)")


if __name__ == "__main__":
    app()