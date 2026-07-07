# ==========================================
# RNN Practical
# Character Text Generation using Simple RNN
# One-to-Many
# Dataset : names.txt
# ==========================================

import os
import pickle
import numpy as np
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

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False
    Sequential = None
    load_model = None
    Embedding = None
    SimpleRNN = None
    Dense = None
    Tokenizer = None
    pad_sequences = None


MODEL = os.path.join(BASE_DIR, "text_model.keras")
TOKENIZER = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LEN = 20
EMBEDDING_DIM = 64
RNN_UNITS = 128


def train_model():
    print("Loading Dataset...")

    with open(os.path.join(BASE_DIR, "names.txt"), "r", encoding="utf-8") as file:
        text = file.read().lower()

    print(text[:200])

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts([text])
    total_chars = len(tokenizer.word_index) + 1

    print("Vocabulary Size :", total_chars)

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    sequences = []
    labels = []
    for i in range(MAX_LEN, len(text)):
        seq = text[i - MAX_LEN:i]
        label = text[i]
        sequences.append(seq)
        labels.append(label)

    print("Total Sequences :", len(sequences))

    X = np.array(tokenizer.texts_to_sequences(sequences))
    y = np.array(tokenizer.texts_to_sequences(labels))

    print("X Shape :", X.shape)
    print("Y Shape :", y.shape)

    model = Sequential()
    model.add(Embedding(input_dim=total_chars, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(SimpleRNN(RNN_UNITS))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(total_chars, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    model.save(MODEL)
    print("\nModel Saved Successfully!")

    final_accuracy = history.history["accuracy"][-1]
    final_loss = history.history["loss"][-1]
    print("\nTraining Accuracy :", round(final_accuracy * 100, 2), "%")
    print("Training Loss :", round(final_loss, 4))

    reverse_index = {value: key for key, value in tokenizer.word_index.items()}
    seed = "harshini"
    generated = seed

    for _ in range(40):
        sequence = tokenizer.texts_to_sequences([generated[-MAX_LEN:]])
        sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="pre")
        prediction = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_char = reverse_index.get(predicted_index, "")
        generated += predicted_char

    print("\nGenerated Text\n")
    print(generated)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nTraining Completed Successfully!")


def generate_text(seed_text, length=50):
    model = load_model(MODEL)
    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    reverse_index = {value: key for key, value in tokenizer.word_index.items()}
    generated = seed_text.lower()

    for _ in range(length):
        sequence = tokenizer.texts_to_sequences([generated[-MAX_LEN:]])
        sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="pre")
        prediction = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_char = reverse_index.get(predicted_index, "")
        generated += predicted_char

    return generated


if not os.path.exists(MODEL):
    if TENSORFLOW_AVAILABLE:
        train_model()
    else:
        print("TensorFlow is not installed. Install TensorFlow and rerun.")


if is_streamlit_runtime():
    st.set_page_config(page_title="Character Text Generator", page_icon="✍️")
    st.title("Character Text Generation")
    st.write("### One-to-Many RNN Example")

    seed = st.text_input("Enter Starting Text", value="har")
    length = st.slider("Characters to Generate", min_value=10, max_value=100, value=40)

    if st.button("Generate Text"):
        result = generate_text(seed, length)
        st.success(result)
else:
    print("\nStreamlit is not installed.")
    print("Model is ready.")
    print("Use generate_text('har', 40)")
