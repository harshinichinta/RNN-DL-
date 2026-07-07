# ==========================================
# RNN Practical
# Named Entity Recognition (NER)
# Many-to-Many
# Dataset : ner_dataset.csv
# ==========================================

# Import Libraries

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
from sklearn.preprocessing import LabelEncoder

try:

    from tensorflow.keras.models import (
        Sequential,
        load_model
    )

    from tensorflow.keras.layers import (
        Embedding,
        SimpleRNN,
        Dense,
        TimeDistributed
    )

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    from tensorflow.keras.preprocessing.text import Tokenizer

    from tensorflow.keras.utils import to_categorical

    TENSORFLOW_AVAILABLE = True

except:

    TENSORFLOW_AVAILABLE = False


# ==========================================
# Configuration
# ==========================================

MODEL = os.path.join(BASE_DIR, "ner_model.keras")

WORD_TOKENIZER = os.path.join(BASE_DIR, "word_tokenizer.pkl")

TAG_ENCODER = os.path.join(BASE_DIR, "tag_encoder.pkl")

MAX_LEN = 50

VOCAB_SIZE = 20000


# ==========================================
# Train Model
# ==========================================

def train_model():

    print("Loading Dataset...")

    df = pd.read_csv(
        os.path.join(BASE_DIR, "ner_dataset.csv"),
        encoding="latin1"
    )

    print(df.head())

    # Rename Columns

    df.columns = [
        "Sentence",
        "Word",
        "POS",
        "Tag"
    ]

    df = df.ffill()

    # -----------------------------
    # Create Sentences
    # -----------------------------

    sentences = []

    tags = []

    grouped = df.groupby("Sentence")

    for sentence, data in grouped:

        words = list(data["Word"].values)

        ner_tags = list(data["Tag"].values)

        sentences.append(words)

        tags.append(ner_tags)

    print("Total Sentences :", len(sentences))

    # -----------------------------
    # Tokenizer
    # -----------------------------

    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(sentences)

    X = tokenizer.texts_to_sequences(sentences)

    X = pad_sequences(
        X,
        maxlen=MAX_LEN,
        padding="post"
    )

    with open(
        WORD_TOKENIZER,
        "wb"
    ) as f:

        pickle.dump(
            tokenizer,
            f
        )

    # -----------------------------
    # Encode Tags
    # -----------------------------

    encoder = LabelEncoder()

    all_tags = sorted(
        list(df["Tag"].unique())
    )

    encoder.fit(all_tags)

    y = []

    for tag_list in tags:

        encoded = encoder.transform(tag_list)

        y.append(encoded)

    y = pad_sequences(
        y,
        maxlen=MAX_LEN,
        padding="post"
    )

    with open(
        TAG_ENCODER,
        "wb"
    ) as f:

        pickle.dump(
            encoder,
            f
        )

    num_tags = len(encoder.classes_)

    y = np.array([
        to_categorical(
            i,
            num_classes=num_tags
        )
        for i in y
    ])

    print("X Shape :", X.shape)

    print("Y Shape :", y.shape)

    # -----------------------------
    # Train Test Split
    # -----------------------------

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # -----------------------------
    # Build Model
    # -----------------------------

    model = Sequential()

    model.add(
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=128,
            input_length=MAX_LEN
        )
    )

    model.add(
        SimpleRNN(
            128,
            return_sequences=True
        )
    )

    model.add(
        TimeDistributed(
            Dense(
                num_tags,
                activation="softmax"
            )
        )
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # -----------------------------
    # Train
    # -----------------------------

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=32,
        verbose=1
    )
        # ==========================================
    # Save Model
    # ==========================================

    model.save(MODEL)

    print("\nModel Saved Successfully!")

    # ==========================================
    # Evaluate Model
    # ==========================================

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    print("\nTest Accuracy :", accuracy)

    print("Test Loss :", loss)

    # ==========================================
    # Sample Prediction
    # ==========================================

    sample = x_test[:1]

    prediction = model.predict(
        sample,
        verbose=0
    )

    predicted_tags = np.argmax(
        prediction,
        axis=-1
    )

    actual_tags = np.argmax(
        y_test[:1],
        axis=-1
    )

    print("\nSample Prediction\n")

    reverse_word = {
        value:key
        for key,value in tokenizer.word_index.items()
    }

    reverse_tag = {
        index:tag
        for index,tag in enumerate(
            encoder.classes_
        )
    }

    for word_id, pred, actual in zip(
        sample[0],
        predicted_tags[0],
        actual_tags[0]
    ):

        if word_id == 0:
            continue

        word = reverse_word.get(
            word_id,
            "UNK"
        )

        predicted = reverse_tag.get(
            pred,
            "O"
        )

        actual_value = reverse_tag.get(
            actual,
            "O"
        )

        print(
            f"{word:15}  Predicted:{predicted:10}  Actual:{actual_value}"
        )

    # ==========================================
    # Accuracy Graph
    # ==========================================

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("NER Model Accuracy")

    plt.legend()

    plt.grid(True)

    plt.show()

    # ==========================================
    # Loss Graph
    # ==========================================

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("NER Model Loss")

    plt.legend()

    plt.grid(True)

    plt.show()

    print("\nTraining Completed Successfully!")
    # ==========================================
# Predict Sentence
# ==========================================

def predict_sentence(sentence):

    model = load_model(MODEL)

    with open(WORD_TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    with open(TAG_ENCODER, "rb") as f:
        encoder = pickle.load(f)

    words = sentence.strip().split()

    sequence = tokenizer.texts_to_sequences([words])

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )

    prediction = np.argmax(
        prediction,
        axis=-1
    )[0]

    result = []

    for i, word in enumerate(words):

        if i >= MAX_LEN:
            break

        tag = encoder.inverse_transform(
            [prediction[i]]
        )[0]

        result.append((word, tag))

    return result


# ==========================================
# Auto Train Model
# ==========================================

if not os.path.exists(MODEL):

    if TENSORFLOW_AVAILABLE:

        train_model()

    else:

        print("TensorFlow is not installed.")
        print("Install TensorFlow and rerun.")


def app():
    page_style = """
    <style>
    body {
        background: linear-gradient(135deg, #f9f9ff 0%, #eef2ff 100%);
    }
    .stButton>button {
        background-color: #1f4b99;
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    textarea, input[type='text'] {
        border-radius: 0.75rem;
        border: 1px solid #d6dbf5;
        padding: 0.75rem;
    }
    .stMarkdown>h1, .stMarkdown>h2, .stMarkdown>h3 {
        color: #1f4b99;
    }
    </style>
    """
    if is_streamlit_runtime():
        st.markdown(page_style, unsafe_allow_html=True)
        try:
            st.set_page_config(
                page_title="NER using RNN",
                page_icon="🧠"
            )
        except Exception:
            pass

        st.title("Named Entity Recognition")
        st.write("### Many-to-Many RNN Example")
        st.write("Enter a sentence to label each token with a named entity tag.")
        st.sidebar.header("NER Demo")
        st.sidebar.write("This demo uses a many-to-many RNN to classify tokens in a sentence.")

        sentence = st.text_area(
            "Enter a Sentence",
            "John lives in Hyderabad",
            height=120,
        )

        if st.button("Predict"):
            with st.spinner("Generating named entity tags..."):
                output = predict_sentence(sentence)
            st.success("Prediction Completed")
            data = pd.DataFrame(
                output,
                columns=["Word", "Predicted Tag"]
            )
            st.table(data)

    else:
        print("\nStreamlit is not installed.")
        print("Model is ready.")
        print("Use predict_sentence('John lives in Hyderabad')")


if __name__ == "__main__":
    # Auto-train and run UI when executed directly
    if not os.path.exists(MODEL):
        if TENSORFLOW_AVAILABLE:
            train_model()
        else:
            print("TensorFlow is not installed.")
            print("Install TensorFlow and rerun.")
    app()