
# 🧠 LSTM-Based Next Word Prediction

This project implements an end-to-end LSTM neural network that learns to predict the **next word** in a sentence based on a custom text corpus. It's built using TensorFlow/Keras and includes all stages from data preprocessing to model inference.

## 📁 Project Structure

```

lstm\_next\_word\_prediction/
│
├── data/
│   └── gutenberg\_corpus.txt         # Merged text from classic books
│
├── outputs/
│   ├── model.h5                     # Trained LSTM model
│   └── tokenizer.pkl                # Saved tokenizer
│
├── src/
│   ├── data\_preprocessing.py        # Text cleaning, tokenization, sequence creation
│   ├── model.py                     # LSTM model definition
│   ├── train.py                     # Training script
│   └── predict.py                   # Next-word inference script
│
├── requirements.txt                 # All necessary dependencies
├── README.md                        # Project overview and instructions
└── download\_gutenberg\_books.py      # (Optional) Script to build the dataset

````

---

## 🔧 Installation

Make sure you're using **Python 3.11** or below (TensorFlow is not yet supported in 3.13).

```bash
# Create virtual environment (replace `python3.11` if needed)
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt
````

---

## 📚 Dataset

By default, we use a custom merged dataset of 5 public domain books from **Project Gutenberg**.

To auto-download and merge them into `data/gutenberg_corpus.txt`:

```bash
python download_gutenberg_books.py
```

---

## 🚀 Training the Model

```bash
python src/train.py
```

This script:

* Preprocesses the text into padded sequences
* Builds and trains a Keras LSTM model
* Saves the trained model and tokenizer to the `outputs/` directory

---

## 🔮 Predicting the Next Word

To predict the next word given an input prompt:

```bash
python src/predict.py
```

Inside the file, you can test:

```python
from src.predict import predict_next_word

print(predict_next_word("deep learning is"))
```

---

## 📊 Model Architecture

* **Embedding Layer**: Converts words to dense vectors
* **2 LSTM Layers**: Learns sequence dependencies
* **Dense Layer**: Fully connected relu
* **Output Layer**: Softmax for next word prediction

---

## 📌 Requirements

All dependencies are listed in `requirements.txt`. Key ones include:

* `tensorflow==2.16.1` (Use `tf-nightly` if on Python 3.13)
* `numpy`, `scikit-learn`, `pandas`, `tqdm`, `requests`

> If using Python 3.13, replace `tensorflow` with `tf-nightly`.

---

## 🧪 Example Output

```bash
Input:  "deep learning is"
Output: "powerful"
```

---

## 📈 Future Improvements

* Use Bidirectional LSTMs or Transformers
* Add beam search for more realistic predictions
* Train on larger text corpora (e.g., Wikipedia dump)
* Convert into a Streamlit app
