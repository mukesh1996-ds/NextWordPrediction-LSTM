from src.data_preprocessing import prepare_sequences
from src.model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint

X, y, tokenizer = prepare_sequences('data/your_large_text_corpus.txt')
vocab_size = len(tokenizer.word_index) + 1
seq_len = X.shape[1]

model = create_model(vocab_size, seq_len)
model.summary()

checkpoint = ModelCheckpoint("outputs/model.h5", save_best_only=True)
model.fit(X, y, batch_size=128, epochs=10, validation_split=0.1, callbacks=[checkpoint])
