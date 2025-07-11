from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def create_model(vocab_size, seq_len):
    model = Sequential([
        # ✅ Specify input_length so the model builds immediately
        Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_len),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(128, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model
