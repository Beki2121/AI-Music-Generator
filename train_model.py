# train_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import argparse
import logging
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model on MIDI note sequences.")
    parser.add_argument('--notes', type=str, default='notes.pkl', help='Pickle file with extracted notes')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model_out', type=str, default='model.h5', help='Output file for trained model')
    parser.add_argument('--history_out', type=str, default='history.pkl', help='Output file for training history')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    if not os.path.exists(args.notes):
        logging.error(f"Notes file {args.notes} not found. Run preprocess.py first.")
        exit(1)
    with open(args.notes, 'rb') as f:
        notes = pickle.load(f)
    if not notes or len(notes) < 101:
        logging.error("Not enough notes to train. Need at least 101 notes.")
        exit(1)

    sequence_length = 100
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
    network_output = tf.keras.utils.to_categorical(network_output)

    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    checkpoint = ModelCheckpoint(args.model_out, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    logging.info(f"Starting training for {args.epochs} epochs, batch size {args.batch_size}...")
    history = model.fit(
        network_input, network_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks_list
    )
    logging.info(f"Training complete. Best model saved to {args.model_out}")

    with open(args.history_out, 'wb') as f:
        pickle.dump(history.history, f)
    logging.info(f"Training history saved to {args.history_out}")

    with open("notes.pkl", "wb") as f:
        pickle.dump(pitchnames, f)
    logging.info("Pitch names saved to notes.pkl")