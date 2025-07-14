# generate.py
import numpy as np
import tensorflow as tf
import pickle
import random
import pretty_midi
import os
import uuid
import time

SEQUENCE_LENGTH = 100

def load_model_and_data(model_path='model/model.h5', pitch_path='model/pitch_names.pkl'):
    model = tf.keras.models.load_model(model_path)
    with open(pitch_path, 'rb') as f:
        pitchnames = pickle.load(f)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    return model, pitchnames, note_to_int, int_to_note

def generate_sequence(model, note_to_int, int_to_note, num_notes=300):
    vocab_size = len(note_to_int)
    # Start with a random sequence
    pattern = [random.randint(0, vocab_size - 1) for _ in range(SEQUENCE_LENGTH)]
    prediction_output = []

    for _ in range(num_notes):
        input_seq = np.reshape(pattern, (1, SEQUENCE_LENGTH, 1)) / float(vocab_size)
        prediction = model.predict(input_seq, verbose=0)[0]
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]  # Move window forward
    return prediction_output

def create_midi_from_notes(notes, output_file='output/generated.mid'):
    from music21 import stream, note, chord, instrument
    import random
    midi_stream = stream.Stream()
    # List of realistic instruments
    instruments = [
        instrument.Piano(),
        instrument.Violin(),
        instrument.Guitar(),
        instrument.Flute(),
        instrument.Trumpet(),
        instrument.Clarinet(),
    ]
    chosen_instrument = random.choice(instruments)
    midi_stream.insert(0, chosen_instrument)
    for element in notes:
        if '.' in element:
            # Chord: convert each number to an integer, then to a pitch (C4 as base, MIDI 60)
            chord_notes = []
            for n in element.split('.'):
                try:
                    pitch = note.Note()
                    pitch.midi = 60 + int(n)
                    chord_notes.append(pitch)
                except Exception:
                    continue
            midi_stream.append(chord.Chord(chord_notes))
        else:
            try:
                midi_stream.append(note.Note(element))
            except Exception:
                # If element is a number, treat as MIDI offset from C4
                try:
                    pitch = note.Note()
                    pitch.midi = 60 + int(element)
                    midi_stream.append(pitch)
                except Exception:
                    continue
    midi_stream.write('midi', fp=output_file)
    return output_file

def midi_to_audio(midi_path='output/generated.mid', audio_path='output/generated.wav'):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    audio = midi_data.fluidsynth()
    import soundfile as sf
    sf.write(audio_path, audio, samplerate=44100)
    return audio_path

def generate_and_save_music():
    """
    Generates music using the trained model, saves MIDI and WAV files in static/audio/, and returns the WAV file path.
    """
    import os
    output_dir = os.path.join('static', 'audio')
    os.makedirs(output_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())
    midi_path = os.path.join(output_dir, f'generated_{unique_id}.mid')
    wav_path = os.path.join(output_dir, f'generated_{unique_id}.wav')
    model_path = 'model/model.h5'
    pitch_path = 'model/pitch_names.pkl'
    if not os.path.exists(model_path) or not os.path.exists(pitch_path):
        raise FileNotFoundError('Model or pitch names file not found. Please train the model first.')
    model, pitchnames, note_to_int, int_to_note = load_model_and_data(model_path, pitch_path)
    notes = generate_sequence(model, note_to_int, int_to_note, num_notes=120)
    create_midi_from_notes(notes, output_file=midi_path)
    midi_to_audio(midi_path, audio_path=wav_path)
    return wav_path
