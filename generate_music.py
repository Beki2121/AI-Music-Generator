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

def create_midi_from_notes(notes, output_file='output/generated.mid', instrument_name=None):
    from music21 import stream, note, chord, instrument
    import random
    midi_stream = stream.Stream()
    # List of realistic instruments
    instruments = {
        'Piano': instrument.Piano(),
        'Violin': instrument.Violin(),
        'Guitar': instrument.Guitar(),
        'Flute': instrument.Flute(),
        'Trumpet': instrument.Trumpet(),
        'Clarinet': instrument.Clarinet(),
        'Oboe': instrument.Oboe(),
        'Tuba': instrument.Tuba(),
        'Bassoon': instrument.Bassoon(),
    }
    instrument_names = list(instruments.keys())
    if instrument_name == 'mixed':
        chosen_instrument = None  # Will be handled per note
        used_instrument = 'Mixed'
        midi_stream.insert(0, instrument.Piano())  # Default for the stream
    elif instrument_name and instrument_name in instruments:
        chosen_instrument = instruments[instrument_name]
        used_instrument = instrument_name
        midi_stream.insert(0, chosen_instrument)
    else:
        chosen_instrument = random.choice(list(instruments.values()))
        used_instrument = [k for k, v in instruments.items() if v == chosen_instrument][0]
        midi_stream.insert(0, chosen_instrument)
    for element in notes:
        if instrument_name == 'mixed':
            inst = random.choice(list(instruments.values()))
            midi_stream.append(inst.__class__())
        if '.' in element:
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
                try:
                    pitch = note.Note()
                    pitch.midi = 60 + int(element)
                    midi_stream.append(pitch)
                except Exception:
                    continue
    if output_file:
        midi_stream.write('midi', fp=output_file)
        return output_file, used_instrument
    else:
        return midi_stream, used_instrument

def midi_to_audio(midi_path='output/generated.mid', audio_path='output/generated.wav'):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    audio = midi_data.fluidsynth()
    import soundfile as sf
    sf.write(audio_path, audio, samplerate=44100)
    return audio_path

def generate_and_save_music(instrument_names=None, length_seconds=60):
    """
    Generates music using the trained model, saves MIDI and WAV files in static/audio/, and returns the WAV file path and instrument(s) used.
    instrument_names: list of instrument names (strings)
    length_seconds: approximate duration in seconds for the generated music
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
    # Estimate: 1 note ≈ 0.5 seconds (adjust as needed)
    num_notes = max(10, int(length_seconds / 0.5))
    # If no instruments or only 'mixed', use mixed mode
    if not instrument_names or (len(instrument_names) == 1 and instrument_names[0] == 'mixed'):
        notes = generate_sequence(model, note_to_int, int_to_note, num_notes=num_notes)
        _, used_instrument = create_midi_from_notes(notes, output_file=midi_path, instrument_name='mixed')
    else:
        from music21 import stream
        midi_stream = stream.Stream()
        used_instruments = []
        for inst_name in instrument_names:
            notes = generate_sequence(model, note_to_int, int_to_note, num_notes=num_notes)
            part, used_inst = create_midi_from_notes(notes, output_file=None, instrument_name=inst_name)
            midi_stream.append(part)
            used_instruments.append(used_inst)
        midi_stream.write('midi', fp=midi_path)
        used_instrument = ', '.join(used_instruments)
    midi_to_audio(midi_path, audio_path=wav_path)
    return wav_path, used_instrument
