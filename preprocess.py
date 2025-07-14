# preprocess.py
import music21
import glob
import pickle
import argparse
import logging


def extract_notes_from_midi(folder="midi_data"):
    notes = []
    for file in glob.glob(f"{folder}/*.mid"):
        try:
            midi = music21.converter.parse(file)
            parts = music21.instrument.partitionByInstrument(midi)
            if parts:  # If there are instrument parts
                for part in parts.parts:
                    elements = part.recurse()
                    for el in elements:
                        if isinstance(el, music21.note.Note):
                            notes.append(str(el.pitch))
                        elif isinstance(el, music21.chord.Chord):
                            notes.append('.'.join(str(n) for n in el.normalOrder))
            else:  # If no instrument parts, use flat notes
                elements = midi.flat.notes
                for el in elements:
                    if isinstance(el, music21.note.Note):
                        notes.append(str(el.pitch))
                    elif isinstance(el, music21.chord.Chord):
                        notes.append('.'.join(str(n) for n in el.normalOrder))
            logging.info(f"Extracted notes from {file}")
        except Exception as e:
            logging.error(f"Error processing {file}: {e}")
    return notes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract notes from MIDI files and save to a pickle file.")
    parser.add_argument('--midi_folder', type=str, default='midi_data', help='Folder containing MIDI files')
    parser.add_argument('--output', type=str, default='notes.pkl', help='Output pickle file for notes')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    notes = extract_notes_from_midi(args.midi_folder)
    if notes:
        with open(args.output, 'wb') as f:
            pickle.dump(notes, f)
        logging.info(f"Saved {len(notes)} notes to {args.output}")
    else:
        logging.warning("No notes extracted. Check your MIDI files.")