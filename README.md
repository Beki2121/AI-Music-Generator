# AI Music Generator

Generate original music using deep learning and MIDI data!

---

## Features
- Uses LSTM neural networks to generate music sequences
- Trains on your own MIDI files (classical, jazz, etc.)
- Generates new music, converts to MIDI and WAV
- Web interface for easy music generation and playback

---

## Requirements

### Python Packages
- Python 3.7+
- flask
- music21
- tensorflow
- pretty_midi
- soundfile
- numpy

Install all Python dependencies:
```bash
pip install -r requirements.txt
```

### System Dependencies
- **fluidsynth** (for MIDI to WAV conversion)
- **A SoundFont file** (e.g., `soundfont.sf2`)

#### Install fluidsynth
- **Windows:**
  - Download from https://github.com/FluidSynth/fluidsynth/releases
  - Add the install directory to your PATH
- **macOS:**
  - `brew install fluid-synth`
- **Linux:**
  - `sudo apt-get install fluidsynth`

#### Get a SoundFont
- Download a free SoundFont, e.g. [GeneralUser GS](https://schristiancollins.com/generaluser.php)
- Place the `.sf2` file in your project root as `soundfont.sf2`

---

## Usage

### 1. Preprocess MIDI Data
Place your MIDI files in the `midi_data/` folder.

Extract notes and save to a pickle file:
```bash
python preprocess.py --midi_folder midi_data --output notes.pkl
```

### 2. Train the Model
Train the LSTM model on your extracted notes:
```bash
python train_model.py --notes notes.pkl --epochs 50 --batch_size 64 --model_out model.h5 --history_out history.pkl
```

### 3. Run the Web App
Start the Flask server:
```bash
python app.py
```
Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

### 4. Generate Music
- Click "Generate Music" in the web UI.
- Wait for the music to be generated and played.
- If there is an error, a message will be shown.

---

## Troubleshooting
- **fluidsynth not found:** Ensure it is installed and in your PATH.
- **soundfont.sf2 not found:** Download a SoundFont and place it in the project root.
- **Model or notes.pkl missing:** Run preprocessing and training first.
- **No sound or errors in browser:** Check the server logs for details.

---

## Customization
- You can use your own MIDI files for training.
- Adjust model parameters in `train_model.py` as needed.
- For longer/shorter generations, change the number of notes in `generate_music.py`.

---

## Credits
- Built with [music21](http://web.mit.edu/music21/), [TensorFlow](https://www.tensorflow.org/), [pretty_midi](https://github.com/craffel/pretty-midi), and [FluidSynth](https://www.fluidsynth.org/). 