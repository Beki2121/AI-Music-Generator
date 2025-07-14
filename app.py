from flask import Flask, render_template, request, send_file, jsonify, make_response
from generate_music import generate_and_save_music
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json(silent=True) or {}
        instruments = data.get('instruments')
        if not instruments:
            instruments = []
        length = data.get('length')
        try:
            length = int(length)
        except (TypeError, ValueError):
            length = 60  # default to 1 minute
        result = generate_and_save_music(instrument_names=instruments, length_seconds=length)
        if isinstance(result, tuple):
            wav_path, used_instrument = result
        else:
            wav_path = result
            used_instrument = ', '.join(instruments) if instruments else 'Random'
        if not isinstance(wav_path, str) or not os.path.exists(wav_path):
            logging.error(f"WAV file not found after generation: {wav_path}")
            return jsonify({'error': 'Music generation failed. Please try again later.'}), 500
        response = make_response(send_file(wav_path, mimetype='audio/wav'))
        response.headers['X-Instrument'] = used_instrument
        return response
    except Exception as e:
        logging.error(f"Music generation error: {e}")
        return jsonify({'error': f'Music generation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)