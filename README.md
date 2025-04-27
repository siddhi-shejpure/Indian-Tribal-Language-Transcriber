# Indian & Tribal Language Transcriber

A tool for transcribing audio files in Indian and tribal languages.

## Features

- Transcribe audio recordings in Indian languages
- Support for multiple transcription models
- Web interface for easy interaction
- Command line interface for batch processing

## Setup and Installation

### Option 1: Run without Docker

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd Indian-Tribal-Language
   ```

2. **Install system dependencies:**
   - ffmpeg
   - libsndfile1

   On Ubuntu/Debian:
   ```
   sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
   ```

3. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the application:**
   
   With web interface:
   ```
   python mapping.py --web --host 0.0.0.0 --port 7860
   ```
   
   Or for command-line usage:
   ```
   python mapping.py --input /path/to/audio/file.mp3 --output /path/for/transcription
   ```

### Option 2: Run with Docker

1. **Build the Docker image:**
   ```
   docker build -t indian-language-transcriber .
   ```

2. **Run the Docker container:**
   ```
   docker run -p 7860:7860 -v $(pwd)/transcriptions:/app/transcriptions indian-language-transcriber
   ```

3. **Access the web interface** by opening http://localhost:7860 in your browser.

## Usage

### Web Interface

Navigate to http://localhost:7860 in your browser to access the web interface. From there, you can:
- Upload audio files
- Select transcription options
- View and download transcription results

### Command Line Interface

Basic usage:
```
python mapping.py --input /path/to/audio/file.mp3 --output /path/for/transcription
```

Additional options:
```
--model_size [tiny, base, small, medium, large]  # Choose model size based on need
--language [hi, ta, te, etc.]                    # Specify language (optional)
--web                                            # Launch web interface
--host 0.0.0.0                                   # Host for web interface
--port 7860                                      # Port for web interface
```

## Environment Variables

You can configure the application using environment variables:
- `MODEL_SIZE`: Default model size to use (tiny, base, small, medium, large)
- `OUTPUT_DIR`: Directory to store transcriptions
- `WEB_INTERFACE`: Set to "True" to start with web interface
- `HOST`: Host for the web interface
- `PORT`: Port for the web interface

## License

This project is licensed under the MIT License - see the LICENSE file for details. 