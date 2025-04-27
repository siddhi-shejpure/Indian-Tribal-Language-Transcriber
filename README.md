# Indian & Tribal Language Transcriber

A tool for transcribing audio files in Indian and tribal languages.

## Features

- Transcribe audio recordings in Indian languages
- Support for multiple transcription models (tiny, base, small, medium, large)
- Web interface for easy interaction
- Command line interface for batch processing
- Support for both native script and Roman transliteration
- Special handling for Sanskrit mantras and Konkani language

## Setup and Installation

### Option 1: Run without Docker

1. **Clone the repository:**
   ```
   git clone https://github.com/siddhi-shejpure/Indian-Tribal-Language-Transcriber.git
   cd Indian-Tribal-Language-Transcriber
   ```

2. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   
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
- Choose between different model sizes
- Select specific languages or use auto-detection

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

## Supported Languages

- Hindi (hi)
- Marathi (mr)
- Bengali (bn)
- Tamil (ta)
- Telugu (te)
- Malayalam (ml)
- Kannada (kn)
- Gujarati (gu)
- Punjabi (pa)
- Odia (or)
- Assamese (as)
- Sanskrit (sa)
- Konkani (kok)
- Auto-detect (auto)

## Environment Variables

You can configure the application using environment variables:
- `MODEL_SIZE`: Default model size to use (tiny, base, small, medium, large)
- `OUTPUT_DIR`: Directory to store transcriptions
- `WEB_INTERFACE`: Set to "True" to start with web interface
- `HOST`: Host for the web interface
- `PORT`: Port for the web interface

## Project Structure

```
Indian-Tribal-Language-Transcriber/
├── mapping.py              # Main application code
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
├── models/                # Directory for model files
└── transcriptions/        # Directory for transcription outputs
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 