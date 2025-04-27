import os
import torch
import whisper
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import gradio as gr
from pathlib import Path
import tempfile
import traceback
import gc
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import resampy
import torchaudio
from tqdm.auto import tqdm
import logging
import shutil  # For file operations
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Dictionary mapping ISO language codes to full names
LANGUAGE_MAP = {
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "sa": "Sanskrit",
    # Tribal languages
    "sat": "Santali",
    "kok": "Konkani",
    "gon": "Gondi",
    "bho": "Bhojpuri",
    "mai": "Maithili",
    "doi": "Dogri",
    "awa": "Awadhi",
    "mag": "Magahi",
    "mni": "Manipuri",
    "ne": "Nepali",
    "urd": "Urdu",
    "sd": "Sindhi",
    "ks": "Kashmiri",
    "auto": "Auto-detect"
}

# Extended mapping language codes to their script systems
SCRIPT_MAP = {
    "hi": sanscript.DEVANAGARI,
    "mr": sanscript.DEVANAGARI,
    "bn": sanscript.BENGALI,
    "ta": sanscript.TAMIL,
    "te": sanscript.TELUGU,
    "ml": sanscript.MALAYALAM,
    "kn": sanscript.KANNADA,
    "gu": sanscript.GUJARATI,
    "pa": sanscript.GURMUKHI,
    "or": sanscript.ORIYA,
    "as": sanscript.BENGALI,  # Assamese uses Bengali script with variations
    "sa": sanscript.DEVANAGARI,
    "sat": "OL_CHIKI",  # Santali has its own script (Ol Chiki)
    "kok": sanscript.DEVANAGARI,  # Konkani primarily uses Devanagari script
    "gon": sanscript.DEVANAGARI,
    "bho": sanscript.DEVANAGARI,
    "mai": sanscript.DEVANAGARI,
    "doi": sanscript.DEVANAGARI,
    "awa": sanscript.DEVANAGARI,
    "mag": sanscript.DEVANAGARI,
    "mni": sanscript.BENGALI,  # Manipuri uses Bengali script
    "ne": sanscript.DEVANAGARI,
    "urd": "URDU",  # Using a placeholder for Urdu script
    "sd": "SINDHI",  # Using a placeholder for Sindhi script
    "ks": "KASHMIRI",  # Using a placeholder for Kashmiri script
}

# Special vowel mappings for better transliteration
VOWEL_MAPPINGS = {
    # Hindi/Devanagari special vowels
    "अ": "a", "आ": "ā", "इ": "i", "ई": "ī", "उ": "u", "ऊ": "ū",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऋ": "ṛ", "ॠ": "ṝ",
    "ऌ": "ḷ", "ॡ": "ḹ", "ं": "ṃ", "ः": "ḥ",
    
    # Konkani special vowels and consonants (Devanagari script with special characters)
    "अं": "aṃ", "आं": "āṃ", "ऑ": "ŏ", "ॉ": "ŏ", "ळ": "ḷa", "ळ्": "ḷ", 
    "न्": "n", "म्": "m", "झ": "jh", "च्": "c", "त्": "t",
    "च्य": "cy", "न्य": "ny", "ल्य": "ly", "त्य": "ty", "द्य": "dy",
    "त्र": "tr", "ज्ञ": "jñ", "श्र": "śr", "क्ष": "kṣ",
    
    # Bengali special vowels
    "অ": "ô", "আ": "a", "ই": "i", "ঈ": "ī", "উ": "u", "ঊ": "ū",
    "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou", "ঋ": "ri",
    
    # Tamil special vowels
    "அ": "a", "ஆ": "ā", "இ": "i", "ஈ": "ī", "உ": "u", "ஊ": "ū",
    "எ": "e", "ஏ": "ē", "ஐ": "ai", "ஒ": "o", "ஓ": "ō", "ஔ": "au",
    
    # Telugu special vowels
    "అ": "a", "ఆ": "ā", "ఇ": "i", "ఈ": "ī", "ఉ": "u", "ఊ": "ū",
    "ఎ": "e", "ఏ": "ē", "ఐ": "ai", "ఒ": "o", "ఓ": "ō", "ఔ": "au", "ఋ": "ṛ",
    
    # Malayalam special vowels
    "അ": "a", "ആ": "ā", "ഇ": "i", "ഈ": "ī", "ഉ": "u", "ഊ": "ū",
    "എ": "e", "ഏ": "ē", "ഐ": "ai", "ഒ": "o", "ഓ": "ō", "ഔ": "au", "ഋ": "ṛ",
    
    # Kannada special vowels
    "ಅ": "a", "ಆ": "ā", "ಇ": "i", "ಈ": "ī", "ಉ": "u", "ಊ": "ū",
    "ಎ": "e", "ಏ": "ē", "ಐ": "ai", "ಒ": "o", "ಓ": "ō", "ಔ": "au", "ಋ": "ṛ",
    
    # Gujarati special vowels
    "અ": "a", "આ": "ā", "ઇ": "i", "ઈ": "ī", "ઉ": "u", "ઊ": "ū",
    "એ": "e", "ઐ": "ai", "ઓ": "o", "ઔ": "au", "ઋ": "ṛ",
    
    # Punjabi (Gurmukhi) special vowels
    "ਅ": "a", "ਆ": "ā", "ਇ": "i", "ਈ": "ī", "ਉ": "u", "ਊ": "ū",
    "ਏ": "e", "ਐ": "ai", "ਓ": "o", "ਔ": "au",
    
    # Odia special vowels
    "ଅ": "a", "ଆ": "ā", "ଇ": "i", "ଈ": "ī", "ଉ": "u", "ଊ": "ū",
    "ଏ": "e", "ଐ": "ai", "ଓ": "o", "ଔ": "au", "ଋ": "ṛ"
}

# Special Unicode combining marks for Indic scripts
COMBINING_MARKS = {
    # Common diacritics and modifiers
    '\u0901': 'ṃ',  # Devanagari candrabindu
    '\u0902': 'ṃ',  # Devanagari anusvara
    '\u0903': 'ḥ',  # Devanagari visarga
    '\u093c': '',    # Devanagari nukta
    '\u0951': '',    # Devanagari udatta
    '\u0952': '',    # Devanagari anudatta
    
    # Bengali
    '\u0981': 'ṃ',  # Bengali candrabindu
    '\u0982': 'ṃ',  # Bengali anusvara
    '\u0983': 'ḥ',  # Bengali visarga
    
    # Tamil
    '\u0b82': 'ṃ',  # Tamil anusvara
    '\u0b83': 'ḥ',  # Tamil visarga
    
    # Other scripts follow similar patterns
}

# Sanskrit transliteration schemes
SANSKRIT_SCHEMES = {
    "iast": "IAST (International Alphabet of Sanskrit Transliteration)",
    "hk": "Harvard-Kyoto",
    "slp1": "Sanskrit Library Phonetic Basic",
    "itrans": "ITRANS (Indian languages TRANSliteration)",
    "velthuis": "Velthuis",
    "wx": "WX notation"
}

# Script coverage analysis - fixing the format from [(start, end)] to (start, end)
script_ranges = {
    "hi": (0x0900, 0x097F),  # Devanagari (Hindi)
    "bn": (0x0980, 0x09FF),  # Bengali
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    "or": (0x0B00, 0x0B7F),  # Odia
    "mr": (0x0900, 0x097F),  # Devanagari (Marathi)
    "sa": (0x0900, 0x097F),  # Devanagari (Sanskrit)
    "kok": (0x0900, 0x097F)  # Devanagari (Konkani)
}

# Sanskrit indicators to help distinguish from Hindi
sanskrit_indicators = ["ॐ", "संस्कृत", "श्लोक", "शास्त्र", "वेद", "मन्त्र", "पुराण", "उपनिषद्", "स्तोत्र", "धर्मसूत्र"]

# Konkani specific indicators to help distinguish from other Devanagari languages
konkani_indicators = ["कोंकणी", "गोंयची", "अंत्रुज", "बाभदे", "गोंय", "आमगेलें", "तूजें", "कोंकणीत"]

# Common Sanskrit mantras in Devanagari
SANSKRIT_MANTRAS_DEVANAGARI = [
    "ॐ भूर्भुवः स्वः", 
    "ॐ भूर्भुवः स्वाहा",
    "तत्सवितुर्वरेण्यं",
    "भर्गो देवस्य धीमहि",
    "धियो यो नः प्रचोदयात्",
    "गायत्री मंत्र",
    "ॐ नमः शिवाय",
    "ॐ नमो भगवते",
    "असतो मा सद्गमय",
    "ॐ शान्ति शान्ति",
    "ॐ त्र्यम्बकं यजामहे"
]

# Common Sanskrit mantras in transliterated form (for detection in English transcription)
SANSKRIT_MANTRAS_ROMAN = [
    "om bhur bhuvah svaha", 
    "om bhur bhuva svaha",
    "om bhur bhuvah swaha",
    "om bhuh bhuvah svaha",
    "tat savitur varenyam",
    "gayatri mantra",
    "om namah shivaya",
    "om namo bhagavate",
    "asato ma sadgamaya",
    "om shanti shanti",
    "om tryambakam yajamahe"
]

# Well-known mantras with their full Sanskrit text
COMPLETE_MANTRAS = {
    "gayatri": {
        "devanagari": "ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात्॥",
        "roman": "oṃ bhūrbhuvaḥ svaḥ tatsaviturvareṇyaṃ bhargo devasya dhīmahi dhiyo yo naḥ pracodayāt॥"
    },
    "mahamrityunjaya": {
        "devanagari": "ॐ त्र्यम्बकं यजामहे सुगन्धिं पुष्टिवर्धनम् उर्वारुकमिव बन्धनान्मृत्योर्मुक्षीय माऽमृतात्॥",
        "roman": "oṃ tryambakaṃ yajāmahe sugandhiṃ puṣṭivardhanam urvārukamiva bandhanānmṛtyormukṣīya mā'mṛtāt॥"
    },
    "shanti": {
        "devanagari": "ॐ सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः। सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत्॥ ॐ शान्तिः शान्तिः शान्तिः॥",
        "roman": "oṃ sarve bhavantu sukhinaḥ sarve santu nirāmayāḥ। sarve bhadrāṇi paśyantu mā kaścidduḥkhabhāgbhavet॥ oṃ śāntiḥ śāntiḥ śāntiḥ॥"
    }
}

class IndianLanguageTranscriber:
    def __init__(self, model_size="large", device=None, compute_type="float16", local_model_path=None):
        """
        Initialize the transcriber with the specified Whisper model.

        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to run the model on (cuda, cpu)
            compute_type: Computation type to use (float16, float32, int8)
            local_model_path: Path to a pre-downloaded model file (to avoid downloading)
        """
        try:
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            self.compute_type = compute_type

            print(f"Loading Whisper {model_size} model on {self.device} using {compute_type}...")
            # Try loading with retries
            max_retries = 5  # Increased from 3 to 5
            download_root = "./models"
            
            # Check if a local model path is provided
            if local_model_path and os.path.exists(local_model_path):
                print(f"Using local model file: {local_model_path}")
                try:
                    # Load the model directly from the provided path
                    self.model = whisper.load_model(local_model_path, device=self.device)
                    if self.compute_type == "float16" and self.device == "cuda":
                        self.model = self.model.half()
                    print("Model loaded successfully from local file")
                    
                    # Store model size for reference
                    self.model_size = model_size
                    
                    # Set up thread pool for background processing
                    self.executor = ThreadPoolExecutor(max_workers=2)
                    
                    # No need for further loading attempts
                    return
                except Exception as e:
                    print(f"Error loading local model: {e}, will try normal loading")
            
            # Try downloading smaller models if connectivity issues persist
            model_fallback_sequence = [model_size]
            if model_size == "large":
                model_fallback_sequence.extend(["medium", "small", "base", "tiny"])
            elif model_size == "medium":
                model_fallback_sequence.extend(["small", "base", "tiny"])
            elif model_size == "small":
                model_fallback_sequence.extend(["base", "tiny"])
            elif model_size == "base":
                model_fallback_sequence.extend(["tiny"])
                
            # Try each model size in the fallback sequence
            for attempt_model_size in model_fallback_sequence:
                try:
                    print(f"Attempting to load {attempt_model_size} model...")
                    
                    for attempt in range(max_retries):
                        try:
                            # Get CUDA device properties for better optimization
                            if self.device == "cuda":
                                cuda_props = torch.cuda.get_device_properties(0)
                                print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                                print(f"CUDA Memory: {cuda_props.total_memory / (1024**3):.2f} GB")
                                
                                # For large models on devices with limited memory, use int8 quantization
                                if attempt_model_size == "large" and cuda_props.total_memory < 6 * (1024**3):  # Less than 6GB
                                    print("Limited GPU memory detected. Using int8 quantization.")
                                    self.compute_type = "int8"
                            
                            # Check if model files exist and remove them if checksum failed previously
                            model_path = os.path.join(download_root, f"{attempt_model_size}.pt")
                            if os.path.exists(model_path) and attempt > 0:
                                print(f"Removing potentially corrupted model file: {model_path}")
                                os.remove(model_path)
                                # Clear any partial downloads
                                partial_path = model_path + ".part"
                                if os.path.exists(partial_path):
                                    os.remove(partial_path)
                            
                            # Check model directory exists
                            if not os.path.exists(download_root):
                                os.makedirs(download_root, exist_ok=True)
                                
                            # Download the model and load it with the appropriate compute type
                            print(f"Attempt {attempt+1}/{max_retries} to load model...")
                            self.model = whisper.load_model(attempt_model_size, device=self.device, download_root=download_root)
                            
                            # Move model to appropriate compute type
                            if self.compute_type == "float16" and self.device == "cuda":
                                self.model = self.model.half()
                            elif self.compute_type == "int8" and hasattr(torch, 'quantization'):
                                # Apply dynamic quantization if possible
                                try:
                                    self.model = torch.quantization.quantize_dynamic(
                                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                                    )
                                    print("Model quantized to int8 successfully")
                                except Exception as e:
                                    print(f"Quantization failed: {e}, using default model")
                                    
                            print(f"Model {attempt_model_size} loaded successfully")
                            # Store the actual model size that was loaded
                            self.model_size = attempt_model_size
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"Error loading model: {e}")
                                print(f"Retrying ({attempt+1}/{max_retries})...")
                                time.sleep(5)
                            else:
                                # Last attempt failed, will try next model in fallback sequence
                                print(f"Failed to load {attempt_model_size} model after {max_retries} attempts.")
                                raise
                    
                    # If we got here without an exception, we've successfully loaded a model
                    break
                    
                except Exception as e:
                    print(f"Could not load {attempt_model_size} model: {e}")
                    if attempt_model_size == model_fallback_sequence[-1]:
                        # We've tried all models and none worked
                        raise RuntimeError(f"Failed to load any model from {model_fallback_sequence}")
                    else:
                        print(f"Trying fallback to smaller model: {model_fallback_sequence[model_fallback_sequence.index(attempt_model_size) + 1]}")
            
            # Set up thread pool for background processing
            self.executor = ThreadPoolExecutor(max_workers=2)
            
            # Precompile some operations for faster processing
            if self.device == "cuda":
                # Run a small dummy transcription to warm up the model
                print("Warming up the model...")
                dummy_audio = torch.zeros((16000,), device=self.device)
                _ = self.model.transcribe(dummy_audio, fp16=(self.compute_type == "float16"))
                torch.cuda.synchronize()
                print("Model warm-up complete")
                
        except Exception as e:
            print(f"Error initializing transcriber: {e}")
            print(traceback.format_exc())
            raise

    def preprocess_audio(self, audio_path, target_sr=16000):
        """
        Preprocess audio file - load and resample if necessary.
        Optimized for speed with multiple loading methods.

        Args:
            audio_path: Path to the audio file
            target_sr: Target sample rate

        Returns:
            Preprocessed audio array
        """
        try:
            print(f"Preprocessing audio: {audio_path}")
            
            # Try multiple loading methods in order of efficiency
            audio = None
            sr = None
            
            # Method 1: Try with soundfile (fast)
            try:
                audio, sr = sf.read(audio_path)
                # Make sure audio is float32
                audio = audio.astype(np.float32)
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                   audio = audio.mean(axis=1).astype(np.float32)
            except Exception as sf_error:
                print(f"Soundfile error: {sf_error}, trying torchaudio...")
                
                # Method 2: Try with torchaudio
                try:
                    audio_tensor, sr = torchaudio.load(audio_path)
                    audio = audio_tensor.mean(dim=0).numpy().astype(np.float32)
                except Exception as torch_error:
                    print(f"Torchaudio error: {torch_error}, using backup method...")
                    
                    # Method 3: Last resort using librosa (slower but more compatible)
                    try:
                        import librosa
                        audio, sr = librosa.load(audio_path, sr=None)
                        audio = audio.astype(np.float32)
                    except Exception as librosa_error:
                        raise RuntimeError(f"Failed to load audio with all methods: {librosa_error}")
        
            # Normalize audio (important for better transcription)
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()
                
            # Apply noise reduction if the signal is noisy
            noise_threshold = 0.005
            if np.abs(audio).mean() < noise_threshold:
                print("Applying noise reduction...")
                try:
                    # Simple noise gate
                    gate_threshold = 0.01
                    audio[np.abs(audio) < gate_threshold] = 0
                except Exception as noise_error:
                    print(f"Noise reduction failed: {noise_error}, continuing with original audio")
        
            # Resample if needed (using resampy for speed)
            if sr != target_sr:
                print(f"Resampling from {sr}Hz to {target_sr}Hz")
                audio = resampy.resample(audio, sr, target_sr)
                audio = audio.astype(np.float32)
                
            # Ensure audio is properly trimmed to remove silence
            # Fast trimming for large files
            if len(audio) > target_sr * 10:  # If longer than 10s
                start_idx = 0
                end_idx = len(audio) - 1
                
                # Find start (simple energy-based detection)
                window = target_sr // 10  # 100ms window
                for i in range(0, len(audio) - window, window):
                    if np.abs(audio[i:i+window]).mean() > 0.01:
                        start_idx = max(0, i - window)
                        break
                        
                # Find end
                for i in range(len(audio) - window, 0, -window):
                    if np.abs(audio[i:i+window]).mean() > 0.01:
                        end_idx = min(len(audio), i + window * 2)
                        break
                        
                if end_idx > start_idx:
                    audio = audio[start_idx:end_idx]
                    print(f"Trimmed audio from {len(audio)/target_sr:.2f}s to {(end_idx-start_idx)/target_sr:.2f}s")
            
            return audio
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            print(traceback.format_exc())
            raise

    def custom_transliterate(self, text, source_script, target_script=sanscript.IAST):
        """
        Enhanced transliteration with special handling for vowels and diacritics
        
        Args:
            text: Text to transliterate
            source_script: Source script
            target_script: Target script (default is IAST - Roman)
            
        Returns:
            Transliterated text
        """
        try:
            # First pass with standard transliteration
            if source_script in [sanscript.DEVANAGARI, sanscript.BENGALI, sanscript.GUJARATI,
                               sanscript.GURMUKHI, sanscript.KANNADA, sanscript.MALAYALAM,
                               sanscript.ORIYA, sanscript.TAMIL, sanscript.TELUGU]:
                transliterated = transliterate(text, source_script, target_script)
            else:
                # For scripts not supported by indic_transliteration
                return text
                
            # Second pass for special vowels and diacritics
            for char, replacement in VOWEL_MAPPINGS.items():
                transliterated = transliterated.replace(char, replacement)
                
            # Handle combining marks
            for mark, replacement in COMBINING_MARKS.items():
                transliterated = transliterated.replace(mark, replacement)
                
            return transliterated
        except Exception as e:
            print(f"Transliteration error: {e}")
            return text  # Return original if transliteration fails

    def transcribe(self, audio_path, language=None, beam_size=5):
        """
        Transcribe audio file to text in native and Roman scripts.
        """
        try:
            # If language is not specified, try to detect
            if language == "auto" or language is None:
                print("No language specified, will try to auto-detect")
                language_option = None
            else:
                print(f"Transcribing in {LANGUAGE_MAP.get(language, language)} language")
                language_option = language

            # Preprocess audio
            audio = self.preprocess_audio(audio_path)

            # Ensure audio is the correct shape and dtype expected by Whisper
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            # First, do a quick transcription to detect if this is a Sanskrit mantra
            # This helps with mantras that might be transliterated in English
            quick_options = {
                "task": "transcribe", 
                "verbose": False,
                "beam_size": 3,
                "fp16": self.device == "cuda" and self.compute_type == "float16"
            }
            
            print("Performing quick initial transcription to check for language patterns...")
            quick_result = self.model.transcribe(audio, **quick_options)
            quick_text = quick_result["text"].strip().lower()
            
            # Check for Sanskrit mantra patterns in the preliminary transcription
            is_sanskrit_mantra = False
            detected_mantra = None
            
            # Check for Sanskrit patterns in Devanagari
            for mantra in SANSKRIT_MANTRAS_DEVANAGARI:
                if mantra in quick_text:
                    is_sanskrit_mantra = True
                    detected_mantra = mantra
                    print(f"Detected Sanskrit mantra in Devanagari: '{mantra}'")
                    break
                
            # Check for Sanskrit patterns in Roman transliteration
            if not is_sanskrit_mantra:
                for mantra in SANSKRIT_MANTRAS_ROMAN:
                    if mantra in quick_text:
                        is_sanskrit_mantra = True
                        detected_mantra = mantra
                        print(f"Detected Sanskrit mantra in Roman script: '{mantra}'")
                        break
            
            # Check for specific well-known mantras
            detected_complete_mantra = None
            if "bhur bhuvah" in quick_text or "भूर्भुवः" in quick_text or "savitur" in quick_text or "सवितुर्" in quick_text:
                detected_complete_mantra = "gayatri"
                is_sanskrit_mantra = True
                print("Detected Gayatri Mantra")
            elif "tryambakam" in quick_text or "त्र्यम्बकं" in quick_text or "mrityunjaya" in quick_text:
                detected_complete_mantra = "mahamrityunjaya"
                is_sanskrit_mantra = True
                print("Detected Mahamrityunjaya Mantra")
            elif "sarve bhavantu" in quick_text or "सर्वे भवन्तु" in quick_text:
                detected_complete_mantra = "shanti"
                is_sanskrit_mantra = True
                print("Detected Shanti Mantra")
                
            # Check for Konkani language patterns
            is_konkani = False
            if any(indicator.lower() in quick_text for indicator in konkani_indicators):
                print("Detected Konkani language pattern")
                is_konkani = True
            
            # Set transcription options
            options = {
                "task": "transcribe",
                "verbose": True,
                "beam_size": beam_size,
                "best_of": min(beam_size + 2, 5),
                "fp16": self.device == "cuda" and self.compute_type == "float16",
                "temperature": 0.0,
                "suppress_blank": True,
                "condition_on_previous_text": True,
                "patience": 1.0
            }

            # Force specific language if detected
            if is_sanskrit_mantra:
                print("Forcing Sanskrit language model due to detected mantra")
                options["language"] = "sa"
                language_option = "sa"
            elif is_konkani:
                print("Setting Konkani as the detected language")
                language_option = "kok"
                # Whisper doesn't have a specific Konkani model, so use Hindi/Marathi as closest match
                options["language"] = "hi"  # Using Hindi model for Konkani
            elif language_option and language_option != "auto":
                options["language"] = language_option

            # Perform full transcription
            print("Starting full transcription...")
            result = self.model.transcribe(audio, **options)

            # Get native text
            native_text = result["text"].strip()
            detected_lang = result.get("language", "auto-detected")
            
            # Check for Konkani-specific patterns in the transcription
            if any(indicator.lower() in native_text.lower() for indicator in konkani_indicators):
                print("Detected Konkani language indicators in transcription")
                detected_lang = "kok"
            
            # Additional Sanskrit detection logic
            devanagari_pattern = any(ord('\u0900') <= ord(c) <= ord('\u097F') for c in native_text)
            sanskrit_pattern = any(indicator in native_text.lower() for indicator in sanskrit_indicators)
            
            # Check for Sanskrit patterns in the text content
            if any(mantra in native_text.lower() for mantra in SANSKRIT_MANTRAS_DEVANAGARI) or any(mantra in native_text.lower() for mantra in SANSKRIT_MANTRAS_ROMAN):
                print("Detected Sanskrit mantra in transcription, forcing Sanskrit language")
                detected_lang = "sa"
                
                # Reprocess if not already using Sanskrit
                if not is_sanskrit_mantra and detected_lang != "sa":
                    options["language"] = "sa"
                    print("Reprocessing audio with Sanskrit language model...")
                    result = self.model.transcribe(audio, **options)
                    native_text = result["text"].strip()
            
            # Special handling for mistaken English detection when text contains Devanagari
            elif detected_lang == "en" and (devanagari_pattern or sanskrit_pattern):
                print(f"Detected language was {detected_lang}, but found Devanagari/Sanskrit patterns. Forcing Sanskrit detection.")
                detected_lang = "sa"
                
                # Reprocess with Sanskrit as the language to get better results
                options["language"] = "sa"
                print("Reprocessing audio with Sanskrit language model...")
                result = self.model.transcribe(audio, **options)
                native_text = result["text"].strip()
            else:
                # Script coverage detection for improved language identification
                total_chars = len(native_text.replace(" ", ""))
                if total_chars > 0:
                    script_coverage = {}
                    for lang, (start, end) in script_ranges.items():
                        script_chars = sum(1 for c in native_text if start <= ord(c) <= end)
                        script_coverage[lang] = script_chars / total_chars
                    
                    # Check if text contains characters from any Indian script
                    detected_script = None
                    max_coverage = 0.15  # Minimum threshold to consider a script dominant
                    for lang, coverage in script_coverage.items():
                        if coverage > max_coverage:
                            detected_script = lang
                            max_coverage = coverage
                            
                    if detected_script:
                        # If the detected script doesn't match the detected language, correct it
                        if detected_lang == "en" or detected_lang not in script_ranges.keys():
                            print(f"Text contains script for {detected_script} but detected as {detected_lang}. Correcting.")
                            
                            # Special handling for languages using Devanagari script
                            if detected_script in ["hi", "mr", "sa", "kok"]:
                                # Check for Konkani indicators
                                if any(indicator.lower() in native_text.lower() for indicator in konkani_indicators):
                                    detected_lang = "kok"
                                # Check for Sanskrit indicators
                                elif detected_script in ["hi", "mr", "sa"] and sanskrit_pattern:
                                    detected_lang = "sa"
                                    
                                    # Reprocess with Sanskrit for better results
                                    options["language"] = "sa"
                                    print("Reprocessing audio with Sanskrit language model...")
                                    result = self.model.transcribe(audio, **options)
                                    native_text = result["text"].strip()
                                else:
                                    detected_lang = detected_script
                            else:
                                detected_lang = detected_script

            # For complete mantras - use the correct Sanskrit text if detected
            if detected_complete_mantra and detected_complete_mantra in COMPLETE_MANTRAS:
                print(f"Applying correct {detected_complete_mantra.capitalize()} Mantra text")
                # Set detected language to Sanskrit
                detected_lang = "sa"
                
                # Replace with the correct Sanskrit text in Devanagari
                native_text = COMPLETE_MANTRAS[detected_complete_mantra]["devanagari"]
                roman_text = COMPLETE_MANTRAS[detected_complete_mantra]["roman"]
                
                return {
                    "native": native_text,
                    "roman": roman_text,
                    "language": detected_lang,
                    "segments": result.get("segments", [])
                }
            
            # Convert to Roman script if possible
            try:
                if detected_lang in SCRIPT_MAP:
                    source_script = SCRIPT_MAP[detected_lang]
                    roman_text = self.custom_transliterate(native_text, source_script)
                else:
                    roman_text = native_text
                    print(f"No script mapping for {detected_lang}, cannot transliterate")
            except Exception as e:
                print(f"Error in transliteration: {e}")
                roman_text = native_text

            return {
                "native": native_text,
                "roman": roman_text,
                "language": detected_lang,
                "segments": result.get("segments", [])
            }
        except Exception as e:
            print(f"Error in transcription: {e}")
            print(traceback.format_exc())
            raise
            
    def _transcribe_in_chunks(self, audio, options, chunk_size_seconds=30):
        """Process long audio in chunks to reduce memory usage"""
        sample_rate = 16000  # Whisper uses 16kHz
        chunk_size = chunk_size_seconds * sample_rate
        overlap = int(1.5 * sample_rate)  # 1.5 second overlap
        
        # Split audio into chunks
        chunks = []
        for i in range(0, len(audio), chunk_size - overlap):
            chunk_end = min(i + chunk_size, len(audio))
            chunks.append(audio[i:chunk_end])
            if chunk_end == len(audio):
                break
                
        # Process each chunk
        all_segments = []
        full_text = ""
        
        print(f"Processing {len(chunks)} chunks...")
        for i, chunk in enumerate(tqdm(chunks)):
            # Clear CUDA cache before each chunk
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            # Process chunk
            result = self.model.transcribe(chunk, **options)
            
            # Adjust timestamp offsets
            chunk_offset = max(0, (i * (chunk_size - overlap)) / sample_rate)
            for segment in result["segments"]:
                segment["start"] += chunk_offset
                segment["end"] += chunk_offset
                all_segments.append(segment)
            
            # Append text (avoiding duplicates in overlap regions)
            if i == 0:
                full_text = result["text"]
            else:
                # Simple deduplication by looking for overlap in last/first few words
                last_words = full_text.split()[-5:]
                new_words = result["text"].split()[:5]
                
                overlap_point = 0
                for j in range(min(len(last_words), len(new_words))):
                    if ' '.join(last_words[-j:]) == ' '.join(new_words[:j]):
                        overlap_point = j
                
                if overlap_point > 0:
                    full_text += ' ' + ' '.join(result["text"].split()[overlap_point:])
                else:
                    full_text += ' ' + result["text"]
        
        # Create combined result
        return {
            "text": full_text.strip(),
            "segments": all_segments,
            "language": result.get("language", "")
        }

    def save_transcription(self, result, output_dir="./transcriptions", base_filename=None):
        """
        Save transcription results to files.

        Args:
            result: Transcription result dictionary
            output_dir: Directory to save output files
            base_filename: Base name for output files
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate base filename if not provided
            if base_filename is None:
                base_filename = f"transcription_{int(time.time())}"

            # Save native script text
            native_path = os.path.join(output_dir, f"{base_filename}_native.txt")
            with open(native_path, "w", encoding="utf-8") as f:
                f.write(result["native"])

            # Save Roman text
            roman_path = os.path.join(output_dir, f"{base_filename}_roman.txt")
            with open(roman_path, "w", encoding="utf-8") as f:
                f.write(result["roman"])

            # Save detailed results (including segments) as JSON
            json_path = os.path.join(output_dir, f"{base_filename}_full.json")
            with open(json_path, "w", encoding="utf-8") as f:
                # Convert segments to serializable format
                result_copy = result.copy()
                if "segments" in result_copy:
                    result_copy["segments"] = [dict(s) for s in result_copy["segments"]]

                json.dump(result_copy, f, ensure_ascii=False, indent=2)
                
            # Save SRT subtitle file for video applications
            srt_path = os.path.join(output_dir, f"{base_filename}_subtitles.srt")
            if "segments" in result and result["segments"]:
                with open(srt_path, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(result["segments"]):
                        # Convert to SRT format
                        start_time = self._format_timestamp(segment["start"])
                        end_time = self._format_timestamp(segment["end"])
                        f.write(f"{i+1}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{segment['text'].strip()}\n\n")

            print(f"Transcription saved to {output_dir}/{base_filename}_*.txt")

            return {
                "native_path": native_path,
                "roman_path": roman_path,
                "json_path": json_path,
                "srt_path": srt_path
            }
        except Exception as e:
            print(f"Error saving transcription: {e}")
            print(traceback.format_exc())
            raise
            
    def _format_timestamp(self, seconds):
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    def unload_model(self):
        """Unload model to free up memory more thoroughly"""
        try:
            if hasattr(self, 'model'):
                # Delete model explicitly 
                del self.model
                
                # Garbage collect to ensure memory is freed
                gc.collect()
                
                if torch.cuda.is_available():
                    # Empty CUDA cache
                    torch.cuda.empty_cache()
                    # Additional aggressive cleanup
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        torch.cuda.reset_peak_memory_stats()
                        
                print("Model unloaded and memory freed")
                
            # Shut down the executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception as e:
            print(f"Error unloading model: {e}")

    def _update_sanskrit_visibility(self, lang):
        """Update visibility of Sanskrit options based on language selection"""
        return {"visible": lang == "sa"}
        
    def _process_audio_for_gradio(self, audio_path, model_size, language, to_english):
        """Process audio file for Gradio interface"""
        if audio_path is None:
            return "Please upload an audio file", None, "", "", "", None, None, None
        
        try:
            # Temporary path for file from Gradio
            if isinstance(audio_path, tuple):
                sr, audio_data = audio_path
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, "audio_input.wav")
                sf.write(temp_path, audio_data, sr)
                audio_path = temp_path
            
            # Load or create a new transcriber with the selected model size
            if not hasattr(self, 'model') or self.model_size != model_size:
                # Unload existing model if any
                if hasattr(self, 'model'):
                    self.unload_model()
                # Create a new model with the selected size
                self.__init__(model_size=model_size)
            
            # Preprocess the audio and create waveform plot
            audio = self.preprocess_audio(audio_path)
            fig = plt.figure(figsize=(12, 4))
            plt.plot(np.linspace(0, len(audio)/16000, len(audio)), audio)
            plt.title("Audio Waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            
            # Add translation task if needed
            transcription_task = "transcribe"
            if to_english:
                transcription_task = "translate"
                
            # Start transcription
            start_time = time.time()
            result = self.transcribe(audio_path, language=language if language != "auto" else None)
            elapsed = time.time() - start_time
            
            # Save results
            output_dir = "./transcriptions"
            os.makedirs(output_dir, exist_ok=True)
            base_filename = f"transcription_{int(time.time())}"
            file_paths = self.save_transcription(result, output_dir, base_filename)
            
            # Prepare output message
            detected_lang = result.get("language", "unknown")
            lang_name = LANGUAGE_MAP.get(detected_lang, detected_lang)
            msg = f"✅ Transcription completed in {elapsed:.2f}s\n"
            msg += f"📌 Detected language: {lang_name}\n"
            msg += f"📊 Real-time factor: {result.get('real_time_factor', 0):.2f}x\n"
            msg += f"💾 Files saved to {output_dir}/{base_filename}_*.txt"
            
            return (
                msg,                  # Status message
                fig,                  # Audio waveform plot
                result["native"],     # Native script text
                result["roman"],      # Roman script text
                detected_lang,        # Detected language
                file_paths.get("native_path"),  # Path to native script file
                file_paths.get("roman_path"),   # Path to roman script file
                file_paths.get("json_path")     # Path to JSON file
            )
        except Exception as e:
            error_msg = f"❌ Error processing audio: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return error_msg, None, "", "", "", None, None, None

    def create_gradio_interface(self):
        """Create Gradio web interface for the transcriber"""
        with gr.Blocks(title="Indian Language Transcriber", 
                      theme=gr.themes.Soft(primary_hue="indigo")) as interface:
            gr.Markdown("# 🇮🇳 Indian Language Transcriber")
            gr.Markdown("Transcribe audio in Indian languages with native and Roman script output.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input options
                    audio_input = gr.Audio(label="Upload or Record Audio", type="filepath")
                    
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large"], 
                            value="large",
                            label="Whisper Model Size"
                        )
                        language_dropdown = gr.Dropdown(
                            choices=["auto"] + sorted(list(LANGUAGE_MAP.keys())),
                            value="auto",
                            label="Language"
                        )
                    
                    with gr.Row():
                        translate_checkbox = gr.Checkbox(
                            label="Translate to English",
                            value=False
                        )
                        
                    # Sanskrit-specific options, shown only when Sanskrit is selected
                    with gr.Row(visible=False) as sanskrit_options:
                        sanskrit_scheme = gr.Dropdown(
                            choices=list(SANSKRIT_SCHEMES.keys()),
                            value="iast",
                            label="Sanskrit Transliteration Scheme"
                        )
                    
                    # Process button
                    process_btn = gr.Button("Transcribe Audio", variant="primary")
                    
                    # Display status messages
                    status_text = gr.Textbox(label="Status", lines=3)
                    
                with gr.Column(scale=2):
                    # Output display
                    waveform_plot = gr.Plot(label="Audio Waveform")
                    
                    with gr.Tabs():
                        with gr.TabItem("Native Script"):
                            native_output = gr.Textbox(
                                label="Native Script Transcription", 
                                lines=10,
                                interactive=True
                            )
                            native_download = gr.File(label="Download Native Script Text")
                            
                        with gr.TabItem("Roman Script"):
                            roman_output = gr.Textbox(
                                label="Roman Script Transcription", 
                                lines=10,
                                interactive=True
                            )
                            roman_download = gr.File(label="Download Roman Script Text")
                            
                        with gr.TabItem("Full Data"):
                            detected_lang = gr.Textbox(label="Detected Language")
                            json_download = gr.File(label="Download Full JSON Data")
            
            # Set up event handlers
            process_btn.click(
                fn=self._process_audio_for_gradio,
                inputs=[audio_input, model_dropdown, language_dropdown, translate_checkbox],
                outputs=[status_text, waveform_plot, native_output, roman_output, detected_lang,
                        native_download, roman_download, json_download]
            )
            
            # Update Sanskrit options visibility based on language selection
            language_dropdown.change(
                fn=self._update_sanskrit_visibility,
                inputs=[language_dropdown],
                outputs=[sanskrit_options]
            )
            
            # Example audio selector
            examples_dir = Path("./examples")
            if examples_dir.exists():
                example_files = list(examples_dir.glob("*.wav")) + list(examples_dir.glob("*.mp3"))
                if example_files:
                    gr.Examples(
                        examples=[[str(f)] for f in example_files],
                        inputs=[audio_input],
                    )
            
            # Documentation at the bottom
            with gr.Accordion("Documentation", open=False):
                gr.Markdown("""
                ## About this tool
                
                The Indian Language Transcriber uses Whisper models to transcribe audio in various Indian languages.
                
                ### Features:
                - Transcription in native script and Roman transliteration
                - Support for multiple Indian languages
                - Special handling for Sanskrit transliteration with multiple schemes
                - Audio waveform visualization
                - Export to TXT, JSON, and SRT formats
                
                ### Supported Languages:
                - Hindi, Marathi, Bengali, Tamil, Telugu, Malayalam, Kannada, Gujarati
                - Punjabi, Odia, Assamese, Sanskrit
                - Many other Indian languages (automatically detected)
                
                ### Tips for best results:
                1. Use clear audio with minimal background noise
                2. Choose the appropriate language if known (or leave on auto-detect)
                3. Large model gives best results but requires more computation time
                """)
        
        return interface
    
    def launch_web_interface(self, share=True, server_name="0.0.0.0", server_port=7860):
        """Launch the Gradio web interface"""
        interface = self.create_gradio_interface()
        interface.launch(
            share=True,
            server_name=server_name,
            server_port=server_port
        )

def main():
    """Main function to start the application"""
    import argparse
    import os
    
    # Load environment variables from .env file if present
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Indian Language Transcriber")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL_SIZE", "tiny"), 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--audio", type=str, help="Path to audio file for direct transcription")
    parser.add_argument("--language", type=str, default=os.environ.get("LANGUAGE", "auto"), 
                        help="Language code (e.g., hi, mr, ta)")
    parser.add_argument("--output", type=str, default=os.environ.get("OUTPUT_DIR", "./transcriptions"), 
                        help="Output directory")
    parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", None), 
                        help="Device to use (cuda, cpu)")
    parser.add_argument("--compute_type", type=str, default=os.environ.get("COMPUTE_TYPE", "float16"),
                        choices=["float16", "float32", "int8"],
                        help="Computation type for model")
    parser.add_argument("--web", action="store_true", default=os.environ.get("WEB_INTERFACE", "false").lower() == "true",
                        help="Launch web interface")
    parser.add_argument("--share", action="store_true", default=os.environ.get("SHARE", "false").lower() == "true",
                        help="Share web interface publicly")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")),
                        help="Port for web interface")
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"),
                        help="Host for web interface")
    parser.add_argument("--local_model_path", type=str, default=os.environ.get("LOCAL_MODEL_PATH", None),
                        help="Path to a pre-downloaded model file (to avoid downloading)")
    
    args = parser.parse_args()
    
    try:
        # Create directories if they don't exist
        os.makedirs(args.output, exist_ok=True)
        os.makedirs("./models", exist_ok=True)
        
        # Create transcriber with specified model
        transcriber = IndianLanguageTranscriber(
            model_size=args.model, 
            device=args.device,
            compute_type=args.compute_type,
            local_model_path=args.local_model_path
        )
        
        if args.audio:
            # Direct transcription of a file
            print(f"Transcribing {args.audio} in {args.language} language...")
            result = transcriber.transcribe(args.audio, language=args.language if args.language != "auto" else None)
            
            # Save results
            transcriber.save_transcription(result, args.output)
            
            # Print results to console
            print("\n" + "="*50)
            print("NATIVE SCRIPT:")
            print("="*50)
            print(result["native"])
            print("\n" + "="*50)
            print("ROMAN SCRIPT:")
            print("="*50)
            print(result["roman"])
        elif args.web:
            # Launch web interface
            print(f"Starting web interface on {args.host}:{args.port}...")
            if args.share:
                print("Interface will be publicly shared")
            transcriber.launch_web_interface(share=args.share, server_port=args.port, server_name=args.host)
        else:
            # Default to web interface
            print("Starting web interface...")
            transcriber.launch_web_interface()
            
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return 1
        
    return 0

if __name__ == "__main__":
    # Create and launch the interface
    main() 