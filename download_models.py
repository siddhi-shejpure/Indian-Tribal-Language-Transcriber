import os
import argparse
import hashlib
import sys
import requests
import tqdm
import torch

# Whisper model URLs and checksums
WHISPER_MODELS = {
    "tiny": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.pt",
        "size": 150437504,
        "sha256": "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"
    },
    "base": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
        "size": 291879500,
        "sha256": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"
    },
    "small": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
        "size": 493973931,
        "sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
    },
    "medium": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        "size": 1573753797,
        "sha256": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1"
    },
    "large-v1": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
        "size": 2884733497,
        "sha256": "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a"
    },
    "large-v2": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
        "size": 2888486702,
        "sha256": "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524"
    },
    "large-v3": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b6ac424/large-v3.pt",
        "size": 2902742760,
        "sha256": "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b6ac424"
    },
    "large": {  # Currently points to large-v3
        "url": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b6ac424/large-v3.pt",
        "size": 2902742760,
        "sha256": "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b6ac424"
    }
}

# Mirror URLs for backup downloads if OpenAI's server fails
MIRROR_URLS = {
    "tiny": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
    "base": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    "small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    # Note: large models aren't available in this format from HF
}

def download_model(model_name, output_dir="./models", force_redownload=False, skip_checksum=False, use_mirror=False):
    """
    Download a Whisper model and verify its checksum.
    
    Args:
        model_name: Name of the model to download (tiny, base, small, medium, large)
        output_dir: Directory to save the model
        force_redownload: If True, redownload even if file exists
        skip_checksum: If True, skip checksum verification (not recommended)
        use_mirror: If True, try to use mirror URL if available
        
    Returns:
        Path to the downloaded model file
    """
    if model_name not in WHISPER_MODELS:
        print(f"Error: Unknown model '{model_name}'. Available models: {', '.join(WHISPER_MODELS.keys())}")
        return None
        
    model_info = WHISPER_MODELS[model_name]
    url = model_info["url"]
    expected_sha256 = model_info["sha256"]
    expected_size = model_info["size"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path - use the basename of the URL
    output_path = os.path.join(output_dir, f"{model_name}.pt")
    
    # Whether to download: force redownload, or file doesn't exist, or has wrong size
    should_download = (
        force_redownload or 
        not os.path.exists(output_path) or 
        (os.path.exists(output_path) and os.path.getsize(output_path) != expected_size)
    )
    
    # If file exists and we're not forcing redownload, verify checksum
    if not should_download and not skip_checksum:
        if os.path.getsize(output_path) == expected_size:
            # Verify checksum
            print(f"Verifying checksum for {output_path}...")
            sha256_hash = hashlib.sha256()
            with open(output_path, "rb") as f:
                # Read in chunks to avoid memory issues with large files
                for byte_block in tqdm.tqdm(iter(lambda: f.read(4096*1024), b""), 
                                         total=expected_size/(4096*1024),
                                         unit="MB", unit_scale=True):
                    sha256_hash.update(byte_block)
                    
            if sha256_hash.hexdigest() == expected_sha256:
                print(f"✅ Model file already exists and checksum is valid: {output_path}")
                return output_path
            else:
                print(f"❌ Model file exists but checksum is invalid. Will re-download.")
                should_download = True
    
    # Create/update a dummy file for immediate Docker usage
    print("Creating a dummy model file for immediate Docker usage (will be replaced)...")
    with open(output_path, 'w') as f:
        f.write("PLACEHOLDER_MODEL_FILE")
    
    # If using skip_checksum and file exists, just return it
    if skip_checksum and os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:
        print(f"⚠️ Using existing model file without checksum verification: {output_path}")
        return output_path
    
    # Check if we should try to download
    if should_download:
        # Try multiple URLs
        urls_to_try = [url]
        
        # Add mirror URL if requested and available
        if use_mirror and model_name in MIRROR_URLS:
            urls_to_try.append(MIRROR_URLS[model_name])
            print(f"Will try mirror URL if primary fails: {MIRROR_URLS[model_name]}")
            
        # Try each URL in order
        for download_url in urls_to_try:
            print(f"Downloading {download_url} to {output_path}...")
            try:
                download_with_progress(download_url, output_path)
                
                # Verify the downloaded file if not skipping checksum
                if not skip_checksum:
                    print(f"Verifying downloaded file...")
                    sha256_hash = hashlib.sha256()
                    with open(output_path, "rb") as f:
                        for byte_block in tqdm.tqdm(iter(lambda: f.read(4096*1024), b""), 
                                                  total=os.path.getsize(output_path)/(4096*1024),
                                                  unit="MB", unit_scale=True):
                            sha256_hash.update(byte_block)
                    
                    if sha256_hash.hexdigest() != expected_sha256:
                        print(f"❌ Downloaded file has invalid checksum!")
                        if download_url == urls_to_try[-1]:  # If this was the last URL to try
                            if skip_checksum:
                                print(f"⚠️ Using file despite invalid checksum (--skip-checksum flag)")
                                return output_path
                            return None
                        else:
                            print(f"Trying next URL...")
                            continue
                
                print(f"✅ Model downloaded and verified successfully: {output_path}")
                return output_path
                
            except Exception as e:
                print(f"Error downloading from {download_url}: {e}")
                if download_url == urls_to_try[-1]:  # If this was the last URL to try
                    print(f"All download attempts failed.")
                    return None
                else:
                    print(f"Trying next URL...")
    else:
        print(f"Using existing model file: {output_path}")
        return output_path

def download_with_progress(url, output_path):
    """Download a file with a progress bar."""
    # Download the file with progress bar
    response = requests.get(url, stream=True, timeout=90)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    temp_path = output_path + ".tmp"
    
    try:
        with open(temp_path, 'wb') as f:
            with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
        
        # Rename temp file to final file
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)
        
    except Exception as e:
        print(f"Error during download: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def can_load_model(model_path):
    """
    Test if a model file can be loaded with PyTorch.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if model can be loaded, False otherwise
    """
    try:
        print(f"Testing if model can be loaded: {model_path}")
        # Load just the model metadata without loading weights
        _ = torch.load(model_path, map_location="cpu")
        print(f"✅ Model can be loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def create_dummy_model(output_dir, model_name):
    """Create a dummy model file for testing Docker setup."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.pt")
    
    try:
        with open(output_path, 'w') as f:
            f.write("DUMMY_MODEL_FILE_FOR_TESTING")
        print(f"Created dummy model file at {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating dummy file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download Whisper models for offline use")
    parser.add_argument("--model", type=str, default="tiny", 
                      choices=list(WHISPER_MODELS.keys()),
                      help="Model size to download")
    parser.add_argument("--output-dir", type=str, default="./models",
                      help="Directory to save the model")
    parser.add_argument("--force", action="store_true",
                      help="Force redownload even if file exists")
    parser.add_argument("--skip-checksum", action="store_true",
                      help="Skip checksum verification (not recommended)")
    parser.add_argument("--use-mirror", action="store_true",
                      help="Try mirror URLs if primary fails")
    parser.add_argument("--create-dummy", action="store_true",
                      help="Just create a dummy model file for testing Docker")
    
    args = parser.parse_args()
    
    # Create dummy model if requested
    if args.create_dummy:
        dummy_path = create_dummy_model(args.output_dir, args.model)
        if dummy_path:
            print(f"\nDummy model created at: {dummy_path}")
            print(f"Use with: --local_model_path {dummy_path} --model {args.model}")
            return 0
        return 1
    
    # Download the model
    model_path = download_model(
        args.model, 
        args.output_dir, 
        force_redownload=args.force,
        skip_checksum=args.skip_checksum,
        use_mirror=args.use_mirror
    )
    
    if model_path:
        can_load = can_load_model(model_path)
        if can_load or args.skip_checksum:
            print("\nModel downloaded and ready for use.")
            print(f"To use this model with the transcriber, use the following argument:")
            print(f"--local_model_path {model_path}")
            return 0
    
    print("\nModel download or verification failed.")
    print("Try with --skip-checksum if you're experiencing connectivity issues.")
    print("Or try with --use-mirror to attempt alternative download sources.")
    return 1

if __name__ == "__main__":
    sys.exit(main()) 