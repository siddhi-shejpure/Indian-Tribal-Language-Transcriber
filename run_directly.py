"""
Simple script to run the Indian Language Transcriber directly without Docker.
This creates a dummy model and runs the app with minimal requirements.
"""

import os
import sys
import subprocess
import platform

def create_dummy_model():
    """Create a dummy model file for the tiny model."""
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "tiny.pt")
    
    # Check if model exists
    if not os.path.exists(model_path):
        try:
            # Try to create using the download script
            if os.path.exists("download_models.py"):
                print("Creating dummy model using download_models.py...")
                subprocess.run([sys.executable, "download_models.py", "--create-dummy", "--model", "tiny"])
            else:
                # Create a simple dummy file
                print("Creating basic dummy model file...")
                with open(model_path, "w") as f:
                    f.write("DUMMY_MODEL_FILE")
            
            if os.path.exists(model_path):
                print(f"✅ Created dummy model at {model_path}")
            else:
                print(f"❌ Failed to create dummy model")
                return False
        except Exception as e:
            print(f"Error creating dummy model: {e}")
            return False
    else:
        print(f"Dummy model already exists at {model_path}")
    
    return True

def run_app():
    """Run the application with the tiny model."""
    cmd = [
        sys.executable, 
        "mapping.py", 
        "--web", 
        "--model", "tiny", 
        "--local_model_path", os.path.join("models", "tiny.pt")
    ]
    
    print("\n=============================================")
    print("Starting Indian Language Transcriber...")
    print("Using tiny model to avoid performance issues")
    print("=============================================\n")
    
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running application: {e}")
        return False

def main():
    """Main function."""
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("transcriptions", exist_ok=True)
    
    # Create dummy model
    if not create_dummy_model():
        print("Failed to create dummy model. Exiting.")
        return 1
    
    # Run the application
    if not run_app():
        print("Application exited with an error.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 