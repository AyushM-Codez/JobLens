"""
Setup script for downloading required NLTK data.
Run this after installing requirements.txt
"""

import nltk
import sys

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        print("✓ Downloaded punkt")
        
        nltk.download('stopwords', quiet=True)
        print("✓ Downloaded stopwords")
        
        nltk.download('wordnet', quiet=True)
        print("✓ Downloaded wordnet")
        
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✓ Downloaded POS tagger")
        
        print("\n✓ All NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)

