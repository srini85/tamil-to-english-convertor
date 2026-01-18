#!/usr/bin/env python3
"""Test script to demonstrate enhanced Tamil OCR capabilities."""

from src.ocr import TamilOCRProcessor
from src.config import config

def test_ocr_configurations():
    """Display available OCR configurations."""
    processor = TamilOCRProcessor()
    
    print("🔧 Enhanced Tamil OCR Configuration")
    print("=" * 50)
    print(f"📊 DPI: {processor.dpi}")
    print(f"📋 Available OCR configurations: {len(processor.ocr_configs)}")
    print()
    
    print("🎯 OCR Configuration Details:")
    for name, config_str in processor.ocr_configs.items():
        print(f"   • {name:15} : {config_str}")
    
    print()
    print("🖼️  Image Preprocessing Options:")
    print("   • Original image")
    print("   • Grayscale conversion")
    print("   • Enhanced contrast (2x)")
    print("   • Sharpening filter")
    print("   • Gaussian blur + noise reduction (with OpenCV)")
    print("   • OTSU thresholding (with OpenCV)")
    print("   • Morphological operations (with OpenCV)")
    
    print()
    print("🚀 Enhanced Features:")
    print("   • Multiple OCR engine attempts per image")
    print("   • Confidence scoring and best result selection")
    print("   • Character count optimization")
    print("   • Tamil-specific preprocessing")
    print("   • Detailed logging and debugging")
    
    print()
    print("⚙️  To enable enhanced OCR:")
    print("   1. Set ENHANCED_OCR_ENABLED=true in .env file")
    print("   2. Adjust OCR_DPI (recommended: 400-600)")
    print("   3. Install opencv-python for advanced preprocessing")
    print("   4. Use --verbose for detailed processing info")

if __name__ == "__main__":
    try:
        test_ocr_configurations()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install opencv-python numpy python-dotenv")