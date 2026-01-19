"""Main CLI application for Tamil PDF OCR and translation."""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ocr import TamilOCRProcessor, OCRError
from src.translation import TamilTranslator, TranslationError, is_translation_available as is_cloud_translation_available
from src.local_translation import LocalTranslator, LocalTranslationError, is_local_translation_available
from src.gemini_translation import GeminiTranslator, GeminiTranslationError, is_gemini_translation_available
from src.file_handler import FileHandler, ContentProcessor


class TamilPDFProcessor:
    """Main orchestrator for Tamil PDF processing pipeline."""
    
    def __init__(self):
        self.ocr_processor = TamilOCRProcessor()
        self.file_handler = FileHandler()
        self.content_processor = ContentProcessor()
    
    def process_pdf(
        self, 
        pdf_path: str, 
        output_path: str = None,
        start_page: int = None, 
        end_page: int = None,
        translate: bool = False,
        use_local_translation: bool = False,
        use_gemini: bool = False
    ) -> str:
        """
        Main processing pipeline for Tamil PDFs.
        
        Args:
            pdf_path: Input PDF file path
            output_path: Output file path (optional)
            start_page: Start page number (1-indexed)
            end_page: End page number (1-indexed)
            translate: Whether to translate to English
            use_local_translation: Use local translation instead of cloud
            use_gemini: Use Google Gemini for translation
            
        Returns:
            Output file path on success
            
        Raises:
            ValueError: For invalid inputs
            OCRError: For OCR failures
            TranslationError: For translation failures
        """
        # Validate inputs
        if not self.file_handler.validate_pdf_exists(pdf_path):
            raise ValueError(f"PDF file not found: {pdf_path}")
        
        if start_page and end_page and start_page > end_page:
            raise ValueError("Start page must be less than or equal to end page")
        
        # Generate output filename
        if not output_path:
            output_path = self.file_handler.generate_output_filename(pdf_path, translate)
        
        # Display processing info
        translation_type = "Local" if use_local_translation else ("Gemini" if use_gemini else "Google Cloud")
        self._display_processing_info(pdf_path, output_path, translate, translation_type)
        
        # Initialize translator if needed
        translator = None
        if translate:
            translator = self._setup_translator(use_local_translation, use_gemini)
        
        try:
            # OCR processing with enhanced mode
            from src.config import config
            extracted_pages = self.ocr_processor.process_pdf(
                pdf_path, start_page, end_page, 
                use_enhanced_ocr=config.enhanced_ocr_enabled
            )
            
            # First, save the OCR Tamil text (always save this)
            tamil_content = self.content_processor.format_pages_content(
                extracted_pages, translator=None, translate=False
            )
            tamil_output_path = self.file_handler.generate_output_filename(pdf_path, translated=False)
            self.file_handler.save_text_file(tamil_output_path, tamil_content)
            print(f"✓ Tamil OCR text saved to: {tamil_output_path}")
            
            # Then, if translation is enabled, translate and save English version
            if translate and translator:
                final_content = self.content_processor.format_pages_content(
                    extracted_pages, translator, translate=True
                )
                # Save translated content
                self.file_handler.save_text_file(output_path, final_content)
                
                # Display results for translated version
                self._display_results(output_path, final_content, translate, start_page or 1)
                return output_path
            else:
                # Display results for Tamil version only
                self._display_results(tamil_output_path, tamil_content, translate, start_page or 1)
                return tamil_output_path
            
        except (OCRError, TranslationError) as e:
            print(f"\n✗ Processing failed: {e}")
            raise
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            raise
    
    def _display_processing_info(self, pdf_path: str, output_path: str, translate: bool, translation_type: str = ""):
        """Display processing information to user."""
        print(f"OCR Processing: {pdf_path}")
        print(f"Output: {output_path}")
        if translate:
            print(f"Translation: Enabled (Tamil → English, {translation_type})")
        else:
            print("Translation: Disabled")
    
    def _setup_translator(self, use_local: bool = False, use_gemini: bool = False):
        """Initialize and test translator connection."""
        try:
            if use_local:
                translator = LocalTranslator()
                available_services = translator.get_available_services()
                if available_services:
                    print(f"✓ Local translation ready (using: {available_services[0]})")
                    return translator
                else:
                    raise LocalTranslationError("No local translation services available")
            elif use_gemini:
                api_key = os.getenv('GEMINI_API_KEY')
                translator = GeminiTranslator(api_key=api_key)
                print("✓ Google Gemini API connected")
                return translator
            else:
                # Get project ID from environment variable
                project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
                translator = TamilTranslator(project_id=project_id)
                print("✓ Google Translate API v3 connected")
                return translator
        except Exception as e:
            error_type = "Local translation" if use_local else ("Gemini translation" if use_gemini else "Google Translate")
            raise TranslationError(f"{error_type} setup failed: {e}")
    
    def translate_text_file(
        self,
        text_file_path: str,
        output_path: str = None,
        use_local_translation: bool = False,
        use_gemini: bool = False
    ) -> str:
        """
        Translate a text file directly without OCR.
        
        Args:
            text_file_path: Input text file path
            output_path: Output file path (optional)
            use_local_translation: Use local translation instead of cloud
            use_gemini: Use Google Gemini for translation
            
        Returns:
            Output file path on success
        """
        try:
            # Validate input file
            if not os.path.exists(text_file_path):
                raise ValueError(f"Text file not found: {text_file_path}")
            
            # Generate output path if not provided
            if not output_path:
                base_name = os.path.splitext(text_file_path)[0]
                output_path = f"{base_name}_english.txt"
            
            # Display processing info
            translation_type = "Local" if use_local_translation else ("Gemini" if use_gemini else "Google Cloud")
            print(f"Text Translation: {text_file_path}")
            print(f"Output: {output_path}")
            print(f"Translation: {translation_type} (Tamil → English)")
            print()
            
            # Read the text file
            print(f"📖 Reading Tamil text from: {text_file_path}")
            with open(text_file_path, 'r', encoding='utf-8') as f:
                tamil_text = f.read()
            
            if not tamil_text.strip():
                raise ValueError("Input text file is empty")
                
            print(f"📊 Text loaded: {len(tamil_text)} characters")
            
            # Setup translator
            print("🔧 Setting up translator...")
            translator = self._setup_translator(use_local_translation, use_gemini)
            
            # Translate the text
            print("🔄 Translating text...")
            if use_gemini and hasattr(translator, 'translate_document'):
                # Use document translation for Gemini
                english_text = translator.translate_document(tamil_text, os.path.basename(text_file_path))
            else:
                # Use regular translation for other translators
                english_text = translator.translate_text(tamil_text)
            
            if not english_text.strip():
                raise TranslationError("Translation resulted in empty text")
            
            print(f"✓ Translation completed: {len(english_text)} characters")
            
            # Save translated content
            print(f"💾 Saving English translation...")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(english_text)
            
            # Display results
            self._display_results(output_path, english_text, True, 1)
            return output_path
            
        except (TranslationError, LocalTranslationError, GeminiTranslationError) as e:
            print(f"\n✗ Translation failed: {e}")
            raise
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            raise

    def _display_results(self, output_path: str, content: str, translated: bool, page_offset: int):
        """Display processing results and sample content."""
        file_size_kb = self.file_handler.get_file_size_kb(output_path)
        print(f"✓ Output saved to: {output_path}")
        print(f"✓ File size: {file_size_kb:.2f} KB")
        
        # Show sample content
        print(f"\n--- Content Sample ---")
        sample_lines = self.content_processor.extract_sample_content(content, page_offset)
        for line in sample_lines:
            print(line)


def create_argument_parser():
    """Create and configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description='OCR Tamil PDF to Unicode text with optional English translation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OCR only (Tamil Unicode)
  python main.py book.pdf
  
  # OCR + Translation to English (Google Cloud)
  python main.py book.pdf --translate
  
  # OCR + Translation to English (Google Gemini - Better Quality)
  python main.py book.pdf --translate --gemini
  
  # OCR + Translation to English (Local/Free)
  python main.py book.pdf --translate --local
  
  # Process specific pages with Gemini translation
  python main.py book.pdf --start 1 --end 5 --translate --gemini
  
  # Custom output file
  python main.py book.pdf output.txt --translate --local
  
  # Translate text file directly (no OCR)
  python main.py tamil_text.txt --text-only --translate --gemini
  
  # Translate with local translation
  python main.py tamil_text.txt --text-only --translate --local

Requirements for translation:
  # For Google Cloud:
  pip install google-cloud-translate
  Set up Google Cloud credentials (GOOGLE_APPLICATION_CREDENTIALS)
  Set project ID (GOOGLE_CLOUD_PROJECT environment variable)
  
  # For Google Gemini (recommended for quality):
  pip install google-genai
  Set API key (GEMINI_API_KEY environment variable)
  Get key from: https://makersuite.google.com/app/apikey
  
  # For local translation (free):
  pip install transformers torch sentencepiece
  # or: pip install argostranslate (for fully offline)
        """
    )
    
    parser.add_argument('input_file', help='Input PDF or text file path')
    parser.add_argument('output_file', nargs='?', help='Output text file path (optional)')
    parser.add_argument('--start', type=int, help='Start page number (1-indexed, PDF only)')
    parser.add_argument('--end', type=int, help='End page number (1-indexed, PDF only)')
    parser.add_argument('--translate', action='store_true', 
                       help='Translate Tamil text to English')
    parser.add_argument('--text-only', action='store_true',
                       help='Process text file directly without OCR (requires .txt file and --translate)')
    parser.add_argument('--local', action='store_true',
                       help='Use local translation instead of Google Translate (requires --translate)')
    parser.add_argument('--gemini', action='store_true',
                       help='Use Google Gemini for translation (requires --translate and GEMINI_API_KEY)')
    
    return parser


def main():
    """Main application entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate text-only mode requirements
    if args.text_only:
        if not args.translate:
            print("Error: --text-only requires --translate option!")
            print("Text-only mode is for translating text files directly without OCR.")
            print("Example: python main.py tamil_text.txt --text-only --translate --gemini")
            sys.exit(1)
        
        # Check file extension
        if not args.input_file.lower().endswith('.txt'):
            print("Error: --text-only mode requires a .txt file as input!")
            print(f"Provided file: {args.input_file}")
            print("Example: python main.py tamil_text.txt --text-only --translate --gemini")
            sys.exit(1)
        
        # Page options don't make sense for text files
        if args.start or args.end:
            print("Warning: --start and --end options are ignored in text-only mode.")
    
    # Validate translation requirements
    if args.translate:
        # Check for conflicting translation options
        translation_options = sum([args.local, args.gemini, not (args.local or args.gemini)])
        if args.local and args.gemini:
            print("Error: Cannot use both --local and --gemini options together!")
            print("Choose one translation method:")
            print("  --local   : Use local/offline translation")
            print("  --gemini  : Use Google Gemini API")
            print("  (none)    : Use Google Cloud Translate")
            sys.exit(1)
        
        if args.local:
            if not is_local_translation_available():
                print("Error: Local translation requested but no local translation services available!")
                print("Install options:")
                print("  pip install transformers torch sentencepiece  # For HuggingFace models")
                print("  pip install argostranslate                    # For Argos Translate (fully offline)")
                print("Or use other translation: python main.py <file> --translate --gemini")
                sys.exit(1)
        elif args.gemini:
            if not is_gemini_translation_available():
                print("Error: Gemini translation requested but google-generativeai not installed!")
                print("Install with: pip install google-generativeai")
                print("Set API key: export GEMINI_API_KEY='your-api-key'")
                print("Get API key from: https://makersuite.google.com/app/apikey")
                print("Or use other translation: python main.py <file> --translate --local")
                sys.exit(1)
        else:
            if not is_cloud_translation_available():
                print("Error: Cloud translation requested but google-cloud-translate not installed!")
                print("Install with: pip install google-cloud-translate")
                print("Set up Google Cloud credentials: https://cloud.google.com/docs/authentication/getting-started")
                print("Set project ID: export GOOGLE_CLOUD_PROJECT='your-project-id'")
                print("Or use alternative: python main.py <file> --translate --gemini")
                sys.exit(1)
    
    try:
        processor = TamilPDFProcessor()
        
        if args.text_only:
            # Process text file directly without OCR
            result = processor.translate_text_file(
                args.input_file,
                args.output_file,
                args.local,
                args.gemini
            )
        else:
            # Process PDF with OCR
            result = processor.process_pdf(
                args.input_file,
                args.output_file, 
                args.start,
                args.end,
                args.translate,
                args.local,
                args.gemini
            )
        
        # Success message
        print(f"\n🎉 Processing completed successfully!")
        if args.text_only:
            translation_type = "Local" if args.local else ("Gemini" if args.gemini else "Google Cloud")
            print(f"📖 English translation ({translation_type}) saved to: {result}")
        elif args.translate:
            translation_type = "Local" if args.local else ("Gemini" if args.gemini else "Google Cloud")
            # Generate Tamil filename for display
            base_name = os.path.splitext(args.input_file)[0]
            tamil_file = f"{base_name}_tamil_unicode.txt"
            print(f"📝 Tamil OCR text saved to: {tamil_file}")
            print(f"📖 English translation ({translation_type}) saved to: {result}")
        else:
            print(f"📝 Tamil Unicode text saved to: {result}")
            
    except (ValueError, OCRError, TranslationError, LocalTranslationError, GeminiTranslationError) as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏸️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()