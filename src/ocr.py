"""OCR processing for Tamil PDFs using Tesseract with enhanced configurations."""

from typing import Optional, List, Tuple, Dict
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from .config import config


class OCRError(Exception):
    """Custom exception for OCR-related errors."""
    pass


class TamilOCRProcessor:
    """Enhanced OCR processor specialized for Tamil text extraction with multiple recognition strategies."""
    
    def __init__(self, dpi: int = None):
        self.dpi = dpi or config.ocr_dpi
        self.ocr_mode = config.ocr_mode
        
        # Define OCR configurations based on performance mode
        if self.ocr_mode == 'fast':
            # Fast mode: Single best configuration
            self.ocr_configs = {
                'default': '--oem 1 --psm 6 -c preserve_interword_spaces=1'
            }
            self.preprocessing_methods = ['original', 'grayscale']
        elif self.ocr_mode == 'thorough':
            # Thorough mode: All configurations
            self.ocr_configs = {
                'default': '--oem 1 --psm 6 -c preserve_interword_spaces=1',
                'single_column': '--oem 1 --psm 4 -c preserve_interword_spaces=1',
                'text_block': '--oem 1 --psm 6 -c preserve_interword_spaces=1',
                'single_line': '--oem 1 --psm 13 -c preserve_interword_spaces=1',
                'legacy_engine': '--oem 0 --psm 6 -c preserve_interword_spaces=1',
                'auto_segment': '--oem 1 --psm 3 -c preserve_interword_spaces=1'
            }
            self.preprocessing_methods = ['original', 'grayscale', 'enhanced_contrast', 'sharpened', 'threshold', 'morphological']
        else:  # balanced mode (default)
            # Balanced mode: 3 best configurations
            self.ocr_configs = {
                'default': '--oem 1 --psm 6 -c preserve_interword_spaces=1',
                'single_column': '--oem 1 --psm 4 -c preserve_interword_spaces=1',
                'auto_segment': '--oem 1 --psm 3 -c preserve_interword_spaces=1'
            }
            self.preprocessing_methods = ['original', 'grayscale', 'enhanced_contrast']
        
        if config.verbose_logging:
            print(f"🔧 OCR Configuration:")
            print(f"   DPI: {self.dpi}")
            print(f"   Performance Mode: {self.ocr_mode}")
            print(f"   Configurations: {len(self.ocr_configs)}")
            print(f"   Preprocessing methods: {len(self.preprocessing_methods)}")
            total_attempts = len(self.ocr_configs) * len(self.preprocessing_methods)
            print(f"   Max attempts per page: {total_attempts}")
    
    def process_pdf(
        self, 
        pdf_path: str, 
        start_page: Optional[int] = None, 
        end_page: Optional[int] = None,
        use_enhanced_ocr: bool = True
    ) -> List[Tuple[int, str]]:
        """
        Extract Tamil text from PDF pages using enhanced OCR techniques.
        
        Args:
            pdf_path: Path to the PDF file
            start_page: First page to process (1-indexed)
            end_page: Last page to process (1-indexed)
            use_enhanced_ocr: Whether to use multiple OCR attempts and preprocessing
            
        Returns:
            List of tuples (page_number, extracted_text)
            
        Raises:
            OCRError: If OCR processing fails
        """
        try:
            images, page_offset = self._convert_pdf_to_images(
                pdf_path, start_page, end_page
            )
            
            print(f"📄 Total pages to process: {len(images)}")
            if use_enhanced_ocr:
                print(f"🔍 Using enhanced OCR with {len(self.ocr_configs)} configurations")
            
            extracted_pages = []
            total_chars = 0
            
            for i, image in enumerate(images, 1):
                page_num = i + page_offset
                print(f"OCR processing page {page_num}...", end='\r')
                
                if use_enhanced_ocr:
                    text, confidence = self._extract_text_enhanced(image)
                else:
                    text = self._extract_text_from_image(image)
                    confidence = 0
                
                if text.strip():
                    extracted_pages.append((page_num, text.strip()))
                    total_chars += len(text.strip())
                    
                    if config.verbose_logging and confidence > 0:
                        print(f"\n   Page {page_num}: {len(text.strip())} chars, confidence: {confidence:.1f}%")
            
            print(f"\n✅ OCR processing complete! Extracted text from {len(extracted_pages)} pages")
            print(f"📊 Total characters extracted: {total_chars:,}")
            
            if not extracted_pages:
                raise OCRError("No text extracted from any pages")
                
            return extracted_pages
            
        except Exception as e:
            if isinstance(e, OCRError):
                raise
            raise OCRError(f"OCR processing failed: {e}")
    
    def _convert_pdf_to_images(
        self, 
        pdf_path: str, 
        start_page: Optional[int], 
        end_page: Optional[int]
    ) -> Tuple[List[Image.Image], int]:
        """Convert PDF pages to images for OCR processing."""
        print("Converting PDF to images...")
        
        if start_page and end_page:
            print(f"Processing pages {start_page} to {end_page}")
            images = convert_from_path(
                pdf_path, 
                first_page=start_page, 
                last_page=end_page,
                dpi=self.dpi
            )
            page_offset = start_page - 1
        else:
            images = convert_from_path(pdf_path, dpi=self.dpi)
            page_offset = 0
        
        return images, page_offset
    
    def _extract_text_enhanced(self, image: Image.Image) -> Tuple[str, float]:
        """Extract Tamil text using optimized OCR configurations and preprocessing."""
        best_text = ""
        best_confidence = 0
        results = {}
        
        # Try different preprocessing approaches
        processed_images = self._preprocess_image(image)
        
        # For fast/balanced mode, try to exit early if we get a good result
        confidence_threshold = 85 if self.ocr_mode == 'fast' else 80
        min_chars = 50 if self.ocr_mode == 'fast' else 30
        
        for preprocess_name, processed_img in processed_images.items():
            for config_name, config_str in self.ocr_configs.items():
                try:
                    # Get text and confidence
                    text = pytesseract.image_to_string(
                        processed_img, 
                        lang='tam',
                        config=config_str
                    )
                    
                    # Get confidence data
                    data = pytesseract.image_to_data(
                        processed_img, 
                        lang='tam',
                        config=config_str,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Calculate average confidence (excluding -1 values)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > -1]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Store result
                    key = f"{preprocess_name}_{config_name}"
                    results[key] = {
                        'text': text.strip(),
                        'confidence': avg_confidence,
                        'char_count': len(text.strip())
                    }
                    
                    # Update best result
                    if (avg_confidence > best_confidence and len(text.strip()) > min_chars) or \
                       (avg_confidence > best_confidence * 0.9 and len(text.strip()) > len(best_text) * 1.2):
                        best_text = text.strip()
                        best_confidence = avg_confidence
                    
                    # Early exit for fast/balanced mode if we get a good result
                    if (self.ocr_mode in ['fast', 'balanced'] and 
                        avg_confidence > confidence_threshold and 
                        len(text.strip()) > min_chars):
                        if config.verbose_logging:
                            print(f"   ✅ Early exit: {avg_confidence:.1f}% confidence, {len(text.strip())} chars")
                        return text.strip(), avg_confidence
                        
                except Exception as e:
                    if config.debug_mode:
                        print(f"OCR failed for {preprocess_name}_{config_name}: {e}")
                    continue
        
        # If no good result found, use the longest text found
        if not best_text and results:
            best_result = max(results.values(), key=lambda x: x['char_count'])
            best_text = best_result['text']
            best_confidence = best_result['confidence']
        
        if config.debug_mode:
            print(f"\n🔍 OCR Results Summary ({self.ocr_mode} mode):")
            for key, result in results.items():
                print(f"   {key}: {result['char_count']} chars, {result['confidence']:.1f}% confidence")
            print(f"   🏆 Best: {len(best_text)} chars, {best_confidence:.1f}% confidence")
        
        return best_text, best_confidence
    
    def _preprocess_image(self, image: Image.Image) -> Dict[str, Image.Image]:
        """Apply selected preprocessing techniques based on OCR mode."""
        processed_images = {}
        
        for method in self.preprocessing_methods:
            if method == 'original':
                processed_images['original'] = image
            elif method == 'grayscale':
                gray_img = image.convert('L')
                processed_images['grayscale'] = gray_img
            elif method == 'enhanced_contrast':
                gray_img = image.convert('L')
                enhancer = ImageEnhance.Contrast(gray_img)
                enhanced_img = enhancer.enhance(2.0)
                processed_images['enhanced_contrast'] = enhanced_img
            elif method == 'sharpened':
                gray_img = image.convert('L')
                enhancer = ImageEnhance.Contrast(gray_img)
                enhanced_img = enhancer.enhance(2.0)
                sharpened_img = enhanced_img.filter(ImageFilter.SHARPEN)
                processed_images['sharpened'] = sharpened_img
            elif method in ['threshold', 'morphological']:
                # Advanced OpenCV preprocessing (only in thorough mode)
                try:
                    gray_img = image.convert('L')
                    cv_img = cv2.cvtColor(np.array(gray_img), cv2.COLOR_RGB2BGR)
                    
                    if method == 'threshold':
                        blurred = cv2.GaussianBlur(cv_img, (3, 3), 0)
                        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        processed_images['threshold'] = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
                    elif method == 'morphological':
                        blurred = cv2.GaussianBlur(cv_img, (3, 3), 0)
                        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                        processed_images['morphological'] = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
                        
                except ImportError:
                    if config.verbose_logging and method in ['threshold', 'morphological']:
                        print(f"⚠️  OpenCV not available, skipping {method} preprocessing")
                except Exception as e:
                    if config.debug_mode:
                        print(f"{method} preprocessing failed: {e}")
        
        return processed_images
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract Tamil text from a single image using Tesseract OCR (simple method)."""
        return pytesseract.image_to_string(
            image, 
            lang='tam',
            config=self.ocr_configs['default']
        )
    
    def _convert_pdf_to_images(
        self, 
        pdf_path: str, 
        start_page: Optional[int], 
        end_page: Optional[int]
    ) -> Tuple[List[Image.Image], int]:
        """Convert PDF pages to images for OCR processing."""
        print("📄 Converting PDF to images...")
        
        if start_page and end_page:
            print(f"📋 Processing pages {start_page} to {end_page}")
            images = convert_from_path(
                pdf_path, 
                first_page=start_page, 
                last_page=end_page,
                dpi=self.dpi
            )
            page_offset = start_page - 1
        else:
            images = convert_from_path(pdf_path, dpi=self.dpi)
            page_offset = 0
        
        return images, page_offset