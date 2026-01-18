"""File handling utilities for Tamil PDF processing."""

import os
from typing import List, Tuple, Optional


class FileHandler:
    """Handles file operations for OCR and translation output."""
    
    @staticmethod
    def generate_output_filename(pdf_path: str, translated: bool = False) -> str:
        """Generate appropriate output filename based on operation type."""
        base_name = os.path.splitext(pdf_path)[0]
        suffix = "_english" if translated else "_tamil_unicode"
        return f"{base_name}{suffix}.txt"
    
    @staticmethod
    def save_text_file(file_path: str, content: str) -> None:
        """Save text content to file with UTF-8 encoding."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    @staticmethod
    def get_file_size_kb(file_path: str) -> float:
        """Get file size in kilobytes."""
        return os.path.getsize(file_path) / 1024
    
    @staticmethod
    def validate_pdf_exists(pdf_path: str) -> bool:
        """Check if PDF file exists."""
        return os.path.exists(pdf_path)


class ContentProcessor:
    """Processes and formats extracted text content."""
    
    @staticmethod
    def format_pages_content(
        pages_data: List[Tuple[int, str]], 
        translator=None, 
        translate: bool = False
    ) -> str:
        """
        Format extracted text pages as continuous content without page breaks.
        
        Args:
            pages_data: List of (page_number, text) tuples
            translator: Translation service instance
            translate: Whether to translate the content
            
        Returns:
            Formatted content string as continuous text
        """
        content_parts = []
        
        for page_num, text in pages_data:
            if translate and translator:
                print(f"Translating page {page_num}...")
                try:
                    translated_text = translator.translate_text(text)
                    content_parts.append(translated_text)
                except Exception as e:
                    print(f"Translation failed for page {page_num}: {e}")
                    content_parts.append(f"[Translation Error for page {page_num}]\n{text}")
            else:
                content_parts.append(text)
        
        # Join content with double line breaks to maintain readability
        # but without page headers
        return '\n\n'.join(content_parts)
    
    @staticmethod
    def extract_sample_content(content: str, page_offset: int = 0, max_lines: int = 8) -> List[str]:
        """Extract sample content for display purposes from continuous text."""
        lines = content.split('\n')
        sample_lines = []
        
        for line in lines:
            if line.strip():  # Only include non-empty lines
                sample_lines.append(line)
                if len(sample_lines) >= max_lines:
                    break
        
        return sample_lines