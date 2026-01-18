"""Tamil text translation service using Google Cloud Translate API v3."""

import time
import os
from typing import List
from .config import config, rate_limiter

try:
    from google.cloud import translate_v3 as translate
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False


class TranslationError(Exception):
    """Custom exception for translation-related errors."""
    pass


class TamilTranslator:
    """High-accuracy Tamil to English translator using Google Cloud API v3."""
    
    def __init__(self, project_id: str = None, location: str = "global"):
        if not GOOGLE_TRANSLATE_AVAILABLE:
            raise ImportError(
                "Google Cloud Translate not available. "
                "Install with: pip install google-cloud-translate"
            )
        
        # Get project ID from config
        self.project_id = project_id or config.google_cloud_project
        if not self.project_id:
            raise ValueError(
                "Project ID not found. Set GOOGLE_CLOUD_PROJECT in .env file or environment variable"
            )
        
        self.location = location
        self.client = translate.TranslationServiceClient()
        self.parent = f"projects/{self.project_id}/locations/{self.location}"
        
        # Show configuration info
        if config.verbose_logging:
            print(f"🔧 Google Translate Configuration:")
            print(f"   Project: {self.project_id}")
            print(f"   Location: {self.location}")
            print(f"   Request delay: {config.google_translate_delay_between_requests}s")
            print(f"   Chunk delay: {config.google_translate_delay_between_chunks}s")
        
        self._test_connection()
    
    def _test_connection(self):
        """Test API connection with a simple translation."""
        try:
            response = self.client.translate_text(
                parent=self.parent,
                contents=["test"],
                target_language_code='en',
                source_language_code='ta'
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Google Translate API v3: {e}")
    
    def translate_text(self, text: str, chunk_size: int = None) -> str:
        """
        Translate Tamil text to English with intelligent chunking.
        
        Args:
            text: Tamil text to translate
            chunk_size: Maximum characters per API request (uses config if not specified)
            
        Returns:
            Translated English text
            
        Raises:
            TranslationError: If translation fails completely
        """
        if not text.strip():
            return ""
        
        # Use configured chunk size if not provided
        chunk_size = chunk_size or config.max_chunk_size
        
        chunks = self._split_text(text, chunk_size)
        translated_chunks = []
        failed_chunks = 0
        
        print(f"🔄 Translating {len(chunks)} chunks with Google Cloud Translate...")
        
        for i, chunk in enumerate(chunks, 1):
            if chunk.strip():
                try:
                    # Rate limiting before request
                    rate_limiter.wait_if_needed('google_translate', 'request')
                    rate_limiter.log_request('google_translate')
                    
                    response = self.client.translate_text(
                        parent=self.parent,
                        contents=[chunk],
                        target_language_code='en',
                        source_language_code='ta'
                    )
                    translated_chunks.append(response.translations[0].translated_text)
                    
                    if len(chunks) > 1:
                        print(f"Translated chunk {i}/{len(chunks)}", end='\r')
                        
                        # Rate limiting between chunks
                        if i < len(chunks):
                            rate_limiter.wait_if_needed('google_translate', 'chunk')
                        
                except Exception as e:
                    print(f"\nWarning: Translation failed for chunk {i}: {e}")
                    translated_chunks.append(f"[Translation failed: {chunk[:100]}...]")
                    failed_chunks += 1
        
        if len(chunks) > 1:
            print()  # New line after progress updates
            
        if failed_chunks == len(chunks):
            raise TranslationError("All translation chunks failed")
            
        return '\n'.join(translated_chunks)
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks while preserving line boundaries."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        for line in text.split('\n'):
            if len(current_chunk + line + '\n') > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


def is_translation_available() -> bool:
    """Check if translation dependencies are available."""
    return GOOGLE_TRANSLATE_AVAILABLE and bool(config.google_cloud_project)