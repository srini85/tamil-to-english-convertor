"""Tamil text translation service using Google Gemini API."""

import time
import os
from typing import List

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiTranslationError(Exception):
    """Custom exception for Gemini translation-related errors."""
    pass


class GeminiTranslator:
    """High-accuracy Tamil to English translator using Google Gemini API."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI library not available. "
                "Install with: pip install google-generativeai"
            )
        
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter. Get your key from: https://makersuite.google.com/app/apikey"
            )
        
        # Get model name from environment or use default
        self.model_name = model_name or os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # Create the prompt template for better Tamil translation
        self.system_prompt = """You are an expert Tamil to English translator with deep understanding of Tamil literature, culture, and context. 

Your task is to translate Tamil text to natural, fluent English while:
1. Preserving the original meaning and context
2. Maintaining the tone and style (formal, informal, poetic, etc.)
3. Using appropriate English equivalents for Tamil cultural concepts
4. Ensuring the translation reads naturally in English
5. Preserving any spiritual or religious terminology appropriately

Please provide only the English translation without any additional commentary or explanations."""
        
        # List available models and display them
        self._list_available_models()
        
        self._test_connection()
    
    def _list_available_models(self):
        """List and display available Gemini models."""
        try:
            print("🤖 Fetching available Gemini models...")
            models = list(self.client.models.list())
            
            print("📋 Available Gemini models:")
            generative_models = []
            for model in models:
                # Filter for generative models (basic text generation capability)
                if hasattr(model, 'name') and 'gemini' in model.name.lower():
                    generative_models.append(model)
                    clean_name = model.name.replace('models/', '') if hasattr(model, 'name') else str(model)
                    status = "✓ CURRENT" if clean_name == self.model_name else ""
                    print(f"   • {clean_name} {status}")
            
            if not generative_models:
                print("   ⚠️  No generative models found")
            
            print(f"🎯 Using model: {self.model_name}")
            print()
            
        except Exception as e:
            print(f"⚠️  Could not fetch available models: {e}")
            print(f"🎯 Using configured model: {self.model_name}")
            print()
    
    def _test_connection(self):
        """Test API connection with a simple translation."""
        try:
            test_prompt = f"{self.system_prompt}\n\nTranslate this Tamil text to English: வணக்கம்"
            response = self.model.generate_content(test_prompt)
            if not response.text:
                raise Exception("Empty response from Gemini API")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Gemini API: {e}")
    
    def translate_text(self, text: str, chunk_size: int = 8000) -> str:
        """
        Translate Tamil text to English using Gemini API with intelligent chunking.
        
        Args:
            text: Tamil text to translate
            chunk_size: Maximum characters per API request (Gemini has higher limits)
            
        Returns:
            Translated English text
            
        Raises:
            GeminiTranslationError: If translation fails completely
        """
        if not text.strip():
            return ""
        
        chunks = self._split_text(text, chunk_size)
        translated_chunks = []
        failed_chunks = 0
        
        for i, chunk in enumerate(chunks, 1):
            if chunk.strip():
                try:
                    # Create translation prompt
                    prompt = f"{self.system_prompt}\n\nTranslate this Tamil text to English:\n\n{chunk}"
                    
                    # Generate translation
                    response = self.model.generate_content(prompt)
                    
                    if response.text:
                        translated_chunks.append(response.text.strip())
                    else:
                        raise Exception("Empty response from Gemini")
                    
                    if len(chunks) > 1:
                        print(f"Translated chunk {i}/{len(chunks)} (Gemini)", end='\r')
                        time.sleep(0.2)  # Rate limiting - Gemini is more generous
                        
                except Exception as e:
                    print(f"\nWarning: Gemini translation failed for chunk {i}: {e}")
                    translated_chunks.append(f"[Translation failed: {chunk[:100]}...]")
                    failed_chunks += 1
        
        if len(chunks) > 1:
            print()  # New line after progress updates
            
        if failed_chunks == len(chunks):
            raise GeminiTranslationError("All Gemini translation chunks failed")
            
        return '\n'.join(translated_chunks)
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks while preserving paragraph boundaries."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first (double newlines)
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size and we have content, start new chunk
            if len(current_chunk + paragraph + '\n\n') > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
            else:
                current_chunk += paragraph + '\n\n'
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


def is_gemini_translation_available() -> bool:
    """Check if Gemini translation dependencies are available."""
    return GEMINI_AVAILABLE