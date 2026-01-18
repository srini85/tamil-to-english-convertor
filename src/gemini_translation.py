"""Tamil text translation service using Google Gemini API (New Google Gen AI SDK)."""

import time
import os
from typing import List
from .config import config, rate_limiter

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
    """High-accuracy Tamil to English translator using Google Gemini API with the new Gen AI SDK."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Gen AI library not available. "
                "Install with: pip install google-genai"
            )
        
        # Configure the Gemini API using config
        api_key = api_key or config.gemini_api_key
        if not api_key:
            raise GeminiTranslationError(
                "GEMINI_API_KEY not found. Set it in .env file or environment variable"
            )
        
        # Set model name using config
        self.model_name = model_name or config.gemini_model
        
        # Initialize the client
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise GeminiTranslationError(f"Failed to initialize Gemini client: {e}")
        
        # System instruction for translation
        self.system_instruction = """You are a professional translator specializing in Tamil to English translation. 
Translate the provided Tamil text accurately while maintaining the original meaning, context, and tone. 
Keep proper nouns and names unchanged unless they have standard English equivalents.
Please provide only the English translation without any additional commentary or explanations."""
        
        # Show configuration info
        if config.verbose_logging:
            print(f"🔧 Gemini Configuration:")
            print(f"   Model: {self.model_name}")
            print(f"   Request delay: {config.gemini_delay_between_requests}s")
            print(f"   Chunk delay: {config.gemini_delay_between_chunks}s")
            print(f"   Max chunk size: {config.max_chunk_size}")
        
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
        """Test if the API connection works."""
        try:
            print("🔗 Testing Gemini API connection...")
            # Simple test generation
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Hello",
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.1,
                    max_output_tokens=100
                )
            )
            if response and response.text:
                print("✅ Gemini API connection successful!")
            else:
                raise GeminiTranslationError("Gemini API test returned empty response")
        except Exception as e:
            raise GeminiTranslationError(f"Gemini API connection failed: {e}")

    def translate_text(self, text: str, chunk_size: int = None) -> str:
        """
        Translate Tamil text to English using Gemini API with intelligent chunking.
        
        Args:
            text: Tamil text to translate
            chunk_size: Maximum characters per API request (uses config if not specified)
            
        Returns:
            Translated English text
            
        Raises:
            GeminiTranslationError: If translation fails completely
        """
        if not text.strip():
            return ""
        
        # Use configured chunk size if not provided
        chunk_size = chunk_size or config.max_chunk_size
        
        try:
            # Split text into manageable chunks if it's too long
            chunks = self._split_text(text, chunk_size)
            translated_chunks = []
            failed_chunks = 0
            
            print(f"🔄 Translating {len(chunks)} chunks with Gemini...")
            
            for i, chunk in enumerate(chunks, 1):
                if chunk.strip():
                    try:
                        # Rate limiting before request
                        rate_limiter.wait_if_needed('gemini', 'request')
                        rate_limiter.log_request('gemini')
                        
                        prompt = f"Translate this Tamil text to English:\n\n{chunk}"
                        
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=self.system_instruction,
                                temperature=config.translation_temperature,
                                max_output_tokens=config.translation_max_output_tokens
                            )
                        )
                        
                        if response and response.text:
                            translated_chunks.append(response.text.strip())
                            
                            if len(chunks) > 1:
                                print(f"Translated chunk {i}/{len(chunks)} (Gemini)", end='\r')
                        else:
                            raise Exception("Empty response from Gemini")
                            
                    except Exception as e:
                        print(f"\nWarning: Gemini translation failed for chunk {i}: {e}")
                        translated_chunks.append(f"[Translation failed for chunk {i}: {chunk[:100]}...]")
                        failed_chunks += 1
                    
                    # Rate limiting between chunks
                    if i < len(chunks):
                        rate_limiter.wait_if_needed('gemini', 'chunk')
            
            if len(chunks) > 1:
                print()  # New line after progress updates
                
            if failed_chunks == len(chunks):
                raise GeminiTranslationError("All Gemini translation chunks failed")
                
            return '\n\n'.join(translated_chunks)
            
        except Exception as e:
            raise GeminiTranslationError(f"Translation failed: {e}")
    
    def _split_text(self, text: str, max_chunk_size: int = None) -> List[str]:
        """Split text into chunks while preserving paragraph boundaries."""
        max_chunk_size = max_chunk_size or config.max_chunk_size
        
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first (double newlines)
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size and we have content, start new chunk
            if len(current_chunk + paragraph + '\n\n') > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
            else:
                current_chunk += paragraph + '\n\n'
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


def is_available() -> bool:
    """Check if Gemini translation is available."""
    return GEMINI_AVAILABLE and bool(config.gemini_api_key)


def translate_text(text: str) -> str:
    """Convenience function for translating text with default settings."""
    if not is_available():
        raise GeminiTranslationError(
            "Gemini translation not available. Set GEMINI_API_KEY in .env file or environment."
        )
    
    translator = GeminiTranslator()
    return translator.translate_text(text)


# Legacy function name for backward compatibility
def is_gemini_translation_available() -> bool:
    """Check if Gemini translation dependencies are available (legacy name)."""
    return is_available()