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
            print(f"   Translation mode: {config.gemini_translation_mode}")
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

    def translate_document(self, text: str, document_title: str = "Tamil Document") -> str:
        """
        Translate an entire Tamil document to English in one request for better context.
        
        This method sends the entire document to Gemini at once, providing maximum
        context for more coherent and accurate translation. Best for documents 
        under 100K characters.
        
        Args:
            text: Complete Tamil text to translate
            document_title: Optional title for better context
            
        Returns:
            Complete translated English text
            
        Raises:
            GeminiTranslationError: If translation fails
        """
        if not text.strip():
            return ""
        
        try:
            # Rate limiting before request
            rate_limiter.wait_if_needed('gemini', 'request')
            rate_limiter.log_request('gemini')
            
            # Enhanced prompt for full document translation with content policy considerations
            prompt = f"""Please provide a professional English translation of this Tamil literary text.
            
This is a cultural/literary document that needs accurate translation while preserving the narrative structure.
            
Translation Guidelines:
            - Provide direct, professional translation from Tamil to English
            - Maintain the original narrative flow and structure
            - Preserve proper nouns and character names
            - Keep the literary tone and style
            - This is educational/cultural content requiring translation services
            
Tamil Literary Text:
            
{text}"""
            
            print(f"🔄 Translating complete document with Gemini ({len(text)} characters)...")
            
            # Check estimated input tokens (Tamil text is roughly 1.5-2 tokens per character)
            estimated_input_tokens = int(len(text) * 1.7)  # Conservative estimate for Tamil
            
            # Calculate appropriate max tokens based on input size
            # Estimate: Tamil to English usually expands by 1.2-1.5x in character count
            # 1 token ≈ 3-4 characters for English, so we need generous token allowance
            estimated_output_tokens = min(
                config.translation_document_max_output_tokens,
                max(15000, int(len(text) * 2.0 / 3))  # More generous estimate
            )
            
            if config.verbose_logging:
                print(f"   📊 Input: {len(text)} chars, Est. input tokens: {estimated_input_tokens}")
                print(f"   📊 Estimated output tokens needed: {estimated_output_tokens}")
                print(f"   📊 Max output tokens configured: {config.translation_document_max_output_tokens}")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=config.translation_temperature,
                    max_output_tokens=estimated_output_tokens
                )
            )
            
            # Always debug the response in detail
            self._debug_gemini_response(response, "Document Translation")
            
            if response and response.text:
                translated_text = response.text.strip()
                
                # Check for potential truncation by comparing input/output ratio
                input_chars = len(text)
                output_chars = len(translated_text)
                expected_min_output = input_chars * 0.7  # Tamil to English usually similar length
                
                if output_chars < expected_min_output:
                    print(f"⚠️  Warning: Translation may be incomplete!")
                    print(f"   📊 Input: {input_chars} chars → Output: {output_chars} chars")
                    print(f"   💡 Consider splitting document or increasing TRANSLATION_DOCUMENT_MAX_OUTPUT_TOKENS")
                
                print(f"✓ Document translation completed ({output_chars} characters)")
                return translated_text
            else:
                # Detailed error analysis
                error_details = self._analyze_failed_response(response, "document translation")
                raise Exception(f"Empty response from Gemini: {error_details}")
                
        except Exception as e:
            raise GeminiTranslationError(f"Full document translation failed: {e}")

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
                        
                        # Debug chunk response if verbose or if it fails
                        if not (response and response.text) or config.verbose_logging:
                            self._debug_gemini_response(response, f"Chunk {i}")
                        
                        if response and response.text:
                            translated_chunks.append(response.text.strip())
                            
                            if len(chunks) > 1:
                                print(f"Translated chunk {i}/{len(chunks)} (Gemini)", end='\r')
                        else:
                            error_details = self._analyze_failed_response(response, f"chunk {i}")
                            raise Exception(f"Empty response from Gemini for chunk {i}: {error_details}")
                            
                    except Exception as e:
                        error_msg = str(e)
                        print(f"\nWarning: Gemini translation failed for chunk {i}: {error_msg}")
                        
                        # Show first 100 chars of failed chunk for context
                        chunk_preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                        translated_chunks.append(f"[Translation failed for chunk {i}: {chunk_preview}]")
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
    
    def has_full_document_support(self) -> bool:
        """Check if full document translation is enabled."""
        return config.gemini_translation_mode == 'document'

    def _debug_gemini_response(self, response, context: str = "Translation"):
        """Debug and log detailed information about Gemini API response."""
        print(f"\n🔍 DEBUG - {context} Response Analysis:")
        print(f"   Response Type: {type(response)}")
        print(f"   Response Object: {response}")
        
        if response:
            # Check basic attributes
            print(f"   Has text attr: {hasattr(response, 'text')}")
            if hasattr(response, 'text'):
                text_value = response.text
                print(f"   Text value: {repr(text_value)}")
                print(f"   Text length: {len(text_value) if text_value else 0}")
            
            # Check candidates
            if hasattr(response, 'candidates'):
                candidates = response.candidates
                print(f"   Candidates count: {len(candidates) if candidates else 0}")
                if candidates:
                    for i, candidate in enumerate(candidates[:2]):  # Show first 2 candidates
                        print(f"   Candidate[{i}]:")
                        if hasattr(candidate, 'content'):
                            print(f"     Content: {candidate.content}")
                        if hasattr(candidate, 'finish_reason'):
                            print(f"     Finish Reason: {candidate.finish_reason}")
                        if hasattr(candidate, 'safety_ratings'):
                            print(f"     Safety Ratings: {candidate.safety_ratings}")
            
            # Check prompt feedback
            if hasattr(response, 'prompt_feedback'):
                feedback = response.prompt_feedback
                print(f"   Prompt Feedback: {feedback}")
                if feedback and hasattr(feedback, 'block_reason'):
                    print(f"     Block Reason: {feedback.block_reason}")
                if feedback and hasattr(feedback, 'safety_ratings'):
                    print(f"     Safety Ratings: {feedback.safety_ratings}")
            
            # Check usage metadata
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                print(f"   Usage Metadata: {usage}")
        
        print("   " + "="*50)

    def _analyze_failed_response(self, response, context: str = "translation") -> str:
        """Analyze a failed Gemini response and return detailed error information."""
        if not response:
            return "No response object returned from Gemini"
        
        error_parts = []
        
        # Check for blocked content
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            feedback = response.prompt_feedback
            if hasattr(feedback, 'block_reason') and feedback.block_reason:
                error_parts.append(f"Content blocked: {feedback.block_reason}")
                
                # Add safety rating details
                if hasattr(feedback, 'safety_ratings') and feedback.safety_ratings:
                    safety_info = []
                    for rating in feedback.safety_ratings:
                        if hasattr(rating, 'category') and hasattr(rating, 'probability'):
                            safety_info.append(f"{rating.category}: {rating.probability}")
                    if safety_info:
                        error_parts.append(f"Safety ratings: {', '.join(safety_info)}")
        
        # Check candidates
        if hasattr(response, 'candidates') and response.candidates:
            for i, candidate in enumerate(response.candidates):
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    if candidate.finish_reason != 'STOP':  # STOP is normal completion
                        error_parts.append(f"Candidate[{i}] finish reason: {candidate.finish_reason}")
                
                # Check safety ratings on candidates
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    safety_issues = []
                    for rating in candidate.safety_ratings:
                        if (hasattr(rating, 'probability') and 
                            rating.probability not in ['NEGLIGIBLE', 'LOW']):
                            safety_issues.append(f"{rating.category}: {rating.probability}")
                    if safety_issues:
                        error_parts.append(f"Candidate[{i}] safety issues: {', '.join(safety_issues)}")
        else:
            error_parts.append("No candidates in response")
        
        # Check if response has text but it's empty
        if hasattr(response, 'text'):
            if response.text is None:
                error_parts.append("Response text is None")
            elif response.text == "":
                error_parts.append("Response text is empty string")
        else:
            error_parts.append("Response has no text attribute")
        
        return " | ".join(error_parts) if error_parts else "Unknown error - response appears valid but empty"

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