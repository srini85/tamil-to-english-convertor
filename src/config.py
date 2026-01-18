"""Configuration management for Tamil PDF OCR Translation."""

import os
import time
from dotenv import load_dotenv
from typing import Optional


class Config:
    """Centralized configuration management with .env file support and rate limiting."""
    
    def __init__(self, env_file: str = ".env"):
        """Initialize configuration by loading from .env file and environment variables."""
        # Load .env file if it exists
        if os.path.exists(env_file):
            load_dotenv(env_file)
            print(f"✅ Loaded configuration from {env_file}")
        else:
            print(f"⚠️  No {env_file} file found, using environment variables only")
            if os.path.exists(".env.template"):
                print(f"💡 Copy .env.template to {env_file} and customize your settings")
    
    # =============================================================================
    # API KEYS & AUTHENTICATION
    # =============================================================================
    
    @property
    def gemini_api_key(self) -> Optional[str]:
        return os.getenv('GEMINI_API_KEY')
    
    @property
    def google_cloud_project(self) -> Optional[str]:
        return os.getenv('GOOGLE_CLOUD_PROJECT')
    
    @property
    def google_application_credentials(self) -> Optional[str]:
        return os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    # =============================================================================
    # MODEL CONFIGURATION
    # =============================================================================
    
    @property
    def gemini_model(self) -> str:
        return os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    
    # =============================================================================
    # RATE LIMITING & PERFORMANCE
    # =============================================================================
    
    @property
    def gemini_delay_between_requests(self) -> float:
        return float(os.getenv('GEMINI_DELAY_BETWEEN_REQUESTS', '2.0'))
    
    @property
    def gemini_delay_between_chunks(self) -> float:
        return float(os.getenv('GEMINI_DELAY_BETWEEN_CHUNKS', '1.0'))
    
    @property
    def gemini_max_requests_per_minute(self) -> int:
        return int(os.getenv('GEMINI_MAX_REQUESTS_PER_MINUTE', '15'))
    
    @property
    def google_translate_delay_between_requests(self) -> float:
        return float(os.getenv('GOOGLE_TRANSLATE_DELAY_BETWEEN_REQUESTS', '0.5'))
    
    @property
    def google_translate_delay_between_chunks(self) -> float:
        return float(os.getenv('GOOGLE_TRANSLATE_DELAY_BETWEEN_CHUNKS', '0.2'))
    
    @property
    def google_translate_max_requests_per_minute(self) -> int:
        return int(os.getenv('GOOGLE_TRANSLATE_MAX_REQUESTS_PER_MINUTE', '100'))
    
    @property
    def local_translate_delay_between_requests(self) -> float:
        return float(os.getenv('LOCAL_TRANSLATE_DELAY_BETWEEN_REQUESTS', '0.5'))
    
    @property
    def local_translate_delay_between_chunks(self) -> float:
        return float(os.getenv('LOCAL_TRANSLATE_DELAY_BETWEEN_CHUNKS', '0.2'))
    
    @property
    def local_translate_max_requests_per_minute(self) -> int:
        return int(os.getenv('LOCAL_TRANSLATE_MAX_REQUESTS_PER_MINUTE', '120'))
    
    # =============================================================================
    # OCR SETTINGS
    # =============================================================================
    
    @property
    def ocr_dpi(self) -> int:
        return int(os.getenv('OCR_DPI', '300'))
    
    @property
    def tesseract_config(self) -> str:
        return os.getenv('TESSERACT_CONFIG', '--oem 1 --psm 6 -c preserve_interword_spaces=1')
    
    @property
    def enhanced_ocr_enabled(self) -> bool:
        return os.getenv('ENHANCED_OCR_ENABLED', 'true').lower() == 'true'
    
    @property
    def ocr_mode(self) -> str:
        return os.getenv('OCR_MODE', 'balanced').lower()
    
    # =============================================================================
    # TRANSLATION SETTINGS
    # =============================================================================
    
    @property
    def max_chunk_size(self) -> int:
        return int(os.getenv('MAX_CHUNK_SIZE', '6000'))
    
    @property
    def translation_temperature(self) -> float:
        return float(os.getenv('TRANSLATION_TEMPERATURE', '0.1'))
    
    @property
    def translation_max_output_tokens(self) -> int:
        return int(os.getenv('TRANSLATION_MAX_OUTPUT_TOKENS', '4096'))
    
    # =============================================================================
    # LOGGING & DEBUG
    # =============================================================================
    
    @property
    def verbose_logging(self) -> bool:
        return os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'
    
    @property
    def debug_mode(self) -> bool:
        return os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    @property
    def log_file_path(self) -> Optional[str]:
        path = os.getenv('LOG_FILE_PATH', '').strip()
        return path if path else None
    
    # =============================================================================
    # FILE PROCESSING
    # =============================================================================
    
    @property
    def output_directory(self) -> str:
        return os.getenv('OUTPUT_DIRECTORY', './output')
    
    @property
    def auto_open_output(self) -> bool:
        return os.getenv('AUTO_OPEN_OUTPUT', 'false').lower() == 'true'


class RateLimiter:
    """Rate limiting utility to manage API request frequency."""
    
    def __init__(self, config: Config):
        self.config = config
        self._request_times = {}
    
    def wait_if_needed(self, service: str, request_type: str = 'request'):
        """Wait if necessary to respect rate limits for the specified service."""
        if service == 'gemini':
            if request_type == 'chunk':
                delay = self.config.gemini_delay_between_chunks
            else:
                delay = self.config.gemini_delay_between_requests
        elif service == 'google_translate':
            if request_type == 'chunk':
                delay = self.config.google_translate_delay_between_chunks
            else:
                delay = self.config.google_translate_delay_between_requests
        elif service == 'local_translate':
            if request_type == 'chunk':
                delay = self.config.local_translate_delay_between_chunks
            else:
                delay = self.config.local_translate_delay_between_requests
        else:
            delay = 0.5  # Default delay
        
        if delay > 0:
            if self.config.verbose_logging:
                print(f"⏳ Rate limiting: waiting {delay}s for {service} {request_type}")
            time.sleep(delay)
    
    def log_request(self, service: str):
        """Log the time of a request for rate limiting tracking."""
        current_time = time.time()
        if service not in self._request_times:
            self._request_times[service] = []
        
        # Keep only requests from the last minute
        minute_ago = current_time - 60
        self._request_times[service] = [
            req_time for req_time in self._request_times[service] 
            if req_time > minute_ago
        ]
        
        self._request_times[service].append(current_time)
        
        if self.config.verbose_logging:
            print(f"📊 {service} requests in last minute: {len(self._request_times[service])}")


# Global configuration instance
config = Config()
rate_limiter = RateLimiter(config)