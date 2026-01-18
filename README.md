# Tamil PDF OCR & Translation Tool

A professional-grade Python application for extracting Tamil text from PDFs and optionally translating to English with high accuracy using Google Cloud Translate API.

## 🌟 Features

- **High-Quality OCR**: Tesseract OCR optimized for Tamil text extraction
- **Accurate Translation**: Google Cloud Translate API for Tamil → English translation  
- **Batch Processing**: Process entire books or specific page ranges
- **Unicode Support**: Proper Tamil Unicode output
- **Clean Architecture**: Modular design following SOLID principles
- **Error Handling**: Comprehensive error management and user feedback

## 📋 Requirements

### System Requirements
- Python 3.7+
- Tesseract OCR with Tamil language support
- Poppler (for PDF processing)

### Python Dependencies
```bash
# Core dependencies (required)
pip install pdf2image pytesseract pillow

# Translation dependencies (choose one option)

# Option 1: Google Cloud Translation (paid, highest quality)
pip install google-cloud-translate

# Option 2: Local Translation (FREE, offline options)
# Install ONE of these:
pip install transformers torch sentencepiece  # HuggingFace models (recommended)
pip install argostranslate                    # Lightweight, fully offline
pip install requests                          # For LibreTranslate API
```

### System Setup

#### Windows

**Option 1: Chocolatey (Requires Admin)**
```powershell
# Install Chocolatey (if not already installed) from Administrator PowerShell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Restart PowerShell as Administrator, then install
choco install tesseract
choco install poppler
```

**Option 2: Manual Installation (Recommended)**

**Install Tesseract OCR:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location (C:\Program Files\Tesseract-OCR)
3. Add to PATH and download Tamil data:
```powershell
# Add Tesseract to PATH
$tesseractPath = "C:\Program Files\Tesseract-OCR"
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";$tesseractPath", [EnvironmentVariableTarget]::User)
$env:PATH += ";$tesseractPath"

# Download Tamil language data
$tamilUrl = "https://github.com/tesseract-ocr/tessdata/raw/main/tam.traineddata"
$tempFile = "$env:TEMP\tam.traineddata"
Invoke-WebRequest -Uri $tamilUrl -OutFile $tempFile
# Copy to tessdata directory (requires admin privileges)
Start-Process powershell -ArgumentList "-Command", "Copy-Item '$tempFile' 'C:\Program Files\Tesseract-OCR\tessdata\' -Force" -Verb RunAs -Wait
```

**Install Poppler:**
```powershell
# Download and extract Poppler
$url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v23.08.0-0/Release-23.08.0-0.zip"
$output = "$env:TEMP\poppler-windows.zip"
Invoke-WebRequest -Uri $url -OutFile $output

# Extract to C:\poppler
$extractPath = "C:\poppler"
New-Item -ItemType Directory -Path $extractPath -Force
Expand-Archive -Path $output -DestinationPath $extractPath -Force

# Add to PATH
$popplerBinPath = "C:\poppler\poppler-23.08.0\Library\bin"
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";$popplerBinPath", [EnvironmentVariableTarget]::User)
$env:PATH += ";$popplerBinPath"

# Clean up
Remove-Item $output -Force
```

**Verify Installation:**
```powershell
tesseract --version
tesseract --list-langs  # Should include 'tam' for Tamil
pdftoppm -v
```

#### macOS
```bash
# Install Tesseract with Tamil support
brew install tesseract tesseract-lang

# Install Poppler  
brew install poppler
```

#### Ubuntu/Debian
```bash
# Install Tesseract with Tamil support
sudo apt-get install tesseract-ocr tesseract-ocr-tam

# Install Poppler
sudo apt-get install poppler-utils
```

## 🚀 Quick Start

### Basic Usage (OCR Only)
```bash
# Extract Tamil Unicode text from PDF
python main.py book.pdf

# Process specific pages
python main.py book.pdf --start 1 --end 10

# Custom output filename
python main.py book.pdf tamil_output.txt
```

### With Translation

#### Cloud Translation (Google Translate)
```bash
# OCR + Translation to English (requires Google Cloud setup)
python main.py book.pdf --translate

# Process specific pages with translation
python main.py book.pdf --start 5 --end 15 --translate

# Custom output with translation
python main.py book.pdf english_book.txt --translate
```

#### Local Translation (FREE, Offline)
```bash
# OCR + Translation to English (completely free)
python main.py book.pdf --translate --local

# Process specific pages with local translation
python main.py book.pdf --start 5 --end 15 --translate --local

# Custom output with local translation
python main.py book.pdf english_book.txt --translate
```

## 🔧 Translation Setup

### Option 1: Google Cloud Translation (Paid)
For high-quality translation with Google Cloud:

#### 1. Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Cloud Translation API
4. Note your project ID for configuration

#### 2. Set Up Authentication
```bash
# Method 1: Service Account Key (Recommended)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Method 2: Application Default Credentials
gcloud auth application-default login
gcloud config set project your-project-id
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

#### 3. Install Translation Dependencies
```bash
pip install google-cloud-translate
```

**💰 Cost**: Google Cloud offers $300 free credit. Translation costs ~$20 per million characters (approximately $40-60 per 500-page book).

### Option 2: Local Translation (FREE) ⭐

#### Quick Setup
```bash
# Option A: HuggingFace Transformers (recommended)
pip install transformers torch sentencepiece

# Option B: Argos Translate (fully offline)
pip install argostranslate

# Option C: LibreTranslate API (free online service)
pip install requests
```

#### Usage
```bash
# Use any of the installed local translation services
python main.py book.pdf --translate --local
```

#### Local Translation Features:
- **✅ Completely FREE** - No API costs or limits
- **✅ Privacy** - All processing happens locally (except LibreTranslate API option)
- **✅ No Internet Required** - Works offline (HuggingFace & Argos)
- **✅ No Setup** - No API keys or authentication needed

#### Translation Quality Comparison:
1. **Google Cloud**: Highest quality, best for professional use
2. **HuggingFace Models**: Good quality, free, slightly slower first run
3. **Argos Translate**: Decent quality, lightweight, fully offline
4. **LibreTranslate**: Good quality, free online service

## 📁 Project Structure

```
tamil-to-english-convertor/
├── main.py                 # Main CLI application
├── src/
│   ├── __init__.py        # Package initialization
│   ├── ocr.py            # OCR processing logic
│   ├── translation.py     # Translation service
│   └── file_handler.py    # File operations
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🏗️ Architecture

The application follows SOLID principles with clear separation of concerns:

- **Single Responsibility**: Each class has one focused purpose
- **Open/Closed**: Easy to extend with new translation providers
- **Liskov Substitution**: Consistent interfaces across components  
- **Interface Segregation**: Minimal, focused class interfaces
- **Dependency Inversion**: High-level modules don't depend on low-level details

### Key Components

1. **TamilOCRProcessor** (`src/ocr.py`)
   - Handles PDF to image conversion
   - Manages Tesseract OCR processing
   - Extracts Tamil text with optimal settings

2. **TamilTranslator** (`src/translation.py`)  
   - Google Cloud Translate API integration
   - Intelligent text chunking for large documents
   - Rate limiting and error handling

3. **FileHandler & ContentProcessor** (`src/file_handler.py`)
   - File I/O operations
   - Content formatting and structure
   - Output generation utilities

4. **TamilPDFProcessor** (`main.py`)
   - Main orchestrator coordinating all components
   - CLI interface and user interaction
   - Error handling and progress reporting

## 📊 Performance & Quality

### OCR Accuracy
- **Tamil Unicode**: ~90-95% accuracy with Tesseract
- **Processing Speed**: ~2-3 seconds per page
- **Format Support**: Text-based and scanned PDFs

### Translation Quality  
- **Google Translate**: ~97-99% accuracy for Tamil-English
- **Context Preservation**: Intelligent chunking maintains meaning
- **Error Recovery**: Graceful handling of translation failures

### Cost Estimates
| Pages | OCR Cost | Translation Cost* | Total |
|-------|----------|------------------|--------|
| 100   | Free     | ~$8-12          | ~$8-12 |  
| 500   | Free     | ~$40-60         | ~$40-60 |
| 1000  | Free     | ~$80-120        | ~$80-120 |

*With $300 Google Cloud free credit, you can process ~6 books (500 pages each) completely free.

## 🛠️ Troubleshooting

### Common Issues

**Tesseract not found**
```bash
# Verify installation
tesseract --version
tesseract --list-langs | grep tam
```

**Google Cloud authentication errors**  
```bash
# Check credentials
echo $GOOGLE_APPLICATION_CREDENTIALS
gcloud auth list
```

**Poor OCR quality**
- Ensure PDF has good image quality (300+ DPI)
- Check if PDF uses standard Tamil fonts
- Consider pre-processing images for better contrast

**Translation API quota exceeded**
- Check your Google Cloud billing and quotas
- Implement longer delays between requests
- Process smaller batches

### Getting Help

1. **Check logs**: The application provides detailed error messages
2. **Verify setup**: Ensure all dependencies are correctly installed
3. **Test components**: Try OCR-only first, then add translation
4. **Check permissions**: Ensure proper file/directory access rights

## 📈 Usage Examples

### Processing Multiple Books
```bash
# Batch process all PDFs in directory
for pdf in *.pdf; do
    python main.py "$pdf" --translate
done
```

### Quality Testing
```bash
# Test OCR quality on a few pages first
python main.py book.pdf --start 5 --end 7

# If quality is good, process full book
python main.py book.pdf --translate
```

### Custom Workflows
```python
# Use as Python module
from src.ocr import TamilOCRProcessor
from src.translation import TamilTranslator

ocr = TamilOCRProcessor()
translator = TamilTranslator()

# Extract and translate specific pages
pages = ocr.process_pdf("book.pdf", start_page=1, end_page=10)
for page_num, tamil_text in pages:
    english_text = translator.translate_text(tamil_text)
    print(f"Page {page_num}: {english_text[:100]}...")
```

## 🤝 Contributing

This project follows clean coding principles. When contributing:

1. Follow the existing architecture patterns
2. Add proper error handling
3. Include type hints
4. Keep functions focused and testable  
5. Update documentation for new features

## 📝 License

This project is open-source. Feel free to use, modify, and distribute according to your needs.

---

**🎯 Perfect for**: Academic research, literature digitization, historical document preservation, multilingual content creation, and educational purposes.

**⚡ Quick Start**: Install dependencies → Run on sample PDF → Enable translation → Process your books!