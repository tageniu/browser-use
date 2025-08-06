"""PDF processor for extracting content from PDF documents."""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional
import httpx
import logging

from .exceptions import PDFProcessingError, PDFDownloadError, PDFExtractionError, PDFNotFoundError

logger = logging.getLogger(__name__)


class PDFProcessor:
    """PDF processor that can extract text content from PDF documents."""
    
    def __init__(self, max_file_size_mb: int = 50):
        """Initialize PDF processor.
        
        Args:
            max_file_size_mb: Maximum PDF file size to process in MB
        """
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
    async def extract_text_from_url(self, url: str, query: Optional[str] = None) -> str:
        """Extract text content from a PDF URL.
        
        Args:
            url: URL of the PDF document
            query: Optional query to focus extraction on specific content
            
        Returns:
            Extracted text content from the PDF
            
        Raises:
            PDFProcessingError: If PDF processing fails
        """
        try:
            # Download PDF to temporary file
            temp_file = await self._download_pdf(url)
            
            try:
                # Extract text from the PDF
                text_content = await self._extract_text_from_file(temp_file, query)
                return text_content
            finally:
                # Clean up temporary file
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
                    
        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Failed to process PDF from URL {url}: {str(e)}") from e
    
    async def _download_pdf(self, url: str) -> Path:
        """Download PDF from URL to temporary file.
        
        Args:
            url: URL of the PDF document
            
        Returns:
            Path to the downloaded temporary file
            
        Raises:
            PDFDownloadError: If download fails
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # First, make a HEAD request to check content type and size
                try:
                    head_response = await client.head(url, follow_redirects=True)
                    content_type = head_response.headers.get('content-type', '').lower()
                    content_length = head_response.headers.get('content-length')
                    
                    # Check if it's actually a PDF
                    if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                        logger.warning(f"URL {url} may not be a PDF (content-type: {content_type})")
                    
                    # Check file size
                    if content_length:
                        size_bytes = int(content_length)
                        if size_bytes > self.max_file_size_bytes:
                            raise PDFDownloadError(
                                f"PDF file too large: {size_bytes / 1024 / 1024:.1f}MB "
                                f"(max: {self.max_file_size_mb}MB)"
                            )
                except httpx.HTTPError:
                    # HEAD request failed, continue with GET
                    logger.debug(f"HEAD request failed for {url}, proceeding with GET")
                
                # Download the PDF
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                # Check actual content type from response
                content_type = response.headers.get('content-type', '').lower()
                if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                    logger.warning(f"Downloaded content may not be a PDF (content-type: {content_type})")
                
                # Check downloaded size
                if len(response.content) > self.max_file_size_bytes:
                    raise PDFDownloadError(
                        f"Downloaded PDF too large: {len(response.content) / 1024 / 1024:.1f}MB "
                        f"(max: {self.max_file_size_mb}MB)"
                    )
                
                # Save to temporary file
                temp_file = Path(tempfile.mktemp(suffix='.pdf'))
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded PDF from {url} to {temp_file} ({len(response.content)} bytes)")
                return temp_file
                
        except httpx.HTTPError as e:
            raise PDFDownloadError(f"Failed to download PDF from {url}: {str(e)}") from e
        except Exception as e:
            raise PDFDownloadError(f"Unexpected error downloading PDF from {url}: {str(e)}") from e
    
    async def _extract_text_from_file(self, pdf_path: Path, query: Optional[str] = None) -> str:
        """Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            query: Optional query to focus extraction on specific content
            
        Returns:
            Extracted text content
            
        Raises:
            PDFExtractionError: If text extraction fails
        """
        if not pdf_path.exists():
            raise PDFNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Try PyPDF2 first (fastest, but may not work for all PDFs)
            text_content = await self._extract_with_pypdf2(pdf_path)
            
            if text_content.strip():
                logger.info(f"Successfully extracted {len(text_content)} characters with PyPDF2")
                return self._format_extracted_text(text_content, query)
            
            # If PyPDF2 fails or returns empty content, try pdfplumber
            text_content = await self._extract_with_pdfplumber(pdf_path)
            
            if text_content.strip():
                logger.info(f"Successfully extracted {len(text_content)} characters with pdfplumber")
                return self._format_extracted_text(text_content, query)
            
            # If both fail, return a message indicating the PDF might be image-based
            return f"Could not extract text from PDF. The PDF might contain images or scanned content that requires OCR processing."
            
        except Exception as e:
            raise PDFExtractionError(f"Failed to extract text from PDF {pdf_path}: {str(e)}") from e
    
    async def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2."""
        def _sync_extract():
            try:
                from PyPDF2 import PdfReader
                
                with open(pdf_path, 'rb') as file:
                    reader = PdfReader(file)
                    text_parts = []
                    
                    for page_num, page in enumerate(reader.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text.strip():
                                text_parts.append(f"=== Page {page_num} ===\n{page_text.strip()}")
                        except Exception as e:
                            logger.warning(f"Failed to extract text from page {page_num}: {e}")
                            text_parts.append(f"=== Page {page_num} ===\n[Text extraction failed]")
                    
                    return "\n\n".join(text_parts)
                    
            except ImportError:
                logger.warning("PyPDF2 not available, skipping PyPDF2 extraction")
                return ""
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
                return ""
        
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_extract)
    
    async def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        def _sync_extract():
            try:
                import pdfplumber
                
                text_parts = []
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text_parts.append(f"=== Page {page_num} ===\n{page_text.strip()}")
                            else:
                                # Try to extract table data if no text found
                                tables = page.extract_tables()
                                if tables:
                                    table_text = self._format_tables(tables)
                                    if table_text:
                                        text_parts.append(f"=== Page {page_num} (Tables) ===\n{table_text}")
                                    else:
                                        text_parts.append(f"=== Page {page_num} ===\n[No extractable text found]")
                                else:
                                    text_parts.append(f"=== Page {page_num} ===\n[No extractable text found]")
                        except Exception as e:
                            logger.warning(f"Failed to extract text from page {page_num}: {e}")
                            text_parts.append(f"=== Page {page_num} ===\n[Text extraction failed]")
                
                return "\n\n".join(text_parts)
                
            except ImportError:
                logger.warning("pdfplumber not available, skipping pdfplumber extraction")
                return ""
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
                return ""
        
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_extract)
    
    def _format_tables(self, tables) -> str:
        """Format table data as text."""
        try:
            formatted_tables = []
            for i, table in enumerate(tables, 1):
                if table:
                    table_text = f"Table {i}:\n"
                    for row in table:
                        if row:
                            # Filter out None values and join with tabs
                            clean_row = [str(cell) if cell is not None else "" for cell in row]
                            table_text += "\t".join(clean_row) + "\n"
                    formatted_tables.append(table_text)
            return "\n".join(formatted_tables)
        except Exception as e:
            logger.warning(f"Failed to format tables: {e}")
            return ""
    
    def _format_extracted_text(self, text: str, query: Optional[str] = None) -> str:
        """Format and optionally filter extracted text based on query.
        
        Args:
            text: Raw extracted text
            query: Optional query to highlight relevant sections
            
        Returns:
            Formatted text content
        """
        if not text.strip():
            return "No text content could be extracted from this PDF."
        
        # Basic cleanup
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        formatted_text = '\n'.join(cleaned_lines)
        
        # If query is provided, add a note about the query context
        if query:
            formatted_text = f"Query: {query}\n\n{formatted_text}"
        
        # Limit text length to avoid context overflow (approximately 20,000 tokens)
        max_chars = 60000
        if len(formatted_text) > max_chars:
            formatted_text = (
                formatted_text[:max_chars // 2] + 
                "\n\n... [Content truncated due to length - showing first and last portions] ...\n\n" +
                formatted_text[-max_chars // 2:]
            )
        
        return formatted_text
