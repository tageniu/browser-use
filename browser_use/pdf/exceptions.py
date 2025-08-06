"""PDF processing exceptions."""


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""
    pass


class PDFDownloadError(PDFProcessingError):
    """Error downloading PDF file."""
    pass


class PDFExtractionError(PDFProcessingError):
    """Error extracting content from PDF."""
    pass


class PDFNotFoundError(PDFProcessingError):
    """PDF file not found."""
    pass
