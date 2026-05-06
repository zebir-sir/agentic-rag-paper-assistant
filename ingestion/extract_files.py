import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Docling 相关导入
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # 禁用符号链接警告

@dataclass
class PDFExtractionConfig:
    """支持 GPU 的 PDF 提取配置。"""
    enable_ocr: bool = True
    images_scale: float = 2.0
    include_images: bool = True
    include_tables: bool = True 

class PDFExtractor:
    """使用 Docling 的 PDF 内容提取器，支持 GPU。"""
    
    def __init__(self, config: PDFExtractionConfig = None):
        self.config = config or PDFExtractionConfig()
        self.setup_converter()
    
    def setup_converter(self):
        """使用选项配置 Docling 文档转换器。"""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.enable_ocr
        pipeline_options.do_picture_description = self.config.include_images
        pipeline_options.do_table_structure = self.config.include_tables
        pipeline_options.images_scale = self.config.images_scale
        try:
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        except Exception as e:
            logging.error(f"Failed to initialize docling: {e}")

    def extract_pdf_content(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """从单个 PDF 文件中提取内容。"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting content from: {pdf_path.name}")
        start_time = time.time()
                
        result = self.converter.convert(str(pdf_path))
        end_time = time.time()
        
        doc = result.document
        content_text = doc.export_to_markdown()
        
        metadata = {
            "source": str(pdf_path),
            "title": pdf_path.stem,
            "processing_time": round(end_time - start_time, 2),
            "pages": len(doc.pages),
            "texts": len(doc.texts),
            "pictures": len(doc.pictures),
            "tables": len(doc.tables),
            "extraction_method": "docling",
            "content_type": "pdf"
        }        
        return content_text, metadata
    

def create_pdf_extractor(config: PDFExtractionConfig = None) -> PDFExtractor:
    """创建 PDF 提取器实例。"""
    return PDFExtractor(config)
