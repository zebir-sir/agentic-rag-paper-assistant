import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .title_parser import extract_document_title

def _load_docling_dependencies():
    """Load Docling only for an actual extraction attempt.

    Docling imports Torch transitively. Keeping that import out of module load
    preserves the PDFium fallback on hosts where the local Torch DLL is absent.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    return DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions

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
    image_output_dir: str = "documents/.vision_assets"
    max_images: Optional[int] = None

class PDFExtractor:
    """使用 Docling 的 PDF 内容提取器，支持 GPU。"""
    
    def __init__(self, config: PDFExtractionConfig = None):
        self.config = config or PDFExtractionConfig()
        self.setup_converter()
    
    def setup_converter(self):
        """使用选项配置 Docling 文档转换器。"""
        self.converter = None
        self.docling_error: Optional[str] = None
        try:
            DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions = _load_docling_dependencies()
        except Exception as exc:  # pragma: no cover - depends on the host Torch runtime
            self.docling_error = str(exc)
            logger.warning("Docling is unavailable; PDFium fallback will be used: %s", self.docling_error)
            return
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.enable_ocr
        # 图片由方舟视觉适配器解释，避免 Docling 再启动一套未配置的视觉模型。
        pipeline_options.do_picture_description = False
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
            logger.warning("Docling initialization failed; PDFium fallback will be used: %s", e)
            self.docling_error = str(e)
            self.converter = None

    def extract_pdf_content(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """从单个 PDF 文件中提取内容。"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting content from: {pdf_path.name}")
        start_time = time.time()
                
        try:
            if self.converter is None:
                raise RuntimeError(self.docling_error or "Docling converter is unavailable")
            result = self.converter.convert(str(pdf_path))
            end_time = time.time()
            doc = result.document
            content_text = doc.export_to_markdown()
            metadata = {
                "source": str(pdf_path),
                "title": extract_document_title(content_text, str(pdf_path)),
                "title_source": "parsed_pdf_content",
                "processing_time": round(end_time - start_time, 2),
                "pages": len(doc.pages),
                "texts": len(doc.texts),
                "pictures": len(doc.pictures),
                "tables": len(doc.tables),
                "extraction_method": "docling",
                "content_type": "pdf",
            }
        except Exception as exc:
            logger.warning("Docling extraction failed for %s; using PDFium fallback: %s", pdf_path.name, exc)
            content_text, metadata = self._extract_with_pdfium_fallback(pdf_path, start_time)
            doc = None
        if self.config.include_images and doc is not None and doc.pictures:
            asset_dir = Path(self.config.image_output_dir)
            asset_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
            saved = []
            for index, picture in enumerate(doc.pictures[: self.config.max_images]):
                try:
                    image = picture.get_image(doc)
                    if image is None:
                        continue
                    image_path = asset_dir / f"{digest}-figure-{index + 1}.png"
                    image.save(image_path, format="PNG")
                    saved.append({
                        "index": index,
                        "path": str(image_path),
                        "caption": picture.caption_text(doc).strip(),
                    })
                except Exception as exc:
                    logger.warning("Failed to export figure %s from %s: %s", index + 1, pdf_path.name, exc)
            metadata["vision_assets"] = saved
            metadata["vision_asset_count"] = len(saved)
        return content_text, metadata

    def _extract_with_pdfium_fallback(self, pdf_path: Path, start_time: float) -> Tuple[str, Dict[str, Any]]:
        """Docling 模型不可用时，使用已安装的 PDFium 读取文本并渲染一页图像。

        这不是 Docling 的替代品，只保证单图视觉入库仍可完成，避免首次模型下载阻塞任务。
        """
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        page_texts = []
        target_page = 0
        caption = ""
        figure_pattern = re.compile(r"(?:fig(?:ure)?\.?\s*\d+[^\n]{0,240})", re.IGNORECASE)
        for index in range(len(document)):
            page = document.get_page(index)
            text = page.get_textpage().get_text_range()
            page_texts.append(text)
            match = figure_pattern.search(text)
            if match and not caption:
                target_page = index
                caption = re.sub(r"\s+", " ", match.group(0)).strip()

        metadata = {
            "source": str(pdf_path),
            "title": extract_document_title("\n".join(page_texts), str(pdf_path)),
            "title_source": "parsed_pdf_content",
            "processing_time": round(time.time() - start_time, 2),
            "pages": len(document),
            "texts": sum(bool(value.strip()) for value in page_texts),
            "pictures": 0,
            "tables": 0,
            "extraction_method": "pdfium_fallback",
            "content_type": "pdf",
        }
        if self.config.include_images:
            asset_dir = Path(self.config.image_output_dir)
            asset_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
            asset_path = asset_dir / f"{digest}-figure-page-{target_page + 1}.png"
            page = document.get_page(target_page)
            page.render(scale=2).to_pil().save(asset_path, format="PNG")
            metadata["vision_assets"] = [{
                "index": 0,
                "path": str(asset_path),
                "caption": caption or f"PDF page {target_page + 1} visual evidence",
                "page": target_page + 1,
                "asset_kind": "rendered_page_fallback",
            }]
            metadata["vision_asset_count"] = 1
        return "\n\n".join(page_texts), metadata
    

def create_pdf_extractor(config: PDFExtractionConfig = None) -> PDFExtractor:
    """创建 PDF 提取器实例。"""
    return PDFExtractor(config)
