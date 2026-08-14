"""Stable raster fallback for PDF pages whose embedded fonts are not browser-safe."""

from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from threading import Lock

import pypdfium2 as pdfium


_PAGE_IMAGE_CACHE_MAX_ITEMS = 16
_page_image_cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
_cache_lock = Lock()


def render_pdf_page_png(pdf_bytes: bytes, page_number: int, scale: float = 2.0) -> bytes:
    """Render one 1-based PDF page to PNG using PDFium's native font/image engine."""
    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    document = pdfium.PdfDocument(pdf_bytes)
    page = None
    bitmap = None
    image = None
    try:
        if page_number > len(document):
            raise IndexError(f"PDF page {page_number} is outside document range")
        page = document.get_page(page_number - 1)
        bitmap = page.render(scale=scale, prefer_bgrx=True)
        image = bitmap.to_pil().convert("RGB")
        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        return output.getvalue()
    finally:
        if image is not None:
            image.close()
        if bitmap is not None:
            bitmap.close()
        if page is not None:
            page.close()
        document.close()


def render_cached_pdf_page_png(pdf_bytes: bytes, document_id: str, page_number: int) -> bytes:
    """Render a page once per document and retain a small LRU cache of PNG bytes."""
    cache_key = (str(document_id), int(page_number))
    with _cache_lock:
        cached = _page_image_cache.get(cache_key)
        if cached is not None:
            _page_image_cache.move_to_end(cache_key)
            return cached

    image = render_pdf_page_png(pdf_bytes, page_number)
    with _cache_lock:
        _page_image_cache[cache_key] = image
        _page_image_cache.move_to_end(cache_key)
        while len(_page_image_cache) > _PAGE_IMAGE_CACHE_MAX_ITEMS:
            _page_image_cache.popitem(last=False)
    return image
