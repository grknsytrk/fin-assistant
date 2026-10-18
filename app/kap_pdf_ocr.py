"""CPU OCR for KAP PDFs whose pages have no usable text layer."""

from __future__ import annotations

import logging
import re
import statistics
import threading
from contextlib import closing
from typing import Any

LOGGER = logging.getLogger(__name__)
_OCR_LOCK = threading.Lock()
_OCR_ENGINE: Any = None
_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")


def _join_cells(items: list[dict[str, Any]]) -> str:
    """Keep wrapped issuer names together before joining table columns."""
    cells: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda item: item["left"]):
        if cells and item["left"] < max(part["right"] for part in cells[-1]):
            cells[-1].append(item)
        else:
            cells.append([item])
    return "\t".join(
        " ".join(part["text"] for part in sorted(cell, key=lambda part: part["y"]))
        for cell in cells
    )


def ocr_result_text(result: list[Any]) -> str:
    """Rebuild rows by geometry, including two-line cells next to an ISIN."""
    items = []
    for box, text, score in result:
        if not str(text).strip() or float(score) < 0.5:
            continue
        top, bottom = min(point[1] for point in box), max(point[1] for point in box)
        items.append({
            "left": min(point[0] for point in box),
            "right": max(point[0] for point in box),
            "y": (top + bottom) / 2,
            "height": max(1, bottom - top),
            "text": str(text).strip(),
        })
    if not items:
        return ""
    height = statistics.median(item["height"] for item in items)
    anchors = sorted(
        (item for item in items if _ISIN.fullmatch(item["text"].upper())),
        key=lambda item: item["y"],
    )
    used: set[int] = set()
    lines = []
    for index, anchor in enumerate(anchors):
        radius = height * 1.1
        above = min(radius, (anchor["y"] - anchors[index - 1]["y"]) / 2) if index else radius
        below = min(radius, (anchors[index + 1]["y"] - anchor["y"]) / 2) if index + 1 < len(anchors) else radius
        row = [
            item for item in items
            if id(item) not in used and anchor["y"] - above <= item["y"] < anchor["y"] + below
        ]
        used.update(id(item) for item in row)
        lines.append((anchor["y"], _join_cells(row)))

    remaining = sorted((item for item in items if id(item) not in used), key=lambda item: item["y"])
    row: list[dict[str, Any]] = []
    for item in remaining:
        if row and item["y"] - row[0]["y"] > height * 0.6:
            lines.append((row[0]["y"], _join_cells(row)))
            row = []
        row.append(item)
    if row:
        lines.append((row[0]["y"], _join_cells(row)))
    return "\n".join(text for _y, text in sorted(lines))


def extract_pdf_ocr_pages(pdf_data: bytes, page_indices: list[int]) -> dict[int, str]:
    """Render only image pages; reuse a bounded, lazy CPU inference engine."""
    import pypdfium2 as pdfium
    from rapidocr_onnxruntime import RapidOCR

    global _OCR_ENGINE
    texts = {}
    # PDFium and the OCR preprocessor have mutable native state. Keep their
    # work serialized across simultaneous fund refreshes in this process.
    with _OCR_LOCK:
        if _OCR_ENGINE is None:
            _OCR_ENGINE = RapidOCR(
                det_limit_side_len=2200,
                max_side_len=2200,
                intra_op_num_threads=2,
                inter_op_num_threads=2,
            )
        with closing(pdfium.PdfDocument(pdf_data)) as document:
            for index in page_indices:
                try:
                    with closing(document[index]) as page:
                        scale = min(3.0, 2200 / max(page.get_size()))
                        with closing(page.render(scale=scale)) as bitmap:
                            result, _elapsed = _OCR_ENGINE(bitmap.to_numpy().copy())
                    texts[index] = ocr_result_text(result or [])
                except Exception:
                    LOGGER.warning("KAP PDF OCR failed on page %s", index + 1, exc_info=True)
    return texts
