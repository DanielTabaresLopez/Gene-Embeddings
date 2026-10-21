#!/usr/bin/env python3
"""Stack the supplied CKA and kNN PDFs without scaling or redrawing them.

Only title/subtitle removal, replacement colourbar wording, panel lettering,
and translation onto a taller page are performed. Input PDFs are untouched.
Requires PyMuPDF. Run after cs_12_plot_component_heatmaps.py:
    python scripts/consensus_graph/cs_13_assemble_heatmap_panel.py --root /path/to/analysis
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math

import fitz


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("01_calibrated_cka_reference_order.pdf", "a", "Calibrated linear CKA"),
    ("02_calibrated_knn_k100_reference_order.pdf", "b", "Calibrated neighborhood overlap"),
)
OLD_COLORBAR = "Calibrated similarity (square-root colour scale)"
TOP_BLANK_CROP_PT = 52.0
PANEL_GAP_PT = 24.0
REFERENCE_WIDTH_PX = 1448.0
REFERENCE_LETTER_EM_PX = 22.0


def spans(page):
    return [span for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", []) for span in line["spans"]]


def span_key(span):
    return (span["text"], span["font"], round(span["size"], 4))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fonts(doc):
    result = {}
    for entry in doc[0].get_fonts():
        _, _, _, buffer = doc.extract_font(entry[0])
        result["bold" if "Bold" in entry[3] else "regular"] = buffer
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    figures = args.root.expanduser().resolve() / "results/consensus_graph_v2/10_figures"
    inputs = args.input_dir or figures / "component_heatmaps_reference_order"
    destination = args.output_dir or figures / "component_heatmaps_panel"
    destination.mkdir(parents=True, exist_ok=True)
    originals = [inputs / row[0] for row in SOURCES]
    input_hashes = {p.name: sha(p) for p in originals}
    working = []
    audits = []
    expected_preserved_spans = []
    original_image_digests = []
    for path, (_, letter, title) in zip(originals, SOURCES):
        doc = fitz.open(path)
        if len(doc) != 1:
            raise ValueError("Each input must contain one page")
        page = doc[0]
        original_spans = spans(page)
        fonts = source_fonts(doc)
        cbar_spans = [s for s in original_spans if s["text"] == OLD_COLORBAR]
        if len(cbar_spans) != 1:
            raise ValueError("Expected exactly one original colourbar title")
        cbar = cbar_spans[0]
        # The original title and subtitle are the only text above y=52 pt.
        headings = [s for s in original_spans if s["bbox"][3] < TOP_BLANK_CROP_PT]
        if len(headings) != 2:
            raise ValueError("Unexpected header layout; refusing broad redaction")
        removed = headings + [cbar]
        preserved = [s for s in original_spans if s not in removed]
        original_images = page.get_image_info(hashes=True)
        for s in removed:
            r = fitz.Rect(s["bbox"])
            r.x0 -= .15; r.x1 += .15; r.y0 -= .15; r.y1 += .15
            page.add_redact_annot(r, fill=False)
        page.apply_redactions(images=0, graphics=0, text=0)
        if Counter(map(span_key, spans(page))) != Counter(map(span_key, preserved)):
            raise AssertionError("Redaction changed text other than the three selected headings")
        after_images = page.get_image_info(hashes=True)
        if [(i["digest"], i["bbox"]) for i in original_images] != [
                (i["digest"], i["bbox"]) for i in after_images]:
            raise AssertionError("Redaction changed the heatmap or colourbar artwork")
        original_image_digests.extend(i["digest"].hex() for i in original_images)

        # Use the original embedded font and the original 11 pt font size.
        font = fitz.Font(fontbuffer=fonts["regular"])
        page.insert_font(fontname="PanelRegular", fontbuffer=fonts["regular"])
        length = font.text_length(title, fontsize=cbar["size"])
        center_y = (cbar["bbox"][1] + cbar["bbox"][3]) / 2
        page.insert_text((cbar["origin"][0], center_y + length / 2), title,
                         fontsize=cbar["size"], fontname="PanelRegular",
                         color=(0, 0, 0), rotate=90)
        clip = fitz.Rect(0, TOP_BLANK_CROP_PT, page.rect.width, page.rect.height)
        expected_preserved_spans.append(preserved)
        working.append((doc, clip, fonts, original_images))
        audits.append({"panel": letter, "input": path.name,
                       "original_page_points": list(page.rect),
                       "preserved_text_spans": len(preserved),
                       "embedding_font_pt": 9.2,
                       "colourbar_font_pt": cbar["size"],
                       "new_colourbar_label": title,
                       "images_unchanged_after_text_edit": True})

    widths = [clip.width for _, clip, _, _ in working]
    if not math.isclose(widths[0], widths[1], abs_tol=1e-5):
        raise ValueError("Different page widths; refusing to rescale either input")
    width = widths[0]
    height = sum(clip.height for _, clip, _, _ in working) + PANEL_GAP_PT
    output = fitz.open()
    page = output.new_page(width=width, height=height)
    offset = 0.
    for (doc, clip, fonts, images), (_, letter, _) in zip(working, SOURCES):
        rectangle = fitz.Rect(0, offset, width, offset + clip.height)
        page.show_pdf_page(rectangle, doc, pno=0, clip=clip, keep_proportion=True)
        # Match the reference lettering at the same displayed figure width.
        # There is no point-size metadata in the PNG reference.
        label_size = REFERENCE_LETTER_EM_PX * width / REFERENCE_WIDTH_PX
        label_x = 77 * width / REFERENCE_WIDTH_PX
        matrix_top = images[0]["bbox"][1] - TOP_BLANK_CROP_PT + offset
        baseline_y = matrix_top + 28 * width / REFERENCE_WIDTH_PX
        page.insert_font(fontname="PanelBold", fontbuffer=fonts["bold"])
        page.insert_text((label_x, baseline_y), letter, fontsize=label_size,
                         fontname="PanelBold", color=(0, 0, 0))
        offset += clip.height + PANEL_GAP_PT

    pdf = destination / "consensus_heatmaps_panel_original_sizes.pdf"
    output.save(pdf, garbage=4, deflate=True)
    output.close()
    final = fitz.open(pdf)
    final_spans = spans(final[0])
    actual_counts = Counter(map(span_key, final_spans))
    required_counts = Counter(span_key(s) for group in expected_preserved_spans for s in group)
    if required_counts - actual_counts:
        raise AssertionError("Original text or font sizes changed in assembly")
    image_info = final[0].get_image_info(hashes=True)
    if Counter(i["digest"].hex() for i in image_info) != Counter(original_image_digests):
        raise AssertionError("An image was resampled or changed")
    for (_, clip, _, original_images), placed_images in zip(working, (image_info[:2], image_info[2:])):
        for original, placed in zip(original_images, placed_images):
            a, b = fitz.Rect(original["bbox"]), fitz.Rect(placed["bbox"])
            if abs(a.width - b.width) > .002 or abs(a.height - b.height) > .002:
                raise AssertionError("Heatmap or colourbar size changed")
    if input_hashes != {p.name: sha(p) for p in originals}:
        raise AssertionError("A source PDF was modified")
    # Raster export of the completed PDF, so PNG and PDF share one layout.
    final[0].get_pixmap(dpi=300, alpha=False).save(
        destination / "consensus_heatmaps_panel_original_sizes.png")
    audit = {"status": "PASS", "source_sha256": input_hashes,
             "page_count": 1, "page_points": [width, height],
             "source_scale_factor": 1.0, "no_resampling_or_redrawing": True,
             "original_text_font_sizes_preserved": True,
             "original_heatmap_and_colourbar_dimensions_preserved": True,
             "original_image_pixels_preserved": True,
             "panel_letter_font": "DejaVu Sans Bold", "panel_letter_size_pt": label_size,
             "panel_letter_reference": "FIG4BC_labeled.png; black lowercase bold lettering matched at equal displayed width",
             "source_top_blank_space_removed_pt": TOP_BLANK_CROP_PT,
             "panel_gap_pt": PANEL_GAP_PT, "panels": audits}
    (destination / "assembly_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS: original PDF artwork assembled at 100% scale")
    print("PASS: all retained text font sizes and image pixels/dimensions preserved")
    print("PASS: titles/subtitles removed; only colourbar wording replaced")
    print(pdf)


if __name__ == "__main__":
    main()
