# app.py
# Gradio UI for RF-DETR inference on either:
#  - a page from a PDF (rendered to an image), or
#  - an uploaded image.
#
# Run:
#   pip install gradio pymupdf pillow matplotlib
#   # and your RF-DETR package installed per its instructions
#   python app.py
#
# Open the printed local URL in your browser.

from pathlib import Path
import json
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
import gradio as gr

from rfdetr import RFDETRBase


# ----------------------------
# Configuration defaults
# ----------------------------
DEFAULT_PDF_PATH = Path("sample_drawings.pdf")  # <- change if you like
DEFAULT_CKPT     = Path("outputs_10_epochs/checkpoint_best_ema.pth")
DEFAULT_CONF_TH  = 0.5
DEFAULT_ZOOM     = 2.0

# ----------------------------
# Helpers
# ----------------------------
_model = None  # lazy-loaded global to avoid reloading on every click

def get_model(ckpt_path: str | Path) -> RFDETRBase:
    global _model
    ckpt_path = Path(ckpt_path)
    if _model is None:
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        _model = RFDETRBase(pretrain_weights=str(ckpt_path))
        try:
            _model.optimize_for_inference()
        except Exception:
            pass
    return _model

def pdf_to_pil_pages(pdf_path: Path, zoom: float = 2.0) -> List[Image.Image]:
    """Render each PDF page to a PIL RGB image using PyMuPDF."""
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    pages = []
    with fitz.open(pdf_path) as doc:
        mat = fitz.Matrix(zoom, zoom)
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
    return pages

def to_numpy(x):
    if x is None: return None
    if hasattr(x, "detach"): return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):    return x.cpu().numpy()
    return np.asarray(x)

def parse_preds(pred) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize RF-DETR predict() outputs into numpy arrays:
    returns (xyxy[N,4], scores[N], labels[N]).
    """
    try:
        import supervision as sv  # optional
        if isinstance(pred, sv.Detections):
            xyxy   = pred.xyxy
            scores = pred.confidence if pred.confidence is not None else np.ones((len(pred),), dtype=float)
            labels = pred.class_id   if pred.class_id   is not None else np.zeros((len(pred),), dtype=int)
            return xyxy, scores, labels
    except Exception:
        pass

    if isinstance(pred, list) and len(pred) > 0:
        pred = pred[0]

    if isinstance(pred, dict):
        xyxy   = to_numpy(pred.get("boxes"))  or np.zeros((0, 4), dtype=float)
        scores = to_numpy(pred.get("scores")) or np.ones((xyxy.shape[0],), dtype=float)
        labels = to_numpy(pred.get("labels")) or np.zeros((xyxy.shape[0]), dtype=int)
        return xyxy, scores, labels

    # Unknown → empty
    return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=int)

def draw_rect(draw: ImageDraw.ImageDraw, box, color=(255, 0, 0), width=3):
    x1, y1, x2, y2 = map(int, box)
    for t in range(width):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

def overlay_predictions(img: Image.Image, boxes: np.ndarray, scores: np.ndarray) -> Image.Image:
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    for box, s in zip(boxes, scores):
        draw_rect(draw, box, color=(255, 0, 0), width=3)
        x1, y1, _, _ = map(int, box)
        draw.text((x1, max(0, y1 - 14)), f"{float(s):.2f}", fill=(255, 0, 0))
    return vis

def run_inference_on_image(
    image: Image.Image,
    ckpt_path: str | Path,
    conf_th: float,
) -> Tuple[Image.Image, str]:
    """Run RF-DETR on a PIL image and return overlay + summary JSON."""
    model = get_model(ckpt_path)
    raw = model.predict(image, threshold=conf_th)  # some builds pre-filter, we filter again for safety
    boxes, scores, labels = parse_preds(raw)
    if scores.size > 0:
        keep = scores >= conf_th
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    vis = overlay_predictions(image, boxes, scores)
    summary = {
        "num_detections": int(len(boxes)),
        "threshold": float(conf_th),
        "boxes_xyxy_first5": boxes[:5].round(1).tolist(),
        "scores_first5": [float(x) for x in scores[:5]],
        "labels_first5": [int(x) for x in labels[:5]],
        "image_size": image.size,
    }
    return vis, json.dumps(summary, indent=2)

# ----------------------------
# Gradio handlers
# ----------------------------
def infer_handler(
    mode: str,
    pdf_file,            # gr.File (pdf) or None
    page_number: int,
    uploaded_image,      # gr.Image or None
    conf_th: float,
    zoom: float,
    ckpt_path: str,
    fallback_pdf_path: str,
):
    """
    mode: 'PDF' or 'Image'
    pdf_file: a temp path from gradio or None
    uploaded_image: PIL image or None
    """
    try:
        # Decide source image
        if mode == "PDF":
            # Prefer uploaded pdf, else fallback path
            if pdf_file is not None and getattr(pdf_file, "name", None):
                pdf_path = Path(pdf_file.name)
            else:
                pdf_path = Path(fallback_pdf_path) if fallback_pdf_path else DEFAULT_PDF_PATH

            pages = pdf_to_pil_pages(pdf_path, zoom=max(0.5, float(zoom)))
            if len(pages) == 0:
                return None, "No pages found in PDF."
            if not (1 <= page_number <= len(pages)):
                return None, f"Invalid page_number. Must be 1..{len(pages)} for this PDF."

            image = pages[page_number - 1]

        else:  # Image
            if uploaded_image is None:
                return None, "Please upload an image."
            image = uploaded_image

        # Inference
        ckpt = ckpt_path.strip() if ckpt_path.strip() else str(DEFAULT_CKPT)
        overlay, summary = run_inference_on_image(image, ckpt, float(conf_th))
        return overlay, summary

    except Exception as e:
        return None, f"Error: {e}"

# ----------------------------
# Build UI
# ----------------------------
with gr.Blocks(title="RF-DETR: PDF Page / Image Inference") as demo:
    gr.Markdown("## RF-DETR Inference • PDF Page or Uploaded Image")

    with gr.Row():
        mode = gr.Radio(choices=["PDF", "Image"], value="PDF", label="Input Mode")

    with gr.Tab("PDF"):
        pdf_file   = gr.File(label="Upload PDF (optional; else use fallback path)", file_types=[".pdf"], interactive=True)
        fallback   = gr.Textbox(label="Fallback PDF Path", value=str(DEFAULT_PDF_PATH), interactive=True)
        page_num   = gr.Number(label="Page Number (1-based)", value=1, precision=0)
        zoom_in    = gr.Slider(label="Render Zoom (PDF → Image)", minimum=0.5, maximum=4.0, value=DEFAULT_ZOOM, step=0.1)

    with gr.Tab("Image"):
        img_in = gr.Image(type="pil", label="Upload Image")

    with gr.Row():
        ckpt_in  = gr.Textbox(label="Checkpoint Path (.pth)", value=str(DEFAULT_CKPT), interactive=True)
        conf_in  = gr.Slider(label="Confidence Threshold", minimum=0.05, maximum=0.95, value=DEFAULT_CONF_TH, step=0.05)

    run_btn = gr.Button("Run Inference", variant="primary")

    with gr.Row():
        out_img = gr.Image(type="pil", label="Predictions Overlay")
        out_txt = gr.Code(label="Summary JSON")

    def dispatch(mode_sel, pdf_f, page_n, img, conf, zoom_val, ckpt, fb_pdf):
        return infer_handler(mode_sel, pdf_f, int(page_n or 1), img, conf, zoom_val, ckpt, fb_pdf)

    run_btn.click(
        fn=dispatch,
        inputs=[mode, pdf_file, page_num, img_in, conf_in, zoom_in, ckpt_in, fallback],
        outputs=[out_img, out_txt],
    )

if __name__ == "__main__":
    demo.launch(inline=True, share=False, show_api=False)