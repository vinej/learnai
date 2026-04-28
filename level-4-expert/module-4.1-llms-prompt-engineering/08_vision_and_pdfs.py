"""
08 — Multimodal inputs: images and PDFs

Claude can take images and PDFs alongside text. Two ways to provide them:
  • base64-encoded inline (small files).
  • via a URL (Anthropic's `url` source type — public images, no auth).

This file:
  1. Generates a tiny synthetic image with PIL.
  2. Sends it (base64) along with a question.
  3. Mentions the same pattern for PDFs.

Run: python 08_vision_and_pdfs.py    # requires ANTHROPIC_API_KEY
"""

import base64
import io
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw, ImageFont

from _common import DEFAULT_MODEL, print_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# Make a synthetic image
# ───────────────────────────────────────────────────────────
def make_image() -> bytes:
    img = Image.new("RGB", (300, 100), color=(240, 240, 250))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((30, 30), "Hello, Claude!", fill=(20, 20, 20), font=font)
    draw.rectangle([(10, 10), (290, 90)], outline=(70, 130, 180), width=3)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


img_bytes = make_image()
img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
out = Path(__file__).parent / "scratch"
out.mkdir(exist_ok=True)
(out / "demo.png").write_bytes(img_bytes)
print(f"saved demo image to: {out / 'demo.png'}  ({len(img_bytes)} bytes)")


# ───────────────────────────────────────────────────────────
# Send image + text question
# ───────────────────────────────────────────────────────────
response = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=300,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": "What text appears in this image? "
                            "Describe the visual style in one short phrase.",
                },
            ],
        },
    ],
)
print("\nresponse:")
print(response.content[0].text.strip())
print_usage("vision", response.usage)


# ───────────────────────────────────────────────────────────
# Sending an image by URL (public, no auth)
# ───────────────────────────────────────────────────────────
# Replace the source block with:
#   {"type": "image",
#    "source": {"type": "url", "url": "https://..."}}
# Useful when you have a CDN-hosted image and want to skip the base64 dance.


# ───────────────────────────────────────────────────────────
# PDFs: same pattern, different type
# ───────────────────────────────────────────────────────────
# Claude can read PDFs as IMAGES (one page each) or, more efficiently, via
# the documents API:
#
#   {"type": "document",
#    "source": {"type": "base64", "media_type": "application/pdf",
#               "data": <base64 pdf bytes>}}
#
# Pages are processed in order. Behind the scenes Claude both OCRs and
# rasterizes them, so handwriting / scanned PDFs work too.


# ───────────────────────────────────────────────────────────
# Cost notes
# ───────────────────────────────────────────────────────────
# Image and PDF tokens count against `input_tokens`. The exact token cost
# depends on resolution. To control cost:
#   • Resize big images to ~1024 px on the long side BEFORE sending.
#   • For multi-page PDFs, consider extracting text first and sending
#     plaintext if you don't need visual layout (much cheaper).
# `response.usage.input_tokens` includes the image's token contribution —
# use it to budget.


# ───────────────────────────────────────────────────────────
# What vision is good at
# ───────────────────────────────────────────────────────────
# • Reading text in screenshots / forms / receipts.
# • Describing scenes, objects, charts.
# • Spatial reasoning ("what's left of the bowl?").
# • Diagram understanding (architecture, mindmaps, flowcharts).
# • Light visual QA but NOT pixel-perfect ("count how many people are wearing red").
#
# What it's NOT good at:
#   - Highly precise object detection or counting in cluttered scenes.
#     For that, use a real object detector (Module 3.2.06).
#   - Reading very small text in low-resolution images.
