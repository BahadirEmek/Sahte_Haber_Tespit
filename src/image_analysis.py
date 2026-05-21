"""Image analysis: ELA manipulation detection, EXIF extraction, and optional OCR."""

from __future__ import annotations

import io

try:
    from PIL import Image, ImageChops, ImageEnhance
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


def extract_text_from_image(image: "Image.Image") -> str:
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(image, lang="eng+tur").strip()
    except Exception:
        try:
            return pytesseract.image_to_string(image).strip()
        except Exception:
            return ""


def perform_ela(image: "Image.Image", quality: int = 90) -> tuple["Image.Image", float]:
    """Error Level Analysis — returns enhanced diff image and manipulation score 0–100."""
    try:
        rgb = image.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        diff = ImageChops.difference(rgb, recompressed)
        enhanced = ImageEnhance.Brightness(diff).enhance(15)

        pixels = list(diff.getdata())
        avg = sum(sum(p) / 3 for p in pixels) / max(len(pixels), 1)
        score = min(round(avg * 15, 2), 100.0)
        return enhanced, score
    except Exception:
        return image, 0.0


def extract_exif(image: "Image.Image") -> dict[str, str]:
    try:
        raw = image.getexif()
        if not raw:
            return {}
        result = {}
        for tag_id, val in raw.items():
            tag = TAGS.get(tag_id, str(tag_id))
            if isinstance(val, bytes):
                if len(val) > 200:
                    continue
                val = val.decode("utf-8", errors="ignore")
            result[str(tag)] = str(val)
        return result
    except Exception:
        return {}


def analyze_image(image_bytes: bytes) -> dict:
    """Full image analysis: ELA, EXIF, OCR, and suspicion signals."""
    result: dict = {
        "ocr_text": "",
        "ela_image_bytes": None,
        "ela_score": 0.0,
        "exif": {},
        "image_size": (0, 0),
        "suspicious_signals": [],
        "ocr_available": TESSERACT_AVAILABLE,
    }

    if not PIL_AVAILABLE:
        result["suspicious_signals"].append("Pillow kütüphanesi yüklü değil; görsel analiz devre dışı.")
        return result

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        result["suspicious_signals"].append(f"Görüntü açılamadı: {exc}")
        return result

    result["image_size"] = image.size
    img_format = (image.format or "").upper()
    is_jpeg = img_format in ("JPEG", "JPG")

    if TESSERACT_AVAILABLE:
        result["ocr_text"] = extract_text_from_image(image)

    ela_img, ela_score = perform_ela(image)
    result["ela_score"] = ela_score
    buf = io.BytesIO()
    ela_img.save(buf, format="PNG")
    result["ela_image_bytes"] = buf.getvalue()

    result["exif"] = extract_exif(image)

    signals: list[str] = []
    w, h = image.size

    if w < 200 or h < 100:
        signals.append(f"Cok dusuk cozunurluk ({w}x{h}) — ekran goruntusu veya alinti olabilir.")

    if ela_score > 20:
        signals.append(f"Yuksek ELA skoru ({ela_score:.1f}/100) — goruntu muhtemelen duzenlenmis.")
    elif ela_score > 10:
        signals.append(f"Orta ELA skoru ({ela_score:.1f}/100) — hafif duzenleme izleri mevcut.")
    else:
        signals.append(f"Dusuk ELA skoru ({ela_score:.1f}/100) — belirgin duzenleme izi yok.")

    if not is_jpeg and img_format:
        signals.append(
            f"Goruntu JPEG formatinda degil ({img_format}) — ELA skoru yaniltici olabilir, "
            "kayipsiz formatlar icin yuksek skor normaldir."
        )

    if not result["exif"]:
        signals.append("EXIF metadata yok — goruntu yeniden kaydedilmis veya sosyal medyadan indirilmis.")
    else:
        sw = result["exif"].get("Software", "")
        if any(x in sw.lower() for x in ("photoshop", "gimp", "affinity", "lightroom", "canva")):
            signals.append(f"Grafik editorü tespit edildi: {sw}")
        dt = result["exif"].get("DateTimeOriginal") or result["exif"].get("DateTime", "")
        if dt:
            signals.append(f"Goruntu tarihi: {dt}")

    result["suspicious_signals"] = signals
    return result
