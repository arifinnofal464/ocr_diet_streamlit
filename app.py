"""
Streamlit app: OCR Label Gizi (PaddleOCR) + NER (spaCy) + Rule Engine Diabetes.

Diadaptasi dari notebook Colab "OCR_PaddleOCR_PPStructureV3_Diet_Diabetes.ipynb".
Bagian training (spaCy NER training, dataset annotation) SENGAJA tidak
disertakan di sini -- app ini hanya menjalankan tahap INFERENCE:
  1. Upload foto label kemasan
  2. OCR (PaddleOCR) -> crop area "Informasi Nilai Gizi" + ekstrak baris teks
  3. NER (model spaCy yang sudah dilatih, folder `model-best/`)
  4. Rule Engine -> evaluasi kelayakan produk untuk diet Diabetes
"""

import os
import re
import json
import difflib
import logging

import cv2
import numpy as np
import pandas as pd
import streamlit as st

logging.getLogger("ppocr").setLevel(logging.ERROR)

st.set_page_config(page_title="OCR Label Gizi - Diet Diabetes", layout="wide")

# ------------------------------------------------------------------
# 1. KATA KUNCI AREA LABEL
# ------------------------------------------------------------------
NUTRITION_START_KEYWORDS = [
    "INFORMASI NILAI GIZI", "INFORMASI NILAI GIZI PER", "INFORMASI", "INFORMASI NILAI", "GIZI", "NUTRITION FACTS",
    "NILAI GIZI", "TAKARAN SAJI", "SERVING SIZE",
]

NUTRITION_END_KEYWORDS = [
    "KOMPOSISI", "BAHAN BAKU", "BAHAN-BAHAN", "INGREDIENTS", "CARA PENYAJIAN", "CARA SAJI",
    "PETUNJUK PENYIMPANAN", "KODE PRODUKSI", "TANGGAL KEDALUWARSA", "DIPRODUKSI OLEH",
    "NO. BPOM", "BPOM RI", "MENGANDUNG ALERGEN", "SARAN PENYAJIAN", "CUSTOMER CARE",
    "LAYANAN KONSUMEN", "TANPA PENGAWET", "PERSEN AKG", "PERSEN", "BERDASARKAN KEBUTUHAN",
]

NUTRITION_HEADER_ONLY_KEYWORDS = [
    "INFORMASI NILAI GIZI", "NUTRITION FACTS", "JUMLAH PER SAJIAN",
    "AMOUNT PER SERVING", "AKG", "DAILY VALUE", "%AKG", "% AKG", "AKG*",
]

NUTRITION_CANONICAL_TERMS = [
    "TAKARAN SAJI", "SERVING SIZE",
    "SAJIAN PER KEMASAN", "SERVINGS PER CONTAINER", "SERVING PER PACKAGE", "SERVINGS PER PACKAGE",
    "ENERGI TOTAL", "TOTAL ENERGY", "ENERGY", "CALORIES", "TOTAL CALORIES",
    "ENERGI DARI LEMAK", "ENERGY FROM FAT", "CALORIES FROM FAT",
    "ENERGI DARI LEMAK JENUH", "ENERGY FROM SATURATED FAT", "CALORIES FROM SATURATED FAT",
    "LEMAK TOTAL", "TOTAL FAT",
    "LEMAK TRANS", "TRANS FAT",
    "LEMAK JENUH", "SATURATED FAT",
    "LEMAK TAK JENUH TUNGGAL", "MONOUNSATURATED FAT",
    "LEMAK TAK JENUH GANDA", "POLYUNSATURATED FAT",
    "KOLESTEROL", "CHOLESTEROL",
    "PROTEIN",
    "KARBOHIDRAT TOTAL", "TOTAL CARBOHYDRATE", "TOTAL CARBOHYDRATES",
    "SERAT PANGAN", "DIETARY FIBER", "DIETARY FIBRE", "SERAT", "FIBER", "FIBRE",
    "GULA", "SUGAR", "SUGARS", "TOTAL SUGAR", "TOTAL SUGARS",
    "SUKROSA", "SUCROSE",
    "LAKTOSA", "LACTOSE",
    "GARAM NATRIUM", "SALT SODIUM",
    "GARAM", "SALT",
    "NATRIUM", "SODIUM",
]
FUZZY_MATCH_THRESHOLD = 0.78
_FUZZY_MAX_PHRASE_WORDS = 4

ACTIVITY_FACTORS = {
    "Sangat Ringan": 1.20, "Ringan": 1.375, "Sedang": 1.55, "Berat": 1.725, "Sangat Berat": 1.90,
}
CONSUMPTION_TYPE_FACTORS = {"Makanan Utama": 0.30, "Snack/Minuman": 0.10}

RULE_ENGINE_CONFIG = {
    "diabetes": {
        "nutrient_labels": {
            "GULA": {"target_key": "gula_g", "display": "Gula"},
            "KARBOHIDRAT": {"target_key": "karbohidrat_g", "display": "Karbohidrat"},
            "PROTEIN": {"target_key": "protein_g", "display": "Protein"},
            "LEMAK_TOTAL": {"target_key": "lemak_total_g", "display": "Lemak Total"},
            "LEMAK_JENUH": {"target_key": "lemak_jenuh_g", "display": "Lemak Jenuh"},
            "SERAT_PANGAN": {"target_key": "serat_g", "display": "Serat", "is_minimum_requirement": True},
            "KOLESTEROL": {"target_key": "kolesterol_mg", "display": "Kolesterol"},
            "MUFA": {"target_key": "mufa_g", "display": "Lemak Tak Jenuh Tunggal"},
            "PUFA": {"target_key": "pufa_g", "display": "Lemak Tak Jenuh Ganda"},
        },
    },
}


# ------------------------------------------------------------------
# 2. MODEL LOADING (cached — hanya dijalankan sekali per sesi server)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model PaddleOCR (hanya sekali) ...")
def load_ocr_engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang="id",
        ocr_version="PP-OCRv5",
        enable_mkldnn=False,
    )


@st.cache_resource(show_spinner="Memuat model NER spaCy (hanya sekali) ...")
def load_ner_model(model_path: str):
    import spacy
    return spacy.load(model_path)


# ------------------------------------------------------------------
# 3. FUNGSI BANTU OCR (diadaptasi dari notebook, Bagian 4)
# ------------------------------------------------------------------
def normalize(text):
    return re.sub(r"[^A-Z0-9 ]", "", text.upper()).strip()


def _fuzzy_ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def fuzzy_match_keyword(text, keywords, threshold=0.80):
    norm_text = normalize(text)
    if not norm_text:
        return None, 0.0
    best_kw, best_score = None, 0.0
    for kw in keywords:
        score = _fuzzy_ratio(norm_text, normalize(kw))
        if score > best_score:
            best_score = score
            best_kw = kw
    if best_score >= threshold:
        return best_kw, best_score
    return None, best_score


def run_full_ocr(ocr_engine, image_path):
    output = ocr_engine.predict(image_path)
    lines = []
    for res in output:
        res_dict = res["res"] if ("res" in res and isinstance(res["res"], dict)) else res
        texts = res_dict.get("rec_texts", [])
        scores = res_dict.get("rec_scores", [])
        polys = res_dict.get("rec_polys", [])
        for box, text, score in zip(polys, texts, scores):
            y_top = min(p[1] for p in box)
            x_left = min(p[0] for p in box)
            lines.append({"box": box, "text": text, "conf": float(score), "y_top": y_top, "x_left": x_left})

    lines.sort(key=lambda l: l["y_top"])
    heights = [max(p[1] for p in l["box"]) - min(p[1] for p in l["box"]) for l in lines] or [1]
    typical_height = sorted(heights)[len(heights) // 2]

    rows = []
    for item in lines:
        placed = False
        for row in rows:
            row_y_top = min(r["y_top"] for r in row)
            row_y_bottom = max(max(p[1] for p in r["box"]) for r in row)
            item_y_bottom = max(p[1] for p in item["box"])
            overlap = min(item_y_bottom, row_y_bottom) - max(item["y_top"], row_y_top)
            if overlap > 0.5 * min(item_y_bottom - item["y_top"], row_y_bottom - row_y_top, typical_height):
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    rows.sort(key=lambda row: min(r["y_top"] for r in row))
    ordered = []
    for row in rows:
        row.sort(key=lambda r: r["x_left"])
        ordered.extend(row)
    return ordered


def resize_if_too_large(image_path, max_side=1800):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, img)
    return image_path


def crop_section_by_keywords(ocr_engine, image_path, start_keywords, end_keywords,
                              padding=25, min_table_height_ratio=0.12):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, [], {"applied_steps": ["Error: Gambar tidak ditemukan"]}

    h, w, _ = img.shape
    lines = run_full_ocr(ocr_engine, image_path)
    if not lines:
        return None, None, [], {"applied_steps": ["OCR tidak mendeteksi teks"]}

    def get_line_bounds(line):
        box = line["box"]
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

    y_start = None
    header_x_min, header_x_max = 0, w

    for line in lines:
        text = line["text"]
        text_lower = text.lower()
        is_exact = any(kw.lower() in text_lower for kw in start_keywords)
        matched_kw, score = (None, 0.0) if is_exact else fuzzy_match_keyword(text, start_keywords, threshold=0.78)
        if is_exact or matched_kw:
            x_min, x_max, y_min, _ = get_line_bounds(line)
            if y_start is None or y_min < y_start:
                y_start = y_min
                header_x_min = max(0, x_min - 100)
                header_x_max = min(w, x_max + 300)

    if y_start is None:
        y_start = 0

    y_end = None
    for line in lines:
        text = line["text"]
        text_lower = text.lower()
        x_min, x_max, y_min, y_max = get_line_bounds(line)
        if y_start and y_min > y_start and (header_x_min <= x_min <= header_x_max):
            is_exact = any(kw.lower() in text_lower for kw in end_keywords)
            matched_kw, score = (None, 0.0) if is_exact else fuzzy_match_keyword(text, end_keywords, threshold=0.80)
            if is_exact or matched_kw:
                y_end = y_max
                break

    if y_end is None:
        y_end = h
    if (y_end - y_start) < (h * min_table_height_ratio):
        y_end = h

    y_start = max(0, int(y_start) - padding)
    y_end = min(h, int(y_end) + padding)

    if y_start >= y_end:
        return None, None, lines, {"applied_steps": ["Koordinat Pemotongan Y Tidak Valid"]}

    cropped = img[y_start:y_end, 0:w]
    if cropped.size == 0:
        return None, None, lines, {"applied_steps": ["Hasil pemotongan gambar kosong"]}

    filtered_section_lines = []
    for line in lines:
        _, _, y_min, y_max = get_line_bounds(line)
        y_center = (y_min + y_max) / 2
        if y_start <= y_center <= y_end:
            filtered_section_lines.append(line)

    preprocess_info = {"applied_steps": [f"Crop Y={y_start}-{y_end} (dari {h}px)"]}
    return cropped, (y_start, y_end), filtered_section_lines, preprocess_info


def _items_from_lines(lines):
    def _norm_kw(s):
        return re.sub(r"[^A-Z0-9 %]", "", s.upper()).strip()

    def _fix_zero_confusion(text):
        text = re.sub(r"\b[Oo](g|mg|kg|ml|l)\b", r"0\1", text)
        text = re.sub(r"\b[Oo]\b(?=\s*(kkal|kal|mg|mcg|g)\b)", "0", text, flags=re.IGNORECASE)
        return text

    items = []
    for l in lines:
        txt = l["text"].strip()
        if not txt:
            continue
        txt = _fix_zero_confusion(txt)
        ys = [p[1] for p in l["box"]]
        xs = [p[0] for p in l["box"]]
        items.append({
            "text": txt, "norm": _norm_kw(txt),
            "x_left": min(xs), "y_top": min(ys), "y_bottom": max(ys),
            "y_center": (min(ys) + max(ys)) / 2, "height": max(max(ys) - min(ys), 1),
        })
    return items


def _cluster_rows_improved(items):
    if not items:
        return [], []
    prepared = []
    for item in items:
        x_min, x_max = item["x_left"], item["x_left"]
        y_min, y_max = item["y_top"], item["y_bottom"]
        h = max(1, y_max - y_min)
        prepared.append({**item, "_x_min": item["x_left"], "_y_center": item["y_center"], "_height": h})
    prepared.sort(key=lambda it: it["_y_center"])
    rows = []
    for item in prepared:
        assigned = False
        for row in rows:
            row_y_center = np.mean([i["_y_center"] for i in row])
            row_avg_h = np.mean([i["_height"] for i in row])
            y_diff = abs(item["_y_center"] - row_y_center)
            allowed_diff = min(item["_height"], row_avg_h) * 0.55
            if y_diff <= allowed_diff:
                row.append(item)
                assigned = True
                break
        if not assigned:
            rows.append([item])
    rows.sort(key=lambda r: np.mean([i["_y_center"] for i in r]))
    return rows, []


def process_row_items_to_text(rows, max_x_gap_multi_column=600):
    display_rows = []
    for row in rows:
        if not row:
            continue
        row.sort(key=lambda item: item["_x_min"])
        sub_rows, current_sub = [], [row[0]]
        for i in range(1, len(row)):
            prev_x_max = current_sub[-1].get("x_right", current_sub[-1]["_x_min"] + 50)
            curr_x_min = row[i]["_x_min"]
            if (curr_x_min - prev_x_max) > max_x_gap_multi_column:
                sub_rows.append(current_sub)
                current_sub = [row[i]]
            else:
                current_sub.append(row[i])
        sub_rows.append(current_sub)
        for sub in sub_rows:
            line_text = " ".join(item["text"] for item in sub)
            if line_text.strip():
                display_rows.append(line_text)
    return display_rows


_NUTRIENT_NAME_HINTS = [
    "LEMAK", "KOLESTEROL", "PROTEIN", "KARBOHIDRAT", "SERAT", "GULA",
    "GARAM", "NATRIUM", "ENERGI", "KALORI", "VITAMIN", "KALSIUM", "ZAT BESI",
]
_UNIT_CONFUSION_PATTERN = re.compile(r"(\d)(\s+)[9Oo](?!\w)")
_UNIT_CONFUSION_FUSED_PATTERN = re.compile(r"\b(\d{1,3})9(?=\s*[\d.,]+\s*%)")


def fix_ocr_unit_confusions_bulk(lines):
    fixed_lines, corrected_indices = [], []
    for i, line in enumerate(lines):
        norm = normalize(line)
        if not any(h in norm for h in _NUTRIENT_NAME_HINTS):
            fixed_lines.append(line)
            continue
        fixed, n1 = _UNIT_CONFUSION_PATTERN.subn(lambda m: f"{m.group(1)}{m.group(2)}g", line)
        fixed, n2 = _UNIT_CONFUSION_FUSED_PATTERN.subn(lambda m: f"{m.group(1)}g", fixed)
        fixed_lines.append(fixed)
        if n1 + n2 > 0:
            corrected_indices.append(i)
    return fixed_lines, corrected_indices


def _similarity_ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def _norm_kw(s):
    return re.sub(r"[^A-Z0-9 %]", "", s.upper()).strip()


def fuzzy_match_nutrition_term(candidate_text, threshold=FUZZY_MATCH_THRESHOLD):
    norm_candidate = _norm_kw(candidate_text)
    if not norm_candidate:
        return None, 0.0
    best_term, best_score = None, 0.0
    for term in NUTRITION_CANONICAL_TERMS:
        score = _similarity_ratio(norm_candidate, term)
        if score > best_score:
            best_score = score
            best_term = term
    if best_score >= threshold:
        return best_term, best_score
    return None, best_score


def fuzzy_correct_nutrition_label(line_text, threshold=FUZZY_MATCH_THRESHOLD, max_phrase_words=_FUZZY_MAX_PHRASE_WORDS):
    match = re.search(r"\d", line_text)
    if match:
        label_part = line_text[:match.start()].strip()
        rest_part = line_text[match.start():]
    else:
        label_part = line_text.strip()
        rest_part = ""
    if not label_part:
        return line_text, None

    words = label_part.split()
    best_term, best_span = None, None
    for span_len in range(min(max_phrase_words, len(words)), 0, -1):
        window = " ".join(words[:span_len])
        term, score = fuzzy_match_nutrition_term(window, threshold=threshold)
        if term:
            best_term, best_span = term, span_len
            break
    if best_term is None:
        return line_text, None

    original_label_window = " ".join(words[:best_span])
    already_exact = _norm_kw(original_label_window) == best_term
    leftover_words = words[best_span:]
    corrected_label = best_term.title()
    if leftover_words:
        corrected_label = corrected_label + " " + " ".join(leftover_words)
    corrected_line = f"{corrected_label} {rest_part}".strip() if rest_part else corrected_label
    if already_exact:
        return corrected_line, None
    return corrected_line, {"original_label": original_label_window, "corrected_label": best_term.title()}


def fuzzy_correct_nutrition_lines_bulk(lines, threshold=FUZZY_MATCH_THRESHOLD):
    corrected_lines, corrections_log = [], []
    for i, line in enumerate(lines):
        corrected, info = fuzzy_correct_nutrition_label(line, threshold=threshold)
        corrected_lines.append(corrected)
        if info:
            info["line_index"] = i
            corrections_log.append(info)
    return corrected_lines, corrections_log


_NUTRIENT_PATTERN = re.compile(
    r"([A-Za-zÀ-ÿ()\.\s]+?)[:\s]+([\d.,]+)\s*(kkal|kal|mcg|mg|g)\b\s*(?:([\d.,]+)\s*%)?",
    re.IGNORECASE,
)
_TAKARAN_PATTERN = re.compile(r"Takaran\s*Saji\b.*?([\d.,]+\s*\w+)", re.IGNORECASE)
_SAJIAN_PATTERN = re.compile(r"([\d.,]+)\s*Sajian\s*per\s*Kemasan", re.IGNORECASE)
_SAJIAN_PATTERN_AFTER = re.compile(r"Sajian\s*per\s*Kemasan.*?([\d.,]+)", re.IGNORECASE)


def parse_nutrition_lines(lines):
    result = {"basis": None, "takaran_saji": None, "sajian_per_kemasan": None, "nutrients": {}}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        norm = normalize(line)
        if any(h in norm for h in NUTRITION_HEADER_ONLY_KEYWORDS):
            continue
        m = _TAKARAN_PATTERN.search(line)
        if m:
            result["takaran_saji"] = m.group(1).strip()
            result["basis"] = "sajian"
            continue
        m = _SAJIAN_PATTERN.search(line) or _SAJIAN_PATTERN_AFTER.search(line)
        if m:
            result["sajian_per_kemasan"] = float(m.group(1).replace(",", "."))
            continue
        matches = _NUTRIENT_PATTERN.findall(line)
        for name, value, unit, pct in matches:
            name = name.strip(" :")
            if not name:
                continue
            entry = {"value": float(value.replace(",", ".")), "unit": unit.lower()}
            if pct:
                entry["akg_percent"] = float(pct.replace(",", "."))
            result["nutrients"][name] = entry
    return result


def run_extraction_pipeline(ocr_engine, image_path):
    output_record = {
        "image_file": image_path, "diet_choice": "Diabetes", "extraction_method": None,
        "extracted_text_lines": [], "cropped_image_path": None, "preprocessing": None,
    }
    cropped, y_range, section_lines, preprocess_info = crop_section_by_keywords(
        ocr_engine, image_path, NUTRITION_START_KEYWORDS, NUTRITION_END_KEYWORDS
    )
    output_record["preprocessing"] = preprocess_info

    if cropped is not None and cropped.size > 0:
        output_record["extraction_method"] = "keyword_based"
        crop_path = os.path.splitext(image_path)[0] + "_cropped.jpg"
        cv2.imwrite(crop_path, cropped)
        output_record["cropped_image_path"] = crop_path

        items = _items_from_lines(section_lines)
        rows, _ = _cluster_rows_improved(items)
        display_rows = process_row_items_to_text(rows, max_x_gap_multi_column=600)
        corrected_rows, corrected_indices = fix_ocr_unit_confusions_bulk(display_rows)
        fuzzy_rows, fuzzy_corrections_log = fuzzy_correct_nutrition_lines_bulk(corrected_rows)

        output_record["extracted_text_lines"] = fuzzy_rows
        output_record["fuzzy_term_corrections"] = fuzzy_corrections_log

        nutrition_data = parse_nutrition_lines(fuzzy_rows)
        output_record["nutrition_data"] = nutrition_data
    return output_record


# ------------------------------------------------------------------
# 4. FUNGSI PROFIL & KEBUTUHAN NUTRISI HARIAN (Bagian 6.5)
# ------------------------------------------------------------------
def calculate_bbi(height_cm):
    base = height_cm - 100
    return base - (0.10 * base)


def calculate_energi_basal(gender, bbi):
    return bbi * 30 if gender == "Laki-laki" else bbi * 25


def calculate_energi_total(energi_basal, activity_level):
    return energi_basal * ACTIVITY_FACTORS[activity_level]


def calculate_daily_nutrition_targets(energi_total):
    return {
        "karbohidrat_g": round((0.65 * energi_total) / 4, 2),
        "protein_g": round((0.20 * energi_total) / 4, 2),
        "lemak_total_g": round((0.25 * energi_total) / 9, 2),
        "lemak_jenuh_g": round((0.07 * energi_total) / 9, 2),
        "pufa_g": round((0.10 * energi_total) / 9, 2),
        "mufa_g": round((0.15 * energi_total) / 9, 2),
        "gula_g": round((0.05 * energi_total) / 4, 2),
        "kolesterol_mg": 200.0,
        "serat_g": 30.0,
    }


def calculate_consumption_limits(daily_targets, jenis_konsumsi):
    factor = CONSUMPTION_TYPE_FACTORS[jenis_konsumsi]
    return {k: round(v * factor, 2) for k, v in daily_targets.items()}


# ------------------------------------------------------------------
# 5. NER (Bagian 16) & PARSING NILAI NUMERIK (Bagian 22)
# ------------------------------------------------------------------
def extract_entities_with_confidence(nlp, doc, beam_width=16):
    ner = nlp.get_pipe("ner")
    beams = ner.beam_parse([doc], beam_width=beam_width)
    scored = ner.scored_ents(beams)[0] if beams else {}
    results = []
    for ent in doc.ents:
        key = (ent.start, ent.end, ent.label_)
        confidence = scored.get(key)
        results.append({
            "entity": ent.label_, "value": ent.text,
            "start_char": ent.start_char, "end_char": ent.end_char,
            "confidence": round(confidence, 4) if confidence is not None else None,
        })
    return results


_NUMERIC_VALUE_PATTERN = re.compile(r"([\d.,]+)\s*(kkal|kal|mcg|mg|g)?", re.IGNORECASE)


def parse_numeric_entity(value_text):
    match = _NUMERIC_VALUE_PATTERN.search(value_text)
    if not match or not match.group(1):
        return None, None
    number = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower() if match.group(2) else None
    return number, unit


# ------------------------------------------------------------------
# 6. RULE ENGINE (Bagian 23)
# ------------------------------------------------------------------
class RuleEngine:
    def __init__(self, config=None):
        self.config = config if config is not None else RULE_ENGINE_CONFIG

    def evaluate_diabetes(self, entities, nutrition_data, consumption_limits):
        cfg = self.config["diabetes"]
        nutrient_cfg = cfg["nutrient_labels"]
        nutrition_data = nutrition_data or {}
        raw_servings = nutrition_data.get("sajian_per_kemasan")

        reason = []
        if raw_servings is None:
            servings_per_container = 1
            reason.append("Jumlah Sajian per Kemasan tidak ditemukan pada hasil OCR -- diasumsikan 1 sajian.")
        else:
            try:
                servings_per_container = max(1, int(round(float(raw_servings))))
            except (TypeError, ValueError):
                servings_per_container = 1

        params = []
        for label, meta in nutrient_cfg.items():
            display = meta["display"]
            is_min = meta.get("is_minimum_requirement", False)
            limit = consumption_limits.get(meta["target_key"]) if consumption_limits else None
            ent = next((e for e in entities if e["entity"] == label), None)
            if ent is None:
                reason.append(f"{display}: entitas tidak ditemukan pada hasil OCR/NER -- dilewati.")
                continue
            value_per_serving, _unit = parse_numeric_entity(ent["value"])
            if value_per_serving is None or limit is None:
                continue
            params.append({"label": label, "display": display, "is_min": is_min, "limit": limit, "value_per_serving": value_per_serving})

        max_daily_servings = 0
        stopped_at_serving = None
        exceeded_nutrients = []
        for n in range(1, servings_per_container + 1):
            n_exceeded = []
            for p in params:
                if p["is_min"]:
                    continue
                nilai_n = round(p["value_per_serving"] * n, 2)
                if nilai_n > p["limit"]:
                    n_exceeded.append({"Parameter": p["display"], "Nilai Produk": nilai_n, "Batas": round(p["limit"], 2)})
            if n_exceeded:
                stopped_at_serving = n
                exceeded_nutrients = n_exceeded
                break
            max_daily_servings = n

        evaluated_n = stopped_at_serving if stopped_at_serving is not None else servings_per_container
        rows = []
        for p in params:
            nilai_n = round(p["value_per_serving"] * evaluated_n, 2)
            lulus = (nilai_n >= p["limit"]) if p["is_min"] else (nilai_n <= p["limit"])
            rows.append({"Parameter": p["display"], "Nilai Produk": nilai_n, "Batas": round(p["limit"], 2), "Status": "✓" if lulus else "✗"})

        if stopped_at_serving == 1:
            category = "Tidak Aman"
            max_daily_servings = 0
        elif stopped_at_serving is not None:
            category = "Dibatasi"
        else:
            category = "Aman"

        return {
            "category": category, "max_daily_servings": max_daily_servings,
            "servings_per_container": servings_per_container,
            "stopped_at_serving": stopped_at_serving,
            "exceeded_nutrients": exceeded_nutrients, "rows": rows, "reason": reason,
        }


# ------------------------------------------------------------------
# 7. STREAMLIT UI
# ------------------------------------------------------------------
NER_MODEL_PATH = os.environ.get("NER_MODEL_PATH", "model-best")

st.title("🥗 OCR Label Gizi — Evaluasi Kelayakan Produk untuk Diet Diabetes")
st.caption("PaddleOCR (PP-OCRv5) + spaCy NER + Rule Engine — diadaptasi dari notebook riset.")

with st.sidebar:
    st.header("1️⃣ Profil Pengguna")
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.number_input("Umur (tahun)", min_value=1, max_value=120, value=30)
    weight = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=300.0, value=60.0)
    height = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, value=160.0)
    activity = st.selectbox("Tingkat Aktivitas", list(ACTIVITY_FACTORS.keys()))
    konsumsi = st.selectbox("Jenis Konsumsi Produk", list(CONSUMPTION_TYPE_FACTORS.keys()))

    bbi = calculate_bbi(height)
    energi_basal = calculate_energi_basal(gender, bbi)
    energi_total = calculate_energi_total(energi_basal, activity)
    daily_targets = calculate_daily_nutrition_targets(energi_total)
    consumption_limits = calculate_consumption_limits(daily_targets, konsumsi)

    st.markdown(f"**BBI:** {bbi:.1f} kg &nbsp;|&nbsp; **TDEE:** {energi_total:.0f} kkal")
    with st.expander("Lihat kebutuhan nutrisi harian & batas konsumsi"):
        st.json({"daily_targets": daily_targets, "consumption_limits (batas per produk)": consumption_limits})

st.header("2️⃣ Upload Foto Label Kemasan")
uploaded_file = st.file_uploader("Foto label gizi (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if not os.path.isdir(NER_MODEL_PATH):
        st.error(
            f"Folder model NER spaCy tidak ditemukan di `{NER_MODEL_PATH}`. "
            "Pastikan folder `model-best/` (hasil training) sudah ada di root repo, "
            "atau atur environment variable `NER_MODEL_PATH`."
        )
        st.stop()

    tmp_path = os.path.join("/tmp", uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    tmp_path = resize_if_too_large(tmp_path)

    ocr_engine = load_ocr_engine()
    nlp_ner = load_ner_model(NER_MODEL_PATH)

    with st.spinner("Menjalankan OCR (mencari area 'Informasi Nilai Gizi') ..."):
        output_record = run_extraction_pipeline(ocr_engine, tmp_path)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Foto Asli")
        st.image(tmp_path, use_container_width=True)
    with col2:
        st.subheader("Area Ter-crop")
        if output_record["cropped_image_path"]:
            st.image(output_record["cropped_image_path"], use_container_width=True)
        else:
            st.warning("Area 'Informasi Nilai Gizi' tidak ditemukan. Coba foto dengan sudut/pencahayaan lebih baik.")

    if output_record["cropped_image_path"]:
        st.subheader("3️⃣ Teks Hasil OCR")
        st.text("\n".join(output_record["extracted_text_lines"]) or "(tidak ada teks terbaca)")

        ner_text_input = "\n".join(output_record["extracted_text_lines"])
        with st.spinner("Menjalankan NER ..."):
            doc = nlp_ner(ner_text_input)
            entities = extract_entities_with_confidence(nlp_ner, doc)

        st.subheader("4️⃣ Entitas Terdeteksi (NER)")
        if entities:
            st.dataframe(pd.DataFrame(entities)[["entity", "value", "confidence"]])
        else:
            st.warning("Tidak ada entitas gizi yang terdeteksi oleh model NER.")

        st.subheader("5️⃣ Hasil Evaluasi Rule Engine — Diet Diabetes")
        rule_engine = RuleEngine()
        result = rule_engine.evaluate_diabetes(entities, output_record.get("nutrition_data"), consumption_limits)

        status_color = {"Aman": "success", "Dibatasi": "warning", "Tidak Aman": "error"}[result["category"]]
        getattr(st, status_color)(f"**Status: {result['category'].upper()}**")
        st.markdown(
            f"- Maksimal konsumsi harian: **{result['max_daily_servings']} sajian** "
            f"(dari {result['servings_per_container']} sajian per kemasan)"
        )
        if result["rows"]:
            st.dataframe(pd.DataFrame(result["rows"]))
        if result["reason"]:
            with st.expander("Detail alasan evaluasi"):
                for r in result["reason"]:
                    st.write(f"• {r}")

        with st.expander("Data mentah (JSON) untuk debugging"):
            st.json({
                "nutrition_data": output_record.get("nutrition_data"),
                "entities": entities,
                "rule_engine_result": result,
            })
else:
    st.info("Silakan upload foto label kemasan untuk memulai.")
