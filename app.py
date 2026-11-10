import os
import re
import json
import base64
import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
try:
    import streamlit.components.v1 as components
except Exception:
    components = None

# ১. পেজ কনফিগারেশন
st.set_page_config(
    page_title="Practice Sheet Studio",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ২. বাহ্যিক style.css ফাইল লোড করা
def load_external_css(css_file="style.css"):
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_external_css()

# ৩. ট্যাগ ফাইল লোড (question_tags.txt ও solution_tags.txt)
DEFAULT_Q_TAGS = [
    "Q", "Question", "Ques", "Prob", "Problem", "Ex", "Example",
    "প্রশ্ন", "এমসিকিউ", "সমস্যা", "উদাহরণ", "❑", "▫", "▪", "■", "□"
]
DEFAULT_SOL_TAGS = [
    "Sol", "Soln", "Solⁿ", "Sol^n", "So1", "So1n", "S0l", "S0ln", "5ol", "5oln",
    "Solution", "Ans", "Answer", "Shortcut", "Hint", "Hints", "Note", "[Note", "[Note:",
    "উত্তর", "উত্ত", "উওর", "উত্তব", "উত্তন", "উত্তম", "ডত্তর", "ডওর", "ডত্তব", "ত্তর", "ত্তব", "উঃ", "উ:", "ত্তঃ", "ত্ত:",
    "সমাধান", "সমাধা", "মাধান", "সলভ", "সল্‌ভ",
    "ব্যাখ্যা", "ব্যখ্যা", "ব্যাখা", "ব্যখা", "্যাখ্যা", "াখ্যা", "রাখ্যা", "য্যাখ্যা", "বাখ্যা", "ব্যাখ",
    "বিকল্প", "িকল্প", "শর্টকাট",
    "প্র্যাকটিস প্রবলেম", "প্র্যাকটিস", "প্রশ্ন-সলভ", "প্রশ্ন-সল্‌ভ",
    "Written", "লিখিত", "বিগত ভর্তি", "DËi", "mgvavb", "e¨vL¨v"
]

def load_tag_file(filename, defaults):
    loaded = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            loaded = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    combined = list(dict.fromkeys(loaded + defaults))
    return combined if combined else defaults

Q_TAGS = load_tag_file("question_tags.txt", DEFAULT_Q_TAGS)
SOL_TAGS = load_tag_file("solution_tags.txt", DEFAULT_SOL_TAGS)

def parse_box_selection_string(sel_str, max_count):
    sel_str = sel_str.strip()
    if not sel_str:
        return set(range(max_count))
    selected = set()
    for part in sel_str.split(","):
        part = part.strip()
        if "-" in part:
            b = part.split("-", 1)
            if b[0].strip().isdigit() and b[1].strip().isdigit():
                for idx in range(max(1, int(b[0])), min(max_count, int(b[1])) + 1):
                    selected.add(idx - 1)
        elif part.isdigit():
            idx = int(part)
            if 1 <= idx <= max_count:
                selected.add(idx - 1)
    return selected

# ৪. মিনিমাল সাইডবার
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-left">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#5E9E8B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
            <span>সেটিংস</span>
        </span>
        <span class="sb-engine-pill">STUDIO</span>
    </div>
    """, unsafe_allow_html=True)

    with st.popover("পেজ সেটআপ", use_container_width=True):
        PAGE_MODE = st.radio("মোড", ["অটো", "বই", "স্লাইড"], index=0, horizontal=True)
        COLUMNS_PER_PAGE = st.radio("কলাম", [1, 2], index=1, horizontal=True)
        PAGE_RANGE_TEXT = st.text_input("পৃষ্ঠা", "", placeholder="1-5 বা ফাঁকা")
        HEADER_TITLE = st.text_input("শিরোনাম", "MCQ Practice Sheet")
        INVERT_DARK_OUT = st.checkbox("সাদা ব্যাকগ্রাউন্ড", value=False)

    with st.expander("মার্জিন ও মাপ", expanded=False):
        QUESTION_GAP_PT = st.slider("গ্যাপ (pt)", 4, 40, 12, step=2)
        MIN_Q_HEIGHT_PT = st.slider("ন্যূনতম উচ্চতা", 10, 45, 16, step=2)
        RIGHT_EDGE_TRIM_PT = st.slider("ডান ছাঁটাই", 0, 35, 0, step=1)

    with st.expander("ব্যাচ এডিট", expanded=False):
        BATCH_TARGET = st.selectbox(
            "টার্গেট",
            ["সব বক্স", "নির্দিষ্ট বক্স", "বাম কলাম", "ডান কলাম", "নির্দিষ্ট পেজ"]
        )
        BATCH_BOX_NUMS = ""
        BATCH_PAGE_NO = 1
        if BATCH_TARGET == "নির্দিষ্ট বক্স":
            BATCH_BOX_NUMS = st.text_input("বক্স নং", "1-5", placeholder="1-4, 7")
        elif BATCH_TARGET == "নির্দিষ্ট পেজ":
            BATCH_PAGE_NO = st.number_input("পেজ নং", min_value=1, value=1, step=1)

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            BATCH_DX = st.number_input("ডানে/বামে (X)", value=0.0, step=2.0, format="%.1f")
            BATCH_DW = st.number_input("প্রস্থ (W)", value=0.0, step=2.0, format="%.1f")
        with b_col2:
            BATCH_DY = st.number_input("নিচে/উপরে (Y)", value=0.0, step=2.0, format="%.1f")
            BATCH_DH = st.number_input("উচ্চতা (H)", value=0.0, step=2.0, format="%.1f")

        APPLY_BATCH_BTN = st.button("প্রয়োগ")

    mode_short = "AUTO" if PAGE_MODE == "অটো" else ("BOOK" if PAGE_MODE == "বই" else "SLIDE")
    range_short = PAGE_RANGE_TEXT.strip() if PAGE_RANGE_TEXT.strip() else "ALL"

    st.markdown(f"""
    <div class="sb-summary-card">
        <div class="sb-summary-row"><span>মোড</span><span class="sb-summary-val">{mode_short} • {COLUMNS_PER_PAGE}C</span></div>
        <div class="sb-summary-row"><span>পৃষ্ঠা • গ্যাপ</span><span class="sb-summary-val">{range_short} • {QUESTION_GAP_PT}pt</span></div>
        <div class="sb-summary-row"><span>ট্যাগ</span><span class="sb-summary-val">{len(Q_TAGS)}Q • {len(SOL_TAGS)}S</span></div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# ৫. শক্তিশালী Sol^n ও সমাধান শনাক্তকরণ রেগুলার এক্সপ্রেশন
# ==============================================================================
SC = 2.5

q_prefix_pattern = "|".join(re.escape(t) for t in sorted(Q_TAGS, key=len, reverse=True))
sol_prefix_pattern = "|".join(re.escape(t) for t in sorted(SOL_TAGS, key=len, reverse=True))

NUM_RE = re.compile(
    rf"^[\(\[]?(?:(?:{q_prefix_pattern})\s*[\.\:\-]?\s*)?([0-9০-৯]{{1,3}})[\.\)\]\:](?![0-9০-৯])",
    re.IGNORECASE
)
PURE_PREFIX_RE = re.compile(rf"^(?:{q_prefix_pattern})[\.\:\-]?\s*[0-9০-৯]*$", re.IGNORECASE)
OPT_TOKEN_RE = re.compile(r"^(?:[\(\[]?[a-dA-Dক-ঘ][\)\]\.]|[ⓐⓑⓒⓓⓔ①②③④])$")
BULLET_STOP_RE = re.compile(r"^[➢➤➣•♦►]\s*")
OPT_UNIT_RE = re.compile(r"^(?:mol|L|g|kg|mL|ml|cm|mm|m|gm|sec|min|s|×|x|10\^?)$", re.IGNORECASE)

SOL_WORD_CORE = (
    r"(?:[sS58\$][oO0aAeE][lL1Ii\|!t](?:[nNmMrRuU23\^\°\*\~\"\'\`\’\”\ⁿ\-\_]+)?|"
    r"solution|ans(?:wer)?|shortcut|hints?|note|"
    r"উত্তর|উত্ত|উওর|উত্তব|উত্তন|উত্তম|ডত্তর|ডওর|ডত্তব|ত্তর|ত্তব|উঃ|উ:|ত্তঃ|ত্ত:|"
    r"সমাধান|সমাধা|মাধান|সলভ|সল্‌ভ|"
    r"ব্যাখ্যা|ব্যখ্যা|ব্যাখা|ব্যখা|্যাখ্যা|াখ্যা|রাখ্যা|য্যাখ্যা|বাখ্যা|ব্যাখ|"
    r"বিকল্প|িকল্প|শর্টকাট|"
    rf"{sol_prefix_pattern})"
)

SOL_TOKEN_RE = re.compile(
    rf"^[\(\[\{{•\*➢➤➣♦►\-\—\–\.\:\;\,\'\"\`\~\|]*{SOL_WORD_CORE}(?:$|[\s\:\ঃ\.\;\,\-\—\–\(\[\{{\"'\`\’\”\ⁿ\^\°\*\/\|0-9০-৯])",
    re.IGNORECASE
)

INLINE_SOL_RE = re.compile(
    rf"(?:^|\s|[\]\)\।\|]){SOL_WORD_CORE}(?:\s*[\:\ঃ\-\—\–\.\(\[]|\s+[nNⁿ\^]\b)",
    re.IGNORECASE
)

HEADER_NOISE_RE = re.compile(
    r"(?:Educationalpdf|Educationaf|Rhombus\s*Publications|পরিমাণগত\s*রসায়ন|ACS\s*Varsity|ACS\s*Chemistry|ACS\s*Cliemistry)",
    re.IGNORECASE
)

def _b2e_int(s):
    t = s.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")).strip()
    return int(t) if t.isdigit() else None

def is_solution_text(text):
    if not text:
        return False
    clean = re.sub(r"[\u200c\u200d\ufeff]", "", text).strip()
    if not clean:
        return False
    if SOL_TOKEN_RE.match(clean):
        return True
    stripped = clean.lstrip("([{•*-➢➤.@©\"'` ").rstrip(".:;ঃ-ⁿ^°*~\"'`)]} ")
    if not stripped:
        return False
    if SOL_TOKEN_RE.match(stripped):
        return True
    s_low = stripped.lower()
    head_part = re.split(r"[\:\ঃ\(\[\.\-]", s_low)[0].strip()
    for tag in SOL_TAGS:
        t_low = tag.lower().strip()
        if not t_low:
            continue
        if s_low == t_low or s_low.startswith(t_low + ":") or s_low.startswith(t_low + "ঃ") or (head_part and head_part == t_low):
            return True
        if len(t_low) >= 4 and s_low.startswith(t_low):
            return True
    return False

def line_has_solution_marker(line, joined_text, max_sol_x):
    if not line:
        return False, None
    wx0, wy0, _, _, t0 = line[0]

    if wx0 <= max_sol_x and is_solution_text(t0):
        return True, wy0

    if len(line) >= 2 and wx0 <= max_sol_x:
        t1 = line[1][4].strip()
        combined_01 = f"{t0}{t1}"
        if is_solution_text(combined_01) or (re.match(r"^[sS58\$][oO0aAeE][lL1Ii\|!t]$", t0.strip("([. ")) and re.match(r"^[nNmMrR23\^\°\*\ⁿ\:\.\-]+$", t1)):
            return True, wy0

    for idx_w, w in enumerate(line[:6]):
        if w[0] <= max_sol_x:
            if is_solution_text(w[4]):
                return True, w[1]
            if idx_w + 1 < len(line):
                comb = f"{w[4]}{line[idx_w + 1][4]}"
                if is_solution_text(comb):
                    return True, w[1]

    if INLINE_SOL_RE.search(joined_text):
        return True, wy0

    return False, None

# ==============================================================================
# ৬. ইমেজ প্রসেসিং ও বাঁকা বিভাজক রেখা বিশ্লেষণ
# ==============================================================================
def parse_page_range(range_str, total_pages):
    range_str = range_str.strip()
    if not range_str:
        return list(range(total_pages))
    selected = set()
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            b = part.split("-", 1)
            if b[0].strip().isdigit() and b[1].strip().isdigit():
                for p in range(max(1, int(b[0])), min(total_pages, int(b[1])) + 1):
                    selected.add(p - 1)
        elif part.isdigit():
            p = int(part)
            if 1 <= p <= total_pages:
                selected.add(p - 1)
    return sorted(selected) if selected else list(range(total_pages))

def page_to_gray(page, scale=SC):
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY)
    g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    if g.mean() < 110:
        g = 255 - g
    return g

def clip_is_dark(page, clip):
    pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), clip=clip, colorspace=fitz.csGRAY, alpha=False)
    return float(np.frombuffer(pix.samples, np.uint8).mean()) < 110

def normalize_shadow_gray(gray_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bg = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
    return cv2.divide(gray_img, bg, scale=255)

def get_ink_mask(gray_img):
    norm = normalize_shadow_gray(gray_img)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 14)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return clean > 0

def find_divider(gray_img, W, H):
    x_start, x_end = int(W * 0.36 * SC), int(W * 0.64 * SC)
    y_start, y_end = int(H * 0.10 * SC), int(H * 0.90 * SC)
    ink_full = get_ink_mask(gray_img).astype(np.uint8)
    roi = cv2.dilate(ink_full[y_start:y_end, x_start:x_end], np.ones((1, 3), np.uint8), iterations=1)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(32 * SC)))
    v_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, v_kernel)
    col_sum = v_lines.sum(axis=0)
    if col_sum.max() > int(35 * SC):
        return (x_start + int(np.argmax(col_sum))) / SC, True
    ink_sum = roi.sum(axis=0)
    row_cover = float((roi.sum(axis=1) > 0).mean())
    if ink_sum.min() == 0 and row_cover >= 0.30:
        zero = (ink_sum == 0).astype(np.int8)
        best_len, best_start, run_start = 0, 0, None
        for i, v in enumerate(list(zero) + [0]):
            if v and run_start is None:
                run_start = i
            elif not v and run_start is not None:
                if i - run_start > best_len:
                    best_len, best_start = i - run_start, run_start
                run_start = None
        if best_len >= int(8 * SC):
            return (x_start + best_start + best_len / 2.0) / SC, True
    return W / 2.0, False

def trace_slanted_divider(gray_img, W, H):
    base_div, is_two_col = find_divider(gray_img, W, H)
    if not is_two_col:
        return base_div, False, 0.0, lambda _y: base_div

    x_start, x_end = int(W * 0.35 * SC), int(W * 0.65 * SC)
    blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    roi = thresh[:, x_start:x_end]

    h_px = gray_img.shape[0]
    pts_y, pts_x = [], []
    for y0 in range(int(0.06 * h_px), int(0.92 * h_px), int(25 * SC)):
        y1 = min(h_px, y0 + int(40 * SC))
        col_proj = roi[y0:y1, :].mean(axis=0)
        best_c = int(np.argmax(col_proj))
        if col_proj[best_c] > 105:
            pts_y.append((y0 + y1) * 0.5 / SC)
            pts_x.append((x_start + best_c) / SC)

    if len(pts_x) >= 3:
        med_x = float(np.median(pts_x))
        inliers = [i for i in range(len(pts_x)) if abs(pts_x[i] - med_x) < 20.0]
        if len(inliers) >= 2:
            ys = np.array([pts_y[i] for i in inliers])
            xs = np.array([pts_x[i] for i in inliers])
            slope, intercept = np.polyfit(ys, xs, 1)
            slope = float(np.clip(slope, -0.04, 0.04))
            return med_x, True, slope, lambda y: float(slope * float(y) + intercept)

    return base_div, True, 0.0, lambda _y: base_div

def detect_horizontal_barriers(gray_img, x0, x1):
    x0_px, x1_px = max(0, int(x0 * SC)), min(gray_img.shape[1], int(x1 * SC))
    if x1_px - x0_px < 40:
        return []
    roi = get_ink_mask(gray_img[:, x0_px:x1_px]).astype(np.uint8)
    min_w = max(30, int((x1 - x0) * 0.55 * SC))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_w, 1))
    h_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, h_kernel)
    thick_blobs = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6)))
    thin_lines = h_lines & (~thick_blobs)
    y_coords = np.where(thin_lines.sum(axis=1) > (min_w * 0.85))[0] / SC
    merged = []
    for y in y_coords:
        if not merged or y - merged[-1] > 6.0:
            merged.append(float(y))
    return merged

def snap_cut_line(ink, xr, y_target, up_win=2.5, down_win=1.0):
    h_px, w_px = ink.shape
    x0, x1 = max(0, int(xr[0] * SC)), min(w_px, int((xr[1] - 8.0) * SC))
    a = max(0, int((y_target - up_win) * SC))
    b = min(h_px, int((y_target + down_win) * SC))
    if b <= a or x1 <= x0:
        return y_target
    prof = ink[a:b, x0:x1].mean(axis=1)
    best_row = int(np.argmin(prof))
    return (a + best_row) / SC

def trim_vertical_whitespace(ink, xr, top, bot):
    h_px, w_px = ink.shape
    x0, x1 = max(0, int(xr[0] * SC)), min(w_px, int((xr[1] - 8.0) * SC))
    y0, y1 = max(0, int(top * SC)), min(h_px, int(bot * SC))
    if y1 <= y0 or x1 <= x0:
        return top, bot
    row_ink = ink[y0:y1, x0:x1].mean(axis=1) > 0.001
    nz = np.where(row_ink)[0]
    if len(nz) == 0:
        return top, bot
    return max(top, (y0 + nz[0]) / SC - 3.5), min(bot, (y0 + nz[-1]) / SC + 5.0)

# ==============================================================================
# ৭. ৩-পাস শক্তিশালী OCR ইঞ্জিন
# ==============================================================================
def group_words_into_lines(words):
    if not words:
        return []
    sorted_w = sorted(words, key=lambda w: ((w[1] + w[3]) * 0.5, w[0]))
    lines = []
    cur_line = [sorted_w[0]]
    cur_cy = (sorted_w[0][1] + sorted_w[0][3]) * 0.5
    for w in sorted_w[1:]:
        w_cy = (w[1] + w[3]) * 0.5
        if abs(w_cy - cur_cy) <= 6.4:
            cur_line.append(w)
            cur_cy = sum((x[1] + x[3]) * 0.5 for x in cur_line) / len(cur_line)
        else:
            cur_line.sort(key=lambda x: x[0])
            lines.append(cur_line)
            cur_line = [w]
            cur_cy = w_cy
    cur_line.sort(key=lambda x: x[0])
    lines.append(cur_line)
    return lines

def get_column_lines(page, gray_img, x0, x1, H, col_i=0, is_two_col=True, div_fn=None):
    pdf_words = []
    for wx0, wy0, wx1, wy1, text, *_ in page.get_text("words"):
        if x0 - 4 <= wx0 < x1 - 4 and 12 < wy0 < H - 18:
            pdf_words.append((wx0, wy0, wx1, wy1, text.strip()))

    pdf_lines = group_words_into_lines(pdf_words)
    sol_in_pdf = sum(1 for l in pdf_lines if line_has_solution_marker(l, " ".join(w[4] for w in l), x0 + 120.0)[0])
    q_in_pdf = sum(1 for w in pdf_words if NUM_RE.match(w[4]) or PURE_PREFIX_RE.match(w[4]))

    ocr_words = []
    ocr_thin_words = []
    ocr_eng_words = []

    if q_in_pdf < 3 or sol_in_pdf < max(1, q_in_pdf):
        x0_px, x1_px = max(0, int(x0 * SC)), min(gray_img.shape[1], int(x1 * SC))
        roi_gray = normalize_shadow_gray(gray_img[:, x0_px:x1_px].copy())

        if is_two_col and div_fn is not None:
            for r_px in range(roi_gray.shape[0]):
                d_px = int(div_fn(r_px / SC) * SC) - x0_px
                if col_i == 1 and 0 <= d_px + 4 < roi_gray.shape[1]:
                    roi_gray[r_px, :max(0, d_px + 4)] = 255
                elif col_i == 0 and 0 <= d_px - 4 < roi_gray.shape[1]:
                    roi_gray[r_px, max(0, d_px - 4):] = 255

        if (not is_two_col or col_i == 1) and roi_gray.shape[1] > int(25 * SC):
            roi_gray[:, -int(12 * SC):] = 255

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_gray)
        clean_bw = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 14)
        base_x = x0_px / SC

        ocr_data = pytesseract.image_to_data(
            clean_bw, lang="eng+ben", config="--psm 6", output_type=pytesseract.Output.DICT
        )
        for i in range(len(ocr_data["text"])):
            txt = ocr_data["text"][i].strip()
            conf = int(ocr_data["conf"][i]) if str(ocr_data["conf"][i]).lstrip("-").isdigit() else -1
            if not txt or conf < 16:
                continue
            wx0 = base_x + ocr_data["left"][i] / SC
            wy0 = ocr_data["top"][i] / SC
            wx1 = wx0 + ocr_data["width"][i] / SC
            wy1 = wy0 + ocr_data["height"][i] / SC
            if 12 < wy0 < H - 18:
                ocr_words.append((wx0, wy0, wx1, wy1, txt))

        thin_bw = cv2.dilate(clean_bw, np.ones((2, 2), np.uint8), iterations=1)
        ocr_data2 = pytesseract.image_to_data(
            thin_bw, lang="ben+eng", config="--psm 6", output_type=pytesseract.Output.DICT
        )
        for i in range(len(ocr_data2["text"])):
            txt = ocr_data2["text"][i].strip()
            conf = int(ocr_data2["conf"][i]) if str(ocr_data2["conf"][i]).lstrip("-").isdigit() else -1
            if not txt or conf < 16:
                continue
            wx0 = base_x + ocr_data2["left"][i] / SC
            wy0 = ocr_data2["top"][i] / SC
            wx1 = wx0 + ocr_data2["width"][i] / SC
            wy1 = wy0 + ocr_data2["height"][i] / SC
            if 12 < wy0 < H - 18:
                ocr_thin_words.append((wx0, wy0, wx1, wy1, txt))

        left_strip_w = min(clean_bw.shape[1], int(105 * SC))
        eng_data = pytesseract.image_to_data(
            clean_bw[:, :left_strip_w], lang="eng", config="--psm 6", output_type=pytesseract.Output.DICT
        )
        for i in range(len(eng_data["text"])):
            txt = eng_data["text"][i].strip()
            conf = int(eng_data["conf"][i]) if str(eng_data["conf"][i]).lstrip("-").isdigit() else -1
            if not txt or conf < 15:
                continue
            wx0 = base_x + eng_data["left"][i] / SC
            wy0 = eng_data["top"][i] / SC
            wx1 = wx0 + eng_data["width"][i] / SC
            wy1 = wy0 + eng_data["height"][i] / SC
            if 12 < wy0 < H - 18:
                ocr_eng_words.append((wx0, wy0, wx1, wy1, txt))

    return (
        pdf_lines,
        group_words_into_lines(ocr_words),
        group_words_into_lines(ocr_thin_words),
        group_words_into_lines(ocr_eng_words),
    )

def analyze_lines_for_markers(lines, actual_left_x, div_slope=0.0, H=842.0):
    starts, stops = [], []
    if not lines:
        return starts, stops

    norm_x0s = [line[0][0] - div_slope * (line[0][1] - H * 0.5) for line in lines if line and len(line[0][4]) >= 2]
    text_left_x = float(np.percentile(norm_x0s, 15)) if norm_x0s else actual_left_x

    for line in lines:
        first_w = line[0]
        wx0, wy0, wx1, wy1, t0 = first_w
        joined_text = " ".join(w[4] for w in line).strip()
        if not joined_text or HEADER_NOISE_RE.search(joined_text):
            continue

        local_shift = div_slope * (wy0 - H * 0.5)
        max_q_x = max(actual_left_x + 52.0, text_left_x + local_shift + 42.0)
        max_sol_x = max(actual_left_x + 120.0, text_left_x + local_shift + 108.0)

        if BULLET_STOP_RE.match(t0) or BULLET_STOP_RE.match(joined_text):
            stops.append(wy0 - 1.0)
            continue

        has_sol, sol_y = line_has_solution_marker(line, joined_text, max_sol_x)
        if has_sol:
            stops.append((sol_y if sol_y is not None else wy0) - 1.2)
            continue

        if ("=" in joined_text or "⇒" in joined_text or "∴" in joined_text) and "?" not in joined_text:
            if not re.search(r"(?:কত|কোনটি|হলে|যদি|নির্ণয়|নির্ণয়|সমান|অর্ধেক|দ্বিগুণ)", joined_text):
                stops.append(wy0 - 1.0)
                continue

        if OPT_TOKEN_RE.match(t0):
            continue

        cand_indices = [0]
        if len(line) > 1 and len(t0) <= 2 and re.match(r"^[\|\!\.\,\;\:\'\"\`\-\—\_\/\\]+$", t0):
            cand_indices.append(1)

        for idx_c in cand_indices:
            cw = line[idx_c]
            c_wx0, c_wy0, c_wx1, c_wy1, c_t0 = cw
            if c_wx0 > max_q_x:
                continue

            is_num_q = bool(NUM_RE.match(c_t0) or PURE_PREFIX_RE.match(c_t0))
            is_dotless_q = False
            if not is_num_q and re.match(r"^[0-9০-৯]{1,2}$", c_t0):
                if "?" in joined_text or re.search(r"(?:কত|কোনটি|হচ্ছে|থাকে|থাকবে|রয়েছে|রয়েছে|আছে|ওজন|ভর|সংখ্যা|আয়তন|আয়তন|মোল|লিটার|গ্রাম|গ্যাস|পরমাণু|অণু|STP|NTP)\b", joined_text):
                    is_dotless_q = True

            if is_num_q or is_dotless_q:
                if re.match(r"^[0-9০-৯]+\.[0-9০-৯]+", c_t0):
                    continue
                m_num = NUM_RE.match(c_t0)
                qn = _b2e_int(m_num.group(1)) if (m_num and m_num.group(1)) else (_b2e_int(c_t0) if is_dotless_q else None)
                if qn is not None and (qn == 0 or qn > 160):
                    continue

                rest_text = " ".join(w[4] for w in line[idx_c + 1:]).strip()
                if not rest_text:
                    continue

                if ("=" in rest_text or "⇒" in rest_text or "∴" in rest_text) and "?" not in rest_text:
                    if not re.search(r"(?:কত|কোনটি|হলে|যদি)", rest_text):
                        continue

                has_q_words = bool(re.search(r"[অ-ঔক-হ]{2,}|[a-zA-Z]{3,}|\?", rest_text))
                if not has_q_words and len(line) > idx_c + 1:
                    nxt_w = line[idx_c + 1]
                    if OPT_UNIT_RE.match(nxt_w[4]) or (nxt_w[0] - c_wx1 < 3.5 and re.match(r"^[0-9০-৯]", nxt_w[4])):
                        continue
                starts.append(c_wy0 - 1.5)
                break

    return starts, stops

def is_valid_question_crop(lines_in_crop):
    if not lines_in_crop:
        return True
    full_txt = " ".join(" ".join(w[4] for w in l) for l in lines_in_crop).strip()
    if not full_txt:
        return False

    first_line = " ".join(w[4] for w in lines_in_crop[0]).strip()
    if re.match(r"^(?:[\=\⇒\∴\∵]|\.\.\s*[0-9০-৯]|V_)", first_line) or is_solution_text(first_line):
        return False

    has_q_mark = "?" in full_txt
    has_options = bool(re.search(r"(?:[\(\[]?[ক-ঘa-dA-D][\)\]\.]|[ⓐⓑⓒⓓ])", full_txt))
    has_q_words = bool(re.search(r"(?:কত|কোনটি|কোন|হচ্ছে|থাকে|থাকবে|রয়েছে|রয়েছে|আছে|নির্ণয়|নির্ণয়|ওজন|ভর|সংখ্যা|আয়তন|আয়তন|ঘনমাত্রা|মোল|লিটার|গ্যাস|পরমাণু|অণু|বিক্রিয়া|বিক্রিয়া|লেখচিত্র|what|which|find|calculate|value)", full_txt, re.IGNORECASE))
    eq_count = full_txt.count("=") + full_txt.count("⇒")

    if eq_count >= 1 and not has_q_mark and not has_options and not has_q_words:
        return False
    if eq_count >= 2 and not has_q_mark and not has_options:
        return False
    return True

# ==============================================================================
# ৮. স্লাইড মোড: বক্সে ঘেরা প্রশ্ন শনাক্ত ও Sol^n ছাঁটাই
# ==============================================================================
def is_slide_page(W, H):
    if PAGE_MODE == "বই":
        return False
    if PAGE_MODE == "স্লাইড":
        return True
    return W > 1.2 * H

def _order_rects_reading(rects):
    rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    rows, out = [], []
    for r in rects:
        if rows and abs(r.y0 - rows[-1][0].y0) < 30:
            rows[-1].append(r)
        else:
            rows.append([r])
    for row in rows:
        out.extend(sorted(row, key=lambda r: r.x0))
    return out

def box_has_question_number(page, gray, rect):
    try:
        words = [w for w in page.get_text("words")
                 if rect.x0 - 2 <= w[0] <= rect.x1 and rect.y0 - 2 <= w[1] <= rect.y1 and w[4].strip()]
        if words:
            top_y = min(w[1] for w in words)
            first_line = sorted([w for w in words if w[1] <= top_y + 12], key=lambda w: w[0])
            return any(NUM_RE.match(w[4].strip()) or PURE_PREFIX_RE.match(w[4].strip()) for w in first_line[:2])
        x0, y0 = max(0, int(rect.x0 * SC)), max(0, int(rect.y0 * SC))
        x1, y1 = int(rect.x1 * SC), int(min(rect.y1, rect.y0 + 45) * SC)
        txt = pytesseract.image_to_string(gray[y0:y1, x0:x1], lang="eng", config="--psm 7").strip()
        return bool(re.match(r"^\W{0,3}[\(\[]?\s*[0-9]{1,3}\s*[\)\]\.]", txt))
    except Exception:
        return False

def trim_solution_inside_slide_box(page, gray, ink, rect):
    try:
        x0_px, y0_px = max(0, int(rect.x0 * SC)), max(0, int(rect.y0 * SC))
        x1_px, y1_px = min(gray.shape[1], int(rect.x1 * SC)), min(gray.shape[0], int(rect.y1 * SC))
        if y1_px - y0_px < 40 or x1_px - x0_px < 60:
            return rect
        roi = gray[y0_px:y1_px, x0_px:x1_px]
        bw = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 13)
        data = pytesseract.image_to_data(bw, lang="eng+ben", config="--psm 6", output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data["text"])):
            t = data["text"][i].strip()
            if not t:
                continue
            wx0 = rect.x0 + data["left"][i] / SC
            wy0 = rect.y0 + data["top"][i] / SC
            wx1 = wx0 + data["width"][i] / SC
            wy1 = wy0 + data["height"][i] / SC
            words.append((wx0, wy0, wx1, wy1, t))
        lines = group_words_into_lines(words)
        for ln in lines:
            joined = " ".join(w[4] for w in ln)
            has_s, sy = line_has_solution_marker(ln, joined, rect.x0 + (rect.width * 0.55))
            if has_s and sy is not None and (sy - rect.y0) >= 18.0:
                new_bot = snap_cut_line(ink, (rect.x0 + 4, rect.x1 - 4), sy - 1.5, up_win=2.5, down_win=0.5)
                return fitz.Rect(rect.x0, rect.y0, rect.x1, max(rect.y0 + 18.0, new_bot))
    except Exception:
        pass
    return rect

def detect_boxed_questions(page, gray, ink, W, H):
    k = int(110 * SC)
    m = ink.astype(np.uint8)
    h_lines = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(h_lines, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        if w >= k and h <= int(9 * SC):
            segs.append((x / SC, (y + h / 2.0) / SC, (x + w) / SC))
    segs.sort(key=lambda t: t[1])

    def side_cover(xc_lo, xc_hi, yt, yb):
        a, b = max(0, int(xc_lo * SC)), min(ink.shape[1], int(xc_hi * SC))
        r0, r1 = int((yt + 8) * SC), int((yb - 8) * SC)
        if b <= a or r1 <= r0:
            return 0.0, None
        win = ink[r0:r1, a:b]
        rows_hit = win.any(axis=1).mean()
        col = win.sum(axis=0)
        return float(rows_hit), (a + int(np.argmax(col))) / SC

    boxes = []
    used_bottom = set()
    for i, (ax0, ay, ax1) in enumerate(segs):
        for j in range(i + 1, len(segs)):
            bx0, by, bx1 = segs[j]
            dy = by - ay
            if dy < 24:
                continue
            if dy > 0.7 * H:
                break
            if abs(ax0 - bx0) > 14 or abs(ax1 - bx1) > 14:
                continue
            lcov, lx = side_cover(ax0 - 22, ax0 + 3, ay, by)
            rcov, rx = side_cover(ax1 - 3, ax1 + 22, ay, by)
            if lcov < 0.8 or rcov < 0.8 or j in used_bottom:
                continue
            rect = fitz.Rect(max(2.0, lx - 4), max(2.0, ay - 4), min(W - 2.0, rx + 4), min(H - 2.0, by + 4))
            if any((rect & r).get_area() > 0.8 * min(rect.get_area(), r.get_area()) for r in [b[0] for b in boxes]):
                break
            if box_has_question_number(page, gray, rect):
                trimmed_rect = trim_solution_inside_slide_box(page, gray, ink, rect)
                boxes.append((trimmed_rect,))
                used_bottom.add(j)
            break
    return _order_rects_reading([b[0] for b in boxes])

# ৯. প্রোগ্রেস কার্ড ও মূল এক্সট্রাকশন পাইপলাইন
def render_progress_card(placeholder, current_idx, total_pages, page_no, found_count, stage_label):
    pct = int(round((current_idx / max(1, total_pages)) * 100))
    placeholder.markdown(f"""
    <div class="progress-studio-card">
        <div class="ps-top-row">
            <div class="ps-status-text">
                <span class="ps-dot"></span>
                <span>{stage_label}</span>
            </div>
            <span class="ps-counter">{current_idx:02d}/{total_pages:02d} • {pct}%</span>
        </div>
        <div class="ps-track">
            <div class="ps-fill" style="width: {pct}%;"></div>
        </div>
        <div class="ps-sub-row">
            <span>PAGE #{page_no}</span>
            <span>FOUND: {found_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def extract_questions_from_pdf(src_doc, page_indices, progress_slot):
    questions = []
    total_to_scan = len(page_indices)

    for idx, pno in enumerate(page_indices):
        render_progress_card(
            progress_slot, idx, total_to_scan, pno + 1, len(questions),
            f"পৃষ্ঠা {pno + 1} স্ক্যান চলছে..."
        )

        page = src_doc[pno]
        W, H = page.rect.width, page.rect.height
        gray = page_to_gray(page, SC)
        ink = get_ink_mask(gray)

        if is_slide_page(W, H):
            boxed = detect_boxed_questions(page, gray, ink, W, H)
            if boxed:
                for r in boxed:
                    questions.append((pno, r))
                render_progress_card(
                    progress_slot, idx + 1, total_to_scan, pno + 1, len(questions),
                    f"পৃষ্ঠা {pno + 1} সম্পন্ন"
                )
                continue

        div_x, is_two_col, div_slope, div_fn = trace_slanted_divider(gray, W, H)

        if is_two_col:
            for r_px in range(ink.shape[0]):
                xd_px = int(div_fn(r_px / SC) * SC)
                ink[r_px, max(0, xd_px - int(3.5 * SC)):min(ink.shape[1], xd_px + int(3.5 * SC))] = False

        col_ranges = [(12.0, div_x - 2.0), (div_x + 2.0, W - 12.0)] if is_two_col else [(12.0, W - 12.0)]

        for col_i, (cx0, cx1) in enumerate(col_ranges):
            inner_x0 = cx0 + (6.0 if col_i == 0 else 4.0)
            inner_x1 = cx1 - (4.0 if (is_two_col and col_i == 0) else 10.0)
            ink_band = ink[int(38 * SC):int((H - 30) * SC), int(inner_x0 * SC):int(inner_x1 * SC)].copy()
            v_bar = cv2.morphologyEx(ink_band.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(25 * SC))))
            ink_band_clean = ink_band & (~(v_bar > 0))
            col_x_profile = np.where(ink_band_clean.mean(axis=0) > 0.003)[0]
            actual_left_x = inner_x0 + (col_x_profile[0] / SC if len(col_x_profile) > 0 else 0)
            actual_right_x = inner_x0 + (col_x_profile[-1] / SC if len(col_x_profile) > 0 else (inner_x1 - inner_x0))

            scan_x0 = min(cx0, div_fn(20.0) + 1.5, div_fn(H - 20.0) + 1.5) if (is_two_col and col_i == 1) else cx0
            scan_x1 = max(cx1, div_fn(20.0) - 1.5, div_fn(H - 20.0) - 1.5) if (is_two_col and col_i == 0) else cx1

            pdf_lines, ocr_lines, ocr_thin_lines, ocr_eng_lines = get_column_lines(
                page, gray, scan_x0, scan_x1, H, col_i=col_i, is_two_col=is_two_col, div_fn=div_fn
            )
            h_barriers = detect_horizontal_barriers(gray, cx0, cx1)

            p_starts, p_stops = analyze_lines_for_markers(pdf_lines, actual_left_x, div_slope, H)
            o_starts, o_stops = analyze_lines_for_markers(ocr_lines, actual_left_x, div_slope, H)
            t_starts, t_stops = analyze_lines_for_markers(ocr_thin_lines, actual_left_x, div_slope, H)
            e_starts, e_stops = analyze_lines_for_markers(ocr_eng_lines, actual_left_x, div_slope, H)

            all_starts = sorted(p_starts + o_starts + t_starts + e_starts)
            all_stops = sorted(p_stops + o_stops + t_stops + e_stops + list(h_barriers))

            clean_starts = []
            for st_y in all_starts:
                if not clean_starts or st_y - clean_starts[-1] > 16.0:
                    clean_starts.append(st_y)

            ref_lines = ocr_lines if len(ocr_lines) >= len(pdf_lines) else pdf_lines

            for qi, st_y in enumerate(clean_starts):
                nxt_q = clean_starts[qi + 1] if qi + 1 < len(clean_starts) else (H - 16.0)
                cand_stops = [y for y in all_stops if st_y + 11.0 < y < nxt_q]
                end_y = min(cand_stops) if cand_stops else nxt_q

                crop_lines = [l for l in ref_lines if l and (st_y - 2.0 <= l[0][1] <= end_y + 2.0)]
                if not is_valid_question_crop(crop_lines):
                    continue

                div_at_top = div_fn(st_y)
                div_at_bot = div_fn(end_y)
                if is_two_col and col_i == 0:
                    q_x0 = max(8.0, cx0 - 2.0, min(actual_left_x - 6.0, 22.0))
                    q_x1 = min(div_at_top, div_at_bot) - 2.0 - RIGHT_EDGE_TRIM_PT
                elif is_two_col and col_i == 1:
                    q_x0 = max(div_at_top, div_at_bot) + 1.5
                    q_x1 = min(W - 8.0, actual_right_x + 6.0) - RIGHT_EDGE_TRIM_PT
                else:
                    q_x0 = max(8.0, cx0 - 2.0, actual_left_x - 6.0)
                    q_x1 = min(cx1, actual_right_x + 6.0) - RIGHT_EDGE_TRIM_PT

                if q_x1 - q_x0 < 75:
                    continue

                top = snap_cut_line(ink, (q_x0, q_x1), st_y, up_win=3.0, down_win=1.0)
                bot = snap_cut_line(ink, (q_x0, q_x1), end_y, up_win=1.8, down_win=0.8)
                top, bot = trim_vertical_whitespace(ink, (q_x0, q_x1), top, bot)

                if bot - top >= float(MIN_Q_HEIGHT_PT) and q_x1 - q_x0 >= 75.0:
                    box_ink = ink[int(top * SC):int(bot * SC), int(q_x0 * SC):int(q_x1 * SC)]
                    if box_ink.size > 0 and box_ink.mean() > 0.004:
                        rect = fitz.Rect(q_x0, top, q_x1, bot)
                        if not rect.is_empty and not rect.is_infinite:
                            questions.append((pno, rect))

        render_progress_card(
            progress_slot, idx + 1, total_to_scan, pno + 1, len(questions),
            f"পৃষ্ঠা {pno + 1} সম্পন্ন"
        )

    return questions

def build_practice_pdf(src_doc, questions):
    out = fitz.open()
    PW, PH = fitz.paper_size("a4")
    MARGIN_X, MARGIN_TOP, MARGIN_BOT, COL_GAP = 24.0, 50.0, 26.0, 18.0
    num_cols = int(COLUMNS_PER_PAGE)
    col_width = (PW - 2 * MARGIN_X - (num_cols - 1) * COL_GAP) / num_cols

    widths = sorted(c.width for _, c in questions)
    wide_set = widths[len(widths) // 2] > 1.5 * col_width
    common_scale = None
    if wide_set:
        num_cols = 1
        col_width = PW - 2 * MARGIN_X
        common_scale = min(1.65, col_width / max(widths))

    def create_styled_page():
        p = out.new_page(width=PW, height=PH)
        if HEADER_TITLE.strip():
            p.insert_text(fitz.Point(MARGIN_X, 30), HEADER_TITLE.strip(), fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))
            p.insert_text(fitz.Point(PW - MARGIN_X - 75, 30), f"Total Q: {len(questions)}", fontsize=10, fontname="helv", color=(0.2, 0.2, 0.2))
            p.draw_line(fitz.Point(MARGIN_X, 37), fitz.Point(PW - MARGIN_X, 37), color=(0.2, 0.2, 0.2), width=0.8)
        if num_cols == 2:
            mid_x = PW / 2.0
            p.draw_line(fitz.Point(mid_x, MARGIN_TOP - 6), fitz.Point(mid_x, PH - MARGIN_BOT), color=(0.3, 0.3, 0.3), width=0.6)
        return p

    page = create_styled_page()
    cur_col, y_cursor = 0, MARGIN_TOP

    for pno, clip in questions:
        scale = common_scale if common_scale else min(col_width / clip.width, 1.35 if num_cols == 2 else 1.65)
        target_w, target_h = clip.width * scale, clip.height * scale

        if y_cursor + target_h > PH - MARGIN_BOT:
            cur_col += 1
            if cur_col >= num_cols:
                page = create_styled_page()
                cur_col = 0
            y_cursor = MARGIN_TOP

        x_start = MARGIN_X + cur_col * (col_width + COL_GAP)
        target_rect = fitz.Rect(x_start, y_cursor, x_start + target_w, y_cursor + target_h)
        if INVERT_DARK_OUT and clip_is_dark(src_doc[pno], clip):
            pix = src_doc[pno].get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip, alpha=False)
            pix.invert_irect(pix.irect)
            page.insert_image(target_rect, pixmap=pix)
        else:
            page.show_pdf_page(target_rect, src_doc, pno, clip=clip)
        y_cursor += target_h + QUESTION_GAP_PT

    pdf_bytes = out.tobytes(garbage=4, deflate=True)
    out.close()
    return pdf_bytes


# ==============================================================================
# ৯.৯ বাহ্যিক canvas_editor.html লোডার ও বক্স সিঙ্ক ইঞ্জিন
# ==============================================================================
def questions_to_rows(questions):
    rows = []
    for pno, r in questions:
        rows.append({
            "keep": True,
            "page": pno + 1,
            "x0": round(float(r.x0), 1),
            "y0": round(float(r.y0), 1),
            "x1": round(float(r.x1), 1),
            "y1": round(float(r.y1), 1),
        })
    return pd.DataFrame(rows, columns=["keep", "page", "x0", "y0", "x1", "y1"])

def rows_to_questions(df, src_doc):
    per_page = {}
    for _, r in df.iterrows():
        try:
            if not bool(r["keep"]):
                continue
            pno = int(r["page"]) - 1
            if not (0 <= pno < len(src_doc)):
                continue
            W, H = src_doc[pno].rect.width, src_doc[pno].rect.height
            y0, y1 = float(r["y0"]), float(r["y1"])
            if pd.isna(y0) or pd.isna(y1):
                continue
            x0 = 14.0 if pd.isna(r["x0"]) else float(r["x0"])
            x1 = W - 14.0 if pd.isna(r["x1"]) else float(r["x1"])
            rect = fitz.Rect(max(0.0, min(x0, x1)), max(0.0, min(y0, y1)), min(W, max(x0, x1)), min(H, max(y0, y1)))
            if rect.width < 18 or rect.height < 6:
                continue
            per_page.setdefault(pno, []).append(rect)
        except Exception:
            continue
    out = []
    for pno in sorted(per_page):
        W, H = src_doc[pno].rect.width, src_doc[pno].rect.height
        rects = per_page[pno]
        if W > 1.2 * H:
            rects = _order_rects_reading(rects)
        else:
            rects = sorted(rects, key=lambda q: (0 if (q.width > 0.6 * W or (q.x0 + q.x1) / 2 < W / 2) else 1, q.y0))
        out.extend((pno, q) for q in rects)
    return out

def apply_sidebar_batch_transform(df, raw_pdf, target_mode, box_nums_str, target_page, dx, dy, dw, dh):
    if df is None or df.empty:
        return df
    doc = fitz.open(stream=raw_pdf, filetype="pdf")
    ordered_qs = rows_to_questions(df, doc)
    ordered_df = questions_to_rows(ordered_qs)
    total_q = len(ordered_df)

    if target_mode == "নির্দিষ্ট বক্স":
        target_indices = parse_box_selection_string(box_nums_str, total_q)
    else:
        target_indices = set(range(total_q))

    updated_rows = []
    for idx, row in ordered_df.iterrows():
        pno = int(row["page"]) - 1
        W = doc[pno].rect.width if 0 <= pno < len(doc) else 595.0
        H = doc[pno].rect.height if 0 <= pno < len(doc) else 842.0
        x0, y0, x1, y1 = float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])
        cx = (x0 + x1) * 0.5

        match = False
        if target_mode == "সব বক্স":
            match = True
        elif target_mode == "নির্দিষ্ট বক্স" and idx in target_indices:
            match = True
        elif target_mode == "বাম কলাম" and cx < (W * 0.5):
            match = True
        elif target_mode == "ডান কলাম" and cx >= (W * 0.5):
            match = True
        elif target_mode == "নির্দিষ্ট পেজ" and int(row["page"]) == int(target_page):
            match = True

        if match:
            x0 = max(2.0, min(W - 24.0, x0 + float(dx)))
            y0 = max(2.0, min(H - 18.0, y0 + float(dy)))
            x1 = max(x0 + 20.0, min(W - 2.0, x1 + float(dx) + float(dw)))
            y1 = max(y0 + 12.0, min(H - 2.0, y1 + float(dy) + float(dh)))

        updated_rows.append({
            "keep": True,
            "page": int(row["page"]),
            "x0": round(x0, 1),
            "y0": round(y0, 1),
            "x1": round(x1, 1),
            "y1": round(y1, 1),
        })

    doc.close()
    return pd.DataFrame(updated_rows, columns=["keep", "page", "x0", "y0", "x1", "y1"])

def prepare_pages_payload_for_canvas(raw_pdf, page_indices):
    doc = fitz.open(stream=raw_pdf, filetype="pdf")
    pages_data = []
    for pno in page_indices:
        if not (0 <= pno < len(doc)):
            continue
        page = doc[pno]
        W, H = float(page.rect.width), float(page.rect.height)
        zoom = min(1.55, 920.0 / max(1.0, W))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 1:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
        pages_data.append({
            "page": pno + 1,
            "w_pt": round(W, 2),
            "h_pt": round(H, 2),
            "img_b64": b64
        })
    doc.close()
    return pages_data

def build_interactive_canvas_html(pages_data, initial_boxes, file_key, template_file="canvas_editor.html"):
    pages_json = json.dumps(pages_data)
    boxes_json = json.dumps(initial_boxes)
    safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", file_key)

    if os.path.exists(template_file):
        with open(template_file, "r", encoding="utf-8") as f:
            tpl = f.read()
        return (
            tpl.replace("{{PAGES_JSON}}", pages_json)
               .replace("{{BOXES_JSON}}", boxes_json)
               .replace("{{SAFE_KEY}}", safe_key)
        )
    return "<div style='color:#fff;padding:12px;'>canvas_editor.html ফাইলটি পাওয়া যায়নি।</div>"

def render_results_and_review(raw_pdf, file_key):
    ver = st.session_state.get("rv_ver", 0)

    bridge_val = st.text_input(
        "BOX_JSON_BRIDGE",
        value="",
        placeholder="BOX_JSON_BRIDGE_MARKER",
        key=f"bridge_in_{ver}",
        label_visibility="collapsed"
    )
    if bridge_val and bridge_val != st.session_state.get("last_bridge_val"):
        st.session_state["last_bridge_val"] = bridge_val
        try:
            payload = json.loads(bridge_val)
            new_boxes = payload.get("boxes", [])
            new_df = pd.DataFrame(new_boxes, columns=["keep", "page", "x0", "y0", "x1", "y1"])
            st.session_state["rv_rows"] = new_df
            src = fitz.open(stream=raw_pdf, filetype="pdf")
            qs = rows_to_questions(new_df, src)
            if qs:
                built_pdf = build_practice_pdf(src, qs)
                st.session_state["rv_auto_pdf"] = built_pdf
            else:
                st.session_state["rv_auto_pdf"] = None
            src.close()
        except Exception as e:
            st.error(f"ত্রুটি: {e}")

    rows_df = st.session_state["rv_rows"]
    active_count = len(rows_df[rows_df["keep"] == True]) if not rows_df.empty else 0

    if st.session_state.get("rv_auto_pdf") and active_count > 0:
        st.markdown(f"""
        <div class="result-banner">
            <div class="rb-left">
                <div class="rb-check">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                </div>
                <div>
                    <div class="rb-title">প্র্যাকটিস শিট প্রস্তুত</div>
                    <div class="rb-sub">{st.session_state.get("rv_pages", 0)} পৃষ্ঠা স্ক্যান সম্পন্ন</div>
                </div>
            </div>
            <div class="rb-stat-pill">{active_count}টি প্রশ্ন</div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="ডাউনলোড (PDF)",
            data=st.session_state["rv_auto_pdf"],
            file_name="Practice_Sheet.pdf",
            mime="application/pdf",
            key=f"dl_top_{ver}_{active_count}"
        )

    with st.expander("বক্স এডিটর", expanded=False):
        selected_indices = st.session_state.get("rv_page_indices", [0])
        pages_payload = prepare_pages_payload_for_canvas(raw_pdf, selected_indices)
        initial_boxes = rows_df.to_dict(orient="records") if not rows_df.empty else []

        if components is not None:
            canvas_html = build_interactive_canvas_html(pages_payload, initial_boxes, file_key)
            components.html(canvas_html, height=820, scrolling=False)

# ১০. মেইন ইন্টারফেস
st.markdown("""
<div class="studio-header">
    <div>
        <h1 class="studio-title">Practice Sheet Studio</h1>
        <p class="studio-sub">সমাধান বাদ দিয়ে পরিচ্ছন্ন প্র্যাকটিস শিট</p>
    </div>
    <span class="studio-badge">VISION ENGINE</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("পিডিএফ ফাইল", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None:
    raw_pdf = uploaded_file.getvalue()
    size_mb = len(raw_pdf) / (1024 * 1024)
    try:
        tmp_doc = fitz.open(stream=raw_pdf, filetype="pdf")
        total_pdf_pages = len(tmp_doc)
        tmp_doc.close()
    except Exception:
        total_pdf_pages = 0

    st.markdown(f"""
    <div class="uploaded-file-card">
        <div class="uf-left">
            <div class="uf-icon-box">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#5E9E8B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
            </div>
            <div>
                <div class="uf-name">{uploaded_file.name}</div>
                <div class="uf-details">{size_mb:.2f} MB • {total_pdf_pages} PAGES</div>
            </div>
        </div>
        <div class="uf-badge">READY</div>
    </div>
    """, unsafe_allow_html=True)

    file_key = f"{uploaded_file.name}|{len(raw_pdf)}"
    if st.session_state.get("rv_file_key") != file_key:
        for _k in ("rv_rows", "rv_auto_pdf", "rv_pages", "rv_page_indices", "last_bridge_val"):
            st.session_state.pop(_k, None)

    if st.button("স্ক্যান করুন"):
        progress_slot = st.empty()
        try:
            src_doc = fitz.open(stream=raw_pdf, filetype="pdf")
            selected_pages = parse_page_range(PAGE_RANGE_TEXT, len(src_doc))
            questions = extract_questions_from_pdf(src_doc, selected_pages, progress_slot)
            progress_slot.empty()

            st.session_state["rv_rows"] = questions_to_rows(questions)
            st.session_state["rv_auto_pdf"] = build_practice_pdf(src_doc, questions) if questions else None
            st.session_state["rv_pages"] = len(selected_pages)
            st.session_state["rv_page_indices"] = selected_pages
            st.session_state["rv_file_key"] = file_key
            st.session_state["rv_ver"] = st.session_state.get("rv_ver", 0) + 1
            src_doc.close()

            if not questions:
                st.warning("কোনো প্রশ্ন পাওয়া যায়নি। বক্স এডিটর থেকে বক্স যোগ করুন।")
        except Exception as e:
            progress_slot.empty()
            st.error(f"ত্রুটি: {e}")

    if APPLY_BATCH_BTN and st.session_state.get("rv_rows") is not None:
        try:
            updated_df = apply_sidebar_batch_transform(
                st.session_state["rv_rows"],
                raw_pdf,
                BATCH_TARGET,
                BATCH_BOX_NUMS,
                BATCH_PAGE_NO,
                BATCH_DX,
                BATCH_DY,
                BATCH_DW,
                BATCH_DH,
            )
            st.session_state["rv_rows"] = updated_df
            src_doc = fitz.open(stream=raw_pdf, filetype="pdf")
            qs = rows_to_questions(updated_df, src_doc)
            st.session_state["rv_auto_pdf"] = build_practice_pdf(src_doc, qs) if qs else None
            src_doc.close()
            st.session_state["rv_ver"] = st.session_state.get("rv_ver", 0) + 1
        except Exception as e:
            st.error(f"ত্রুটি: {e}")

    if st.session_state.get("rv_rows") is not None and st.session_state.get("rv_file_key") == file_key:
        render_results_and_review(raw_pdf, file_key)

# ১১. বাহ্যিক canvas_editor.html ফাইল থেকে গ্লোবাল ইউআই ও ডিভাইস মোড লোড করা
if components is not None:
    try:
        global_ui_html = build_interactive_canvas_html([], [], "GLOBAL_UI_SHELL")
        components.html(global_ui_html, height=0, scrolling=False)
    except Exception:
        pass


# ১২. (নতুন) নরম ঝাপসা ব্যাকগ্রাউন্ড ও সিন্থেসাইজ সাউন্ড এফেক্ট
#     উপরের কোনো কোড বদলানো হয়নি — এই অংশ আলাদা ফাইল (ambient.css, sfx.js) থেকে চলে।
#     কোনো সমস্যা হলে এই ব্লক বা ফাইল দুটো মুছে দিলেই আগের অবস্থায় ফিরবে।
def load_ambient_and_sfx(css_file="ambient.css", js_file="sfx.js"):
    try:
        load_external_css(css_file)
    except Exception:
        pass
    if components is None or not os.path.exists(js_file):
        return
    try:
        with open(js_file, "r", encoding="utf-8") as f:
            js_src = f.read()
        js_literal = json.dumps(js_src).replace("</", "<\\/")
        components.html(
            "<script>(function(){try{"
            "var P=window.parent,D=P.document;"
            "if(D.getElementById('ps-sfx-engine'))return;"
            "var s=D.createElement('script');s.id='ps-sfx-engine';"
            "s.textContent=" + js_literal + ";"
            "D.head.appendChild(s);"
            "}catch(e){}})();</script>",
            height=0,
            scrolling=False,
        )
    except Exception:
        pass

load_ambient_and_sfx()
