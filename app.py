# -*- coding: utf-8 -*-
"""
================================================================================
app.py — SMART SCANNER (True Hybrid: Visual Layout Parser + Option Grid + KB Tag Engine)
================================================================================
AI INSTRUCTION MANUAL (যেকোনো এআই বটের জন্য নির্দেশনা):
1. এই প্রোগ্রামের সমস্ত নিয়ম `knowledge_base.json` ফাইলে সংরক্ষিত আছে।
2. স্টেজ ১ (Visual Layout Parser): ওপেনসিভি মরফোলজি, অনুভূমিক প্রজেকশন এবং কন্টুর বিশ্লেষণের মাধ্যমে
   প্রতিটি লাইনের নিখুঁত ভিজ্যুয়াল রো, পাশাপাশি থাকা অপশন সাব-বক্স (`sub_boxes`) এবং গোল অপশন বৃত্ত
   (`find_option_circles_in_row`) শনাক্ত করে।
3. স্টেজ ২ (Narrow-Strip Upscaled Number OCR + Multi-Pass OCR): কলামের বাম মার্জিনের ন্যারো স্ট্রিপকে
   ২x জুম করে ইংরেজি ও বাংলা প্রশ্ন নম্বর (`4.` থেকে `16.`) শতভাগ নিখুঁতভাবে শনাক্ত করে।
4. স্টেজ ৩ (Option Counter + KB Tag Cutoff): ৪টি অপশন (`options_seen_count`) পূর্ণ হওয়া মাত্রই অথবা
   `knowledge_base.json`-এর যেকোনো `solution_tags` ('উত্তর:', 'ব্যাখ্যা:', 'সমাধান:') বা বাম মার্জিনের কোলন
   দেখলেই প্রশ্নের বক্স কেটে দেয়।
5. গিটহাবে নতুন নিয়ম সেভ করার জন্য `ADMIN_PIN` দ্বারা নিরাপত্তা নিশ্চিত করা হয়েছে।
================================================================================
"""

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

# উইন্ডোজ পিসিতে লোকালহোস্টে চালানোর সময় ডিফল্ট Tesseract পাথ অটো-ডিটেক্ট করা
if os.name == "nt":
    for _t_path in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ):
        if os.path.exists(_t_path):
            pytesseract.pytesseract.tesseract_cmd = _t_path
            break

try:
    import streamlit.components.v1 as components
except Exception:
    components = None

from learning_engine import (
    load_knowledge_base,
    commit_rule_to_kb,
    analyze_user_corrections,
    verify_admin_pin,
    find_all_tag_conflicts,
    resolve_tag_conflicts,
    sync_all_kb_files,
    CATEGORY_LABELS_BN,
    KB_FILENAME
)

# ১. পেজ কনফিগারেশন
st.set_page_config(
    page_title="SMART SCANNER",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ২. বাহ্যিক style.css ফাইল লোড করা
def load_external_css(css_file="style.css"):
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_external_css()

# ==============================================================================
# ৩. কেন্দ্রীয় নলেজ-বেস (knowledge_base.json) থেকে সব নিয়ম ও ট্যাগ লোড করা
# ==============================================================================
KB = load_knowledge_base()
Q_TAGS = [t for t in KB.get("question_tags", []) if str(t).strip() not in {"|", "।", "/", "\\", "-", ".", ":", "বিগত"}]
SOL_TAGS = KB.get("solution_tags", [])
HEADER_NOISE_LIST = KB.get("header_noise_patterns", [])
BULLET_STOP_LIST = KB.get("bullet_stop_chars", ["➢", "➤", "➣", "•", "♦", "►"])
OPTION_UNITS_LIST = KB.get("option_units", ["mol", "L", "g", "kg", "mL", "ml", "cm", "mm", "m", "gm", "sec", "min", "s", "×", "x", "10^"])
DOTLESS_KW_LIST = KB.get("dotless_question_keywords", ["কত", "কোনটি", "নির্ণয়", "নির্ণয়"])
VALID_CROP_KW_LIST = KB.get("valid_crop_keywords", [])
EQ_GUARD_KW_LIST = KB.get("equation_question_guard_words", [])
LAYOUT_CFG = KB.get("layout_thresholds", {})

MAX_Q_NUMBER = int(LAYOUT_CFG.get("max_question_number", 160))
MIN_START_GAP_PT = float(LAYOUT_CFG.get("min_start_gap_pt", 16.0))
MAX_Q_X_OFFSET = min(28.0, float(LAYOUT_CFG.get("max_q_x_offset", 26.0)))
MAX_SOL_X_OFFSET = float(LAYOUT_CFG.get("max_sol_x_offset", 120.0))
DEFAULT_BLANK_GAP_PT = int(float(LAYOUT_CFG.get("max_internal_blank_gap_pt", 24.0)))


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


# ৪. মিনিমাল সাইডবার ও পিন-সুরক্ষিত লাইভ ট্রেনিং প্যানেল
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
        MAX_BLANK_GAP_PT = st.slider(
            "সর্বোচ্চ ভেতরের গ্যাপ",
            12, 80, DEFAULT_BLANK_GAP_PT, step=2,
            help="প্রশ্নের নিচে উত্তর লেখার জন্য বড় ফাঁকা জায়গা থাকলে এই মাপের পর বক্সটি স্বয়ংক্রিয়ভাবে কেটে যাবে।"
        )

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

    with st.expander("স্মার্ট ট্রেনিং ও এক্স-রে", expanded=False):
        XRAY_MODE = st.checkbox("এক্স-রে মোড", value=False)

        if not st.session_state.get("is_admin_unlocked", False):
            pin_in = st.text_input("পিন", type="password", placeholder="🔒 পিন...", key="sb_pin_input", label_visibility="collapsed")
            if st.button("আনলক", key="btn_unlock_admin"):
                if verify_admin_pin(pin_in):
                    st.session_state["is_admin_unlocked"] = True
                    st.rerun()
                else:
                    st.error("ভুল পিন!")
        else:
            RULE_CAT_LABELS = {
                "সমাধান": "solution_tags",
                "প্রশ্ন": "question_tags",
                "জলছাপ": "header_noise_patterns",
                "বুলেট": "bullet_stop_chars",
                "কীওয়ার্ড": "valid_crop_keywords"
            }
            CAT_KEY_TO_LABEL = {v: k for k, v in RULE_CAT_LABELS.items()}

            sel_cat_label = st.selectbox("ক্যাটাগরি", list(RULE_CAT_LABELS.keys()))
            manual_rule_val = st.text_input("ট্যাগ", "", placeholder="ট্যাগ লিখুন...")

            if st.button("✓ সেভ", key="btn_manual_rule"):
                if manual_rule_val.strip():
                    cat_key = RULE_CAT_LABELS[sel_cat_label]
                    raw_in = manual_rule_val.strip()
                    ok, msg = commit_rule_to_kb(cat_key, raw_in, "Manual rule", force_move=False)
                    if ok:
                        st.session_state.pop("sb_conflict_state", None)
                        st.toast("সেভ হয়েছে!")
                        st.rerun()
                    elif msg.startswith("CONFLICT|"):
                        parts = msg.split("|", 2)
                        st.session_state["sb_conflict_state"] = {
                            "val": raw_in,
                            "target_cat": cat_key,
                            "other_cats": parts[1].split(",")
                        }
                    else:
                        st.warning(msg)

            c_state = st.session_state.get("sb_conflict_state")
            if c_state:
                other_short = ", ".join(CAT_KEY_TO_LABEL.get(c, c) for c in c_state["other_cats"])
                st.error(f"⚠️ '{c_state['val']}' আছে: {other_short}")
                choose_options = {
                    "প্রশ্ন": "question_tags",
                    "সমাধান": "solution_tags",
                    "জলছাপ": "header_noise_patterns"
                }
                def_label = CAT_KEY_TO_LABEL.get(c_state["target_cat"], "প্রশ্ন")
                opt_list = list(choose_options.keys())
                def_idx = opt_list.index(def_label) if def_label in opt_list else 0
                resolved_label = st.selectbox(
                    "রাখুন:",
                    opt_list,
                    index=def_idx,
                    key="sb_conflict_resolve_sel"
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("✓ স্থানান্তর", key="btn_confirm_move"):
                        chosen_k = choose_options[resolved_label]
                        commit_rule_to_kb(chosen_k, c_state["val"], f"Moved to {chosen_k}", force_move=True)
                        st.session_state.pop("sb_conflict_state", None)
                        st.toast("স্থানান্তরিত!")
                        st.rerun()
                with rc2:
                    if st.button("✕ বাতিল", key="btn_cancel_move"):
                        st.session_state.pop("sb_conflict_state", None)
                        st.rerun()

            st.markdown("---")
            conflicts_list, _ = find_all_tag_conflicts(KB)
            cat_choice_map = {
                "প্রশ্ন": "question_tags",
                "সমাধান": "solution_tags",
                "জলছাপ": "header_noise_patterns",
                "মুছুন": "DELETE"
            }
            rev_choice_map = {v: k for k, v in cat_choice_map.items()}

            if conflicts_list:
                st.warning(f"🧹 ডুপ্লিকেট: {len(conflicts_list)}টি")
                resolutions_map = {}
                choice_keys = list(cat_choice_map.keys())
                for c_idx, c_item in enumerate(conflicts_list):
                    def_lbl = rev_choice_map.get(c_item.get("default_cat"), "প্রশ্ন")
                    d_idx = choice_keys.index(def_lbl) if def_lbl in choice_keys else 0
                    sel_res = st.selectbox(
                        f"'{c_item['display']}' • [{c_item['where']}]",
                        choice_keys,
                        index=d_idx,
                        key=f"dup_fix_{c_idx}_{c_item['norm_key']}"
                    )
                    resolutions_map[c_item["norm_key"]] = cat_choice_map[sel_res]

                if st.button("🧹 ফিক্স", key="btn_fix_all_dups"):
                    resolve_tag_conflicts(resolutions_map)
                    st.toast("ডুপ্লিকেট ফিক্সড!")
                    st.rerun()
            else:
                st.caption("✅ ডুপ্লিকেট নেই")

            with st.popover("🔍 ট্যাগ তালিকা", use_container_width=True):
                all_tag_items = []
                for ck, clbl in [("question_tags", "প্রশ্ন"), ("solution_tags", "সমাধান"), ("header_noise_patterns", "জলছাপ")]:
                    for t_val in KB.get(ck, []):
                        disp_t = re.sub(r"\\([.\:\-\(\)\[\]\s])", r"\1", str(t_val).strip())
                        all_tag_items.append((f"{disp_t} • [{clbl}]", disp_t, ck))

                if all_tag_items:
                    sel_tag_str = st.selectbox("ট্যাগ", [x[0] for x in all_tag_items], key="inspect_tag_sel")
                    picked = next(x for x in all_tag_items if x[0] == sel_tag_str)
                    cur_lbl = rev_choice_map.get(picked[2], "প্রশ্ন")
                    c_keys = list(cat_choice_map.keys())
                    new_lbl = st.selectbox("রাখুন", c_keys, index=c_keys.index(cur_lbl), key="inspect_cat_sel")
                    if st.button("✓ আপডেট", key="btn_inspect_update"):
                        target_c = cat_choice_map[new_lbl]
                        resolve_tag_conflicts({picked[1].lower(): target_c}, f"Updated {picked[1]}")
                        st.toast("আপডেট হয়েছে!")
                        st.rerun()

            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("🔄 সিঙ্ক", key="btn_manual_sync"):
                    sync_all_kb_files(KB, "Manual sync")
                    st.toast("সিঙ্ক সম্পন্ন!")
                    st.rerun()
            with bc2:
                if st.button("🔒 লক", key="btn_lock_admin"):
                    st.session_state["is_admin_unlocked"] = False
                    st.rerun()

        st.download_button(
            label="⬇ ব্যাকআপ",
            data=json.dumps(KB, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=KB_FILENAME,
            mime="application/json",
            key="dl_kb_backup"
        )

    mode_short = "AUTO" if PAGE_MODE == "অটো" else ("BOOK" if PAGE_MODE == "বই" else "SLIDE")
    range_short = PAGE_RANGE_TEXT.strip() if PAGE_RANGE_TEXT.strip() else "ALL"

    st.markdown(f"""
    <div class="sb-summary-card">
        <div class="sb-summary-row"><span>মোড</span><span class="sb-summary-val">{mode_short} • {COLUMNS_PER_PAGE}C</span></div>
        <div class="sb-summary-row"><span>পৃষ্ঠা • গ্যাপ</span><span class="sb-summary-val">{range_short} • {QUESTION_GAP_PT}pt</span></div>
        <div class="sb-summary-row"><span>ট্যাগ</span><span class="sb-summary-val">{len(Q_TAGS)}Q • {len(SOL_TAGS)}S</span></div>
    </div>
    """, unsafe_allow_html=True)

    try:
        from info_panel import render_info_panel
        render_info_panel()
    except Exception:
        pass

# ==============================================================================
# ৫. ডায়নামিক রেগুলার এক্সপ্রেশন ইঞ্জিন (knowledge_base.json থেকে চালিত)
# ==============================================================================
SC = 2.5

q_prefix_pattern = "|".join(re.escape(t) for t in sorted(Q_TAGS, key=len, reverse=True)) if Q_TAGS else "Q|প্রশ্ন"
sol_prefix_pattern = "|".join(re.escape(t) for t in sorted(SOL_TAGS, key=len, reverse=True)) if SOL_TAGS else "Sol|সমাধান|উত্তর"
bullet_chars_escaped = "".join(re.escape(c) for c in BULLET_STOP_LIST)
unit_pattern = "|".join(re.escape(u) for u in sorted(OPTION_UNITS_LIST, key=len, reverse=True))
dotless_kw_pattern = "|".join(re.escape(w) for w in DOTLESS_KW_LIST) if DOTLESS_KW_LIST else "কত|কোনটি"
valid_crop_kw_pattern = "|".join(re.escape(w) for w in VALID_CROP_KW_LIST) if VALID_CROP_KW_LIST else "কত|কোনটি"
eq_guard_kw_pattern = "|".join(re.escape(w) for w in EQ_GUARD_KW_LIST) if EQ_GUARD_KW_LIST else "কত|কোনটি|হলে|যদি"
noise_pattern_str = "|".join(HEADER_NOISE_LIST) if HEADER_NOISE_LIST else "Educationalpdf"

# প্রশ্ন নম্বর ম্যাচিং: 4., 5.?, 11., 13.Cs, (১৪), Q1: ইত্যাদি
NUM_RE = re.compile(
    rf"^[\(\[]?(?:(?:{q_prefix_pattern})\s*[\.\:\-\ঃ\/\|]*\s*)?([0-9০-৯]{{1,3}})(?:[\.\)\]\:\ঃ\-\?\!\,]+|(?=[A-Za-zঅ-ঔক-হ]))(?![0-9০-৯])",
    re.IGNORECASE
)
PURE_PREFIX_RE = re.compile(rf"^(?:{q_prefix_pattern})[\.\:\-\ঃ\/]*\s*[0-9০-৯]+$", re.IGNORECASE)
SYMBOL_Q_RE = re.compile(r"^[❑▫▪■□]$")
OPT_TOKEN_RE = re.compile(r"^(?:[\(\[]?[a-dA-Dক-ঘ][\)\]\.]|[ⓐⓑⓒⓓⓔ①②③④])$")
INLINE_OPT_TOKEN_RE = re.compile(r"(?:^|\s)(?:[\(\[][a-dA-Dক-ঘ][\)\]]|[a-dA-Dক-ঘ][\)\.]|[ⓐⓑⓒⓓⓔ①②③④])(?=\s|$)")
BULLET_STOP_RE = re.compile(rf"^[{bullet_chars_escaped}]\s*")
OPT_UNIT_RE = re.compile(rf"^(?:{unit_pattern}|10\^?)$", re.IGNORECASE)
BOARD_TAG_RE = re.compile(r"[\[\(].*(?:1[0-9]|2[0-6]|বো|ঢা|রা|চ|য|কু|ব|সি|দি|মা|মেডিকেল|ভার্সিটি|DU|BUET|RU|CU|JU|GST).*[\]\)]", re.IGNORECASE)

SOL_WORD_CORE = (
    r"(?:[sS58\$][oO0aAeE][lL1Ii\|!t](?:[nNmMrRuU23\^\°\*\~\"\'\`\’\”\ⁿ\-\_]+)?|"
    r"solution|ans(?:wer)?|shortcut|hints?|note|"
    rf"{sol_prefix_pattern})"
)

SOL_TOKEN_RE = re.compile(
    rf"^[\(\[\{{•\*{bullet_chars_escaped}\-\—\–\.\:\;\,\'\"\`\~\|]*{SOL_WORD_CORE}(?:$|[\s\:\ঃ\.\;\,\-\—\–\(\[\{{\"'\`\’\”\ⁿ\^\°\*\/\|0-9০-৯])",
    re.IGNORECASE
)

INLINE_SOL_RE = re.compile(
    rf"(?:^|\s|[\]\)\।\|]){SOL_WORD_CORE}(?:\s*[\:\ঃ\-\—\–\.\(\[]|\s+[nNⁿ\^]\b)",
    re.IGNORECASE
)

HEADER_NOISE_RE = re.compile(rf"(?:{noise_pattern_str})", re.IGNORECASE)
DOTLESS_KW_RE = re.compile(rf"(?:{dotless_kw_pattern})\b", re.IGNORECASE)
VALID_CROP_KW_RE = re.compile(rf"(?:{valid_crop_kw_pattern})", re.IGNORECASE)
EQ_GUARD_KW_RE = re.compile(rf"(?:{eq_guard_kw_pattern})", re.IGNORECASE)


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
    stripped = clean.lstrip("([{•*-➢➤.@©\"'`| ").rstrip(".:;ঃ-ⁿ^°*~\"'`)]}| ")
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
# ৬. ইমেজ প্রসেসিং, অপশন সার্কেল ডিটেকশন ও স্মার্ট ব্ল্যাঙ্ক-গ্যাপ ছাঁটাই
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


def find_option_circles_in_row(gray_img, x0_pt, y0_pt, x1_pt, y1_pt):
    """
    ওপেনসিভি কন্টুর (`RETR_LIST`) দিয়ে একটি রো-এর ভেতরে থাকা গোল অপশন বৃত্ত (ক, খ, গ, ঘ বা ⓐ, ⓑ, ⓒ, ⓓ)
    নিখুঁতভাবে শনাক্ত করে এবং তাদের কেন্দ্রগুলোর X-স্থানাঙ্ক (`circle_xs`) রিটার্ন করে।
    """
    x0, y0 = max(0, int(x0_pt * SC)), max(0, int(y0_pt * SC))
    x1, y1 = min(gray_img.shape[1], int(x1_pt * SC)), min(gray_img.shape[0], int(y1_pt * SC))
    if x1 - x0 < 18 or y1 - y0 < 7:
        return []

    roi = gray_img[y0:y1, x0:x1]
    bw = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_d, max_d = int(6.0 * SC), int(16.8 * SC)
    centers_pt = []

    for c in cnts:
        bx, by, bw_c, bh_c = cv2.boundingRect(c)
        if min_d <= bw_c <= max_d and min_d <= bh_c <= max_d:
            aspect = bw_c / float(max(1, bh_c))
            if 0.70 <= aspect <= 1.40:
                area = cv2.contourArea(c)
                rect_area = float(bw_c * bh_c)
                if 0.32 * rect_area <= area <= 0.92 * rect_area:
                    perim = cv2.arcLength(c, True)
                    if perim > 0:
                        circularity = 4.0 * np.pi * area / (perim * perim)
                        if circularity >= 0.42:
                            cx_pt = x0_pt + (bx + bw_c * 0.5) / SC
                            centers_pt.append(cx_pt)

    # কাছাকাছি বা ডাবল-রিং বৃত্তগুলো ডুপ্লিকেটমুক্ত করা
    centers_pt.sort()
    dedup_xs = []
    for cx in centers_pt:
        if not dedup_xs or (cx - dedup_xs[-1]) > 7.5:
            dedup_xs.append(cx)
    return dedup_xs


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


def trim_vertical_whitespace(ink, xr, top, bot, max_gap_pt=None):
    if max_gap_pt is None:
        max_gap_pt = float(MAX_BLANK_GAP_PT)

    h_px, w_px = ink.shape
    x0 = max(0, int((xr[0] + 5.0) * SC))
    x1 = min(w_px, int((xr[1] - 9.0) * SC))
    y0, y1 = max(0, int(top * SC)), min(h_px, int(bot * SC))
    if y1 <= y0 or x1 <= x0:
        return top, bot

    roi = ink[y0:y1, x0:x1].astype(np.uint8)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(14, int(18 * SC))))
    v_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, v_kernel)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int((x1 - x0) * 0.48)), 1))
    h_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, h_kernel)
    clean_roi = roi & (~v_lines) & (~h_lines)

    row_ink = clean_roi.mean(axis=1) > 0.003
    nz = np.where(row_ink)[0]
    if len(nz) == 0:
        return top, bot

    first_row = nz[0]
    last_row = nz[0]
    max_gap_px = int(float(max_gap_pt) * SC)
    min_q_px = int(10.0 * SC)

    for r_idx in nz[1:]:
        if (r_idx - last_row) > max_gap_px and (last_row - first_row) >= min_q_px:
            break
        last_row = r_idx

    return max(top, (y0 + first_row) / SC - 3.5), min(bot, (y0 + last_row) / SC + 5.0)


# ==============================================================================
# ৭. ট্রু হাইব্রিড ইঞ্জিন: ভিজ্যুয়াল লেআউট পার্সার + অপশন গ্রিড কাউন্টার + ওসিআর ট্যাগ ফিউশন
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


def get_column_lines(page, gray_img, x0, x1, H, col_i=0, is_two_col=True, div_fn=None, col_left_hint=None):
    pdf_words = []
    for wx0, wy0, wx1, wy1, text, *_ in page.get_text("words"):
        if x0 - 4 <= wx0 < x1 - 4 and 12 < wy0 < H - 18:
            pdf_words.append((wx0, wy0, wx1, wy1, text.strip()))

    pdf_lines = group_words_into_lines(pdf_words)
    sol_in_pdf = sum(1 for l in pdf_lines if line_has_solution_marker(l, " ".join(w[4] for w in l), x0 + MAX_SOL_X_OFFSET)[0])
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

        # পাস ১: পূর্ণ কলাম eng+ben ওসিআর
        try:
            ocr_data = pytesseract.image_to_data(
                clean_bw, lang="eng+ben", config="--psm 6", output_type=pytesseract.Output.DICT
            )
            for i in range(len(ocr_data["text"])):
                txt = ocr_data["text"][i].strip()
                conf = int(ocr_data["conf"][i]) if str(ocr_data["conf"][i]).lstrip("-").isdigit() else -1
                if not txt or conf < 14:
                    continue
                wx0 = base_x + ocr_data["left"][i] / SC
                wy0 = ocr_data["top"][i] / SC
                wx1 = wx0 + ocr_data["width"][i] / SC
                wy1 = wy0 + ocr_data["height"][i] / SC
                if 12 < wy0 < H - 18:
                    ocr_words.append((wx0, wy0, wx1, wy1, txt))
        except Exception:
            pass

        # পাস ২: হালকা ফন্ট বা ভাঙা বাংলা অক্ষরের জন্য ben+eng ওসিআর
        try:
            thin_bw = cv2.dilate(clean_bw, np.ones((2, 2), np.uint8), iterations=1)
            ocr_data2 = pytesseract.image_to_data(
                thin_bw, lang="ben+eng", config="--psm 6", output_type=pytesseract.Output.DICT
            )
            for i in range(len(ocr_data2["text"])):
                txt = ocr_data2["text"][i].strip()
                conf = int(ocr_data2["conf"][i]) if str(ocr_data2["conf"][i]).lstrip("-").isdigit() else -1
                if not txt or conf < 14:
                    continue
                wx0 = base_x + ocr_data2["left"][i] / SC
                wy0 = ocr_data2["top"][i] / SC
                wx1 = wx0 + ocr_data2["width"][i] / SC
                wy1 = wy0 + ocr_data2["height"][i] / SC
                if 12 < wy0 < H - 18:
                    ocr_thin_words.append((wx0, wy0, wx1, wy1, txt))
        except Exception:
            pass

        # পাস ৩: বাম মার্জিনের ন্যারো স্ট্রিপ (মাত্র 42pt চওড়া) ২x আপস্কেল করে ইংরেজি সংখ্যা (4., 11., 14., 15.) পড়া
        try:
            rel_left_pt = max(0.0, (col_left_hint - base_x - 6.0)) if col_left_hint is not None else 0.0
            s_px0 = max(0, int(rel_left_pt * SC))
            s_px1 = min(clean_bw.shape[1], s_px0 + int(44.0 * SC))
            if s_px1 - s_px0 >= int(20 * SC):
                strip_img = clean_bw[:, s_px0:s_px1]
                up_scale = 1.6
                strip_up = cv2.resize(strip_img, None, fx=up_scale, fy=up_scale, interpolation=cv2.INTER_CUBIC)
                eng_data = pytesseract.image_to_data(
                    strip_up, lang="eng", config="--psm 6", output_type=pytesseract.Output.DICT
                )
                strip_base_x = base_x + (s_px0 / SC)
                for i in range(len(eng_data["text"])):
                    txt = eng_data["text"][i].strip()
                    conf = int(eng_data["conf"][i]) if str(eng_data["conf"][i]).lstrip("-").isdigit() else -1
                    if not txt or conf < 10:
                        continue
                    wx0 = strip_base_x + (eng_data["left"][i] / (SC * up_scale))
                    wy0 = eng_data["top"][i] / (SC * up_scale)
                    wx1 = wx0 + (eng_data["width"][i] / (SC * up_scale))
                    wy1 = wy0 + (eng_data["height"][i] / (SC * up_scale))
                    if 12 < wy0 < H - 18:
                        ocr_eng_words.append((wx0, wy0, wx1, wy1, txt))
        except Exception:
            pass

    return (
        pdf_lines,
        group_words_into_lines(ocr_words),
        group_words_into_lines(ocr_thin_words),
        group_words_into_lines(ocr_eng_words),
    )


def _extract_row_sub_boxes(clean_ink, base_x, by_start, by_end):
    """
    একটি একক ভিজ্যুয়াল লাইনের (by_start:by_end) ভেতর থেকে পাশাপাশি থাকা আলাদা শব্দগুচ্ছ বা
    অপশন ব্লকগুলোকে (`sub_boxes`) নিখুঁতভাবে আলাদা করে।
    """
    band = clean_ink[by_start:by_end, :]
    if band.size == 0 or not band.any():
        return []

    kw_sub = max(3, int(6.8 * SC))
    band_dil = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_RECT, (kw_sub, 3)), iterations=1)
    col_has_ink = band_dil.any(axis=0)

    sbs = []
    in_seg = False
    seg_x0 = 0
    for c_i, val in enumerate(list(col_has_ink) + [False]):
        if val and not in_seg:
            in_seg = True
            seg_x0 = c_i
        elif not val and in_seg:
            in_seg = False
            w_pt = (c_i - seg_x0) / SC
            if w_pt >= 2.8:
                slice_cols = band[:, seg_x0:c_i]
                rows_nz = np.where(slice_cols.any(axis=1))[0]
                if len(rows_nz) > 0:
                    sy0 = (by_start + rows_nz[0]) / SC
                    sy1 = (by_start + rows_nz[-1] + 1) / SC
                    if (sy1 - sy0) >= 2.8:
                        sbs.append({
                            "x0": base_x + seg_x0 / SC,
                            "y0": sy0,
                            "x1": base_x + c_i / SC,
                            "y1": sy1
                        })
    return sbs


def _split_tall_band_recursively(clean_ink, by0, by1):
    """
    যদি বাংলার আ-কার/ঈ-কার বা কম স্পেসিংয়ের কারণে একাধিক লাইন জোড়া লেগে (২১ পয়েন্টের বেশি লম্বা) যায়,
    তবে অনুভূমিক কালির ঘনত্বের সর্বনিম্ন বিন্দু (Valley) খুঁজে সেগুলোকে আলাদা লাইনে ভাগ করে।
    """
    h_pt = (by1 - by0) / SC
    if h_pt <= 20.5:
        return [(by0, by1)]

    band = clean_ink[by0:by1, :]
    prof = band.mean(axis=1)
    min_line_px = int(8.5 * SC)
    max_line_px = int(17.5 * SC)

    if len(prof) < min_line_px * 2:
        return [(by0, by1)]

    search_end = min(len(prof) - min_line_px, max_line_px)
    if search_end <= min_line_px:
        mid = len(prof) // 2
        return [(by0, by0 + mid), (by0 + mid, by1)]

    split_rel = min_line_px + int(np.argmin(prof[min_line_px:search_end]))
    top_part = (by0, by0 + split_rel)
    bot_part = (by0 + split_rel, by1)
    return [top_part] + _split_tall_band_recursively(clean_ink, bot_part[0], bot_part[1])


def build_hybrid_visual_rows(gray_img, ink, scan_x0, scan_x1, H, col_i, is_two_col, div_fn, pdf_lines, ocr_lines, ocr_thin_lines, ocr_eng_lines):
    """
    স্টেজ ১ (Visual Layout Parser):
    ওপেনসিভি মরফোলজি ও প্রজেকশন প্রোফাইল দিয়ে পাতার প্রতিটি লাইনের নিখুঁত জ্যামিতিক রো,
    পাশাপাশি থাকা অপশন ব্লক (`sub_boxes`) এবং অপশন বৃত্ত (`circle_xs`) তৈরি করে।
    """
    x0_px, x1_px = max(0, int(scan_x0 * SC)), min(ink.shape[1], int(scan_x1 * SC))
    roi_ink = ink[:, x0_px:x1_px].astype(np.uint8).copy()

    if is_two_col and div_fn is not None:
        for r_px in range(roi_ink.shape[0]):
            d_px = int(div_fn(r_px / SC) * SC) - x0_px
            if col_i == 1 and 0 <= d_px + 4 < roi_ink.shape[1]:
                roi_ink[r_px, :max(0, d_px + 4)] = 0
            elif col_i == 0 and 0 <= d_px - 4 < roi_ink.shape[1]:
                roi_ink[r_px, max(0, d_px - 4):] = 0

    v_lines = cv2.morphologyEx(roi_ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(25 * SC))))
    h_lines = cv2.morphologyEx(roi_ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(70 * SC), 1)))
    clean_ink = roi_ink & (~v_lines) & (~h_lines)

    base_x = x0_px / SC
    kw = max(3, int(8.5 * SC))
    kh = max(1, int(1.4 * SC))
    dilated = cv2.dilate(clean_ink, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)), iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    seg_boxes = []
    for c in cnts:
        bx, by, bw_c, bh_c = cv2.boundingRect(c)
        w_pt, h_pt = bw_c / SC, bh_c / SC
        y0_pt, y1_pt = by / SC, (by + bh_c) / SC
        if w_pt < 3.0 or h_pt < 3.2 or y0_pt < 28.0 or y1_pt > (H - 22.0):
            continue

        if h_pt > 20.5:
            sub_roi = clean_ink[by:by + bh_c, bx:bx + bw_c]
            row_proj = sub_roi.mean(axis=1) > 0.006
            raw_bands = []
            in_b = False
            b_st = 0
            for r_i, val in enumerate(list(row_proj) + [False]):
                if val and not in_b:
                    in_b = True
                    b_st = r_i
                elif not val and in_b:
                    in_b = False
                    if (r_i - b_st) / SC >= 3.2:
                        raw_bands.append((by + b_st, by + r_i))

            if not raw_bands:
                raw_bands = [(by, by + bh_c)]

            fine_bands = []
            for rb0, rb1 in raw_bands:
                fine_bands.extend(_split_tall_band_recursively(clean_ink, rb0, rb1))

            for fb0, fb1 in fine_bands:
                sbs = _extract_row_sub_boxes(clean_ink, base_x, fb0, fb1)
                for sb in sbs:
                    if 28.0 <= sb["y0"] and sb["y1"] <= (H - 22.0):
                        seg_boxes.append(sb)
        else:
            x0_pt = base_x + bx / SC
            x1_pt = base_x + (bx + bw_c) / SC
            seg_boxes.append({"x0": x0_pt, "y0": y0_pt, "x1": x1_pt, "y1": y1_pt})

    seg_boxes.sort(key=lambda b: ((b["y0"] + b["y1"]) * 0.5, b["x0"]))
    rows = []
    for b in seg_boxes:
        cy = (b["y0"] + b["y1"]) * 0.5
        if rows and abs(cy - rows[-1]["cy"]) <= 6.5:
            rows[-1]["sub_boxes"].append(b)
            sbs = rows[-1]["sub_boxes"]
            rows[-1]["cy"] = sum((x["y0"] + x["y1"]) * 0.5 for x in sbs) / len(sbs)
            rows[-1]["x0"] = min(x["x0"] for x in sbs)
            rows[-1]["x1"] = max(x["x1"] for x in sbs)
            rows[-1]["y0"] = min(x["y0"] for x in sbs)
            rows[-1]["y1"] = max(x["y1"] for x in sbs)
        else:
            rows.append({
                "cy": cy,
                "x0": b["x0"],
                "x1": b["x1"],
                "y0": b["y0"],
                "y1": b["y1"],
                "sub_boxes": [b],
                "words": [],
                "thin_words": [],
                "eng_words": []
            })

    # প্রতিটি ভিজ্যুয়াল রো-এর ভেতরে সাব-বক্সগুলো কাছাকাছি থাকলে মার্জ করা (যাতে একটি অপশনের শব্দগুলো ১টি সাব-বক্স হয়)
    for r in rows:
        sbs = sorted(r["sub_boxes"], key=lambda x: x["x0"])
        merged_sbs = []
        for sb in sbs:
            if merged_sbs and (sb["x0"] - merged_sbs[-1]["x1"]) <= 7.2:
                merged_sbs[-1]["x1"] = max(merged_sbs[-1]["x1"], sb["x1"])
                merged_sbs[-1]["y0"] = min(merged_sbs[-1]["y0"], sb["y0"])
                merged_sbs[-1]["y1"] = max(merged_sbs[-1]["y1"], sb["y1"])
            else:
                merged_sbs.append(dict(sb))
        r["sub_boxes"] = merged_sbs

    primary_lines = ocr_lines if len(ocr_lines) >= len(pdf_lines) else pdf_lines
    if not rows and primary_lines:
        for ln in primary_lines:
            lx0, ly0 = min(w[0] for w in ln), min(w[1] for w in ln)
            lx1, ly1 = max(w[2] for w in ln), max(w[3] for w in ln)
            rows.append({
                "cy": (ly0 + ly1) * 0.5,
                "x0": lx0,
                "x1": lx1,
                "y0": ly0,
                "y1": ly1,
                "sub_boxes": [{"x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3]} for w in ln],
                "words": list(ln),
                "thin_words": [],
                "eng_words": []
            })

    def _assign_words_to_rows(word_lines, target_key):
        for ln in word_lines:
            for w in ln:
                w_cy = (w[1] + w[3]) * 0.5
                best_r, best_d = None, 9.0
                for r in rows:
                    d = abs(w_cy - r["cy"])
                    if d < best_d:
                        best_d = d
                        best_r = r
                if best_r is not None:
                    best_r[target_key].append(w)

    _assign_words_to_rows(primary_lines, "words")
    _assign_words_to_rows(ocr_thin_lines, "thin_words")
    _assign_words_to_rows(ocr_eng_lines, "eng_words")

    for r in rows:
        r["words"].sort(key=lambda w: w[0])
        r["thin_words"].sort(key=lambda w: w[0])
        r["eng_words"].sort(key=lambda w: w[0])
        r["text"] = " ".join(w[4] for w in r["words"]).strip()
        r["thin_text"] = " ".join(w[4] for w in r["thin_words"]).strip()
        r["eng_text"] = " ".join(w[4] for w in r["eng_words"]).strip()
        c_xs = find_option_circles_in_row(gray_img, r["x0"] - 4.0, r["y0"] - 2.0, r["x1"] + 4.0, r["y1"] + 2.0)
        r["circle_xs"] = c_xs
        r["circles"] = len(c_xs)

    return rows


def _looks_like_left_answer_header(r, col_left, col_w):
    """
    অপশনের নিচে বাম মার্জিনে থাকা 'উত্তর:' বা 'ব্যাখ্যা:' লাইন শনাক্ত করে—এমনকি যখন ওসিআর
    বাংলা অক্ষরগুলো ভুল পড়ে।
    """
    if r["x0"] > col_left + 12.0:
        return False

    # ১. মেইন বা থিন ওসিআর-এ সমাধান ট্যাগ চেক করা
    has_s1, _ = line_has_solution_marker(r["words"], r["text"], col_left + MAX_SOL_X_OFFSET)
    if has_s1:
        return True
    has_s2, _ = line_has_solution_marker(r["thin_words"], r["thin_text"], col_left + MAX_SOL_X_OFFSET)
    if has_s2:
        return True

    # ২. প্রথম বা দ্বিতীয় শব্দের শেষে কোলন (:) বা বিসর্গ (ঃ) আছে কি না (যেমন 'ডত্তর:', 'ত্তঃ', 'ব্যখ্যা:')
    for w_list in (r["words"], r["thin_words"], r["eng_words"]):
        for w in w_list[:2]:
            if w[0] <= col_left + 44.0:
                tok = w[4].strip()
                if tok.endswith(":") or tok.endswith("ঃ") or (":" in tok and not re.match(r"^[0-9০-৯]+:$", tok)):
                    return True

    # ৩. ভিজ্যুয়াল জ্যামিতি: 'উত্তর: ঘ' বা 'উত্তর: গ ক্লোরোপ্লাস্ট' প্যাটর্ন
    # (প্রথম সাব-বক্সটি ১৬-৪২ পয়েন্ট চওড়া 'উত্তর:', এবং তার ঠিক পরেই একটি ছোট একক অক্ষর বা ১টি বৃত্ত আছে, কিন্তু ডান কলামে কোনো ২য় অপশন নেই)
    sbs = r["sub_boxes"]
    if len(sbs) >= 2 and r["circles"] <= 1:
        first_w = sbs[0]["x1"] - sbs[0]["x0"]
        gap_01 = sbs[1]["x0"] - sbs[0]["x1"]
        sec_w = sbs[1]["x1"] - sbs[1]["x0"]
        if 15.0 <= first_w <= 42.0 and gap_01 <= 12.0 and sec_w <= 16.0:
            # নিশ্চিত করা যে কলামের মাঝখানে (৪০%-৭০%) নতুন কোনো ২য় অপশন কলাম শুরু হয়নি
            has_mid_option = any((col_left + col_w * 0.40) <= sb["x0"] <= (col_left + col_w * 0.68) for sb in sbs[1:])
            if not has_mid_option:
                return True

    return False


def _row_starts_with_question_number(r, col_left):
    """
    বাম মার্জিন ঘেঁষে আসল প্রশ্ন নম্বর (যেমন 4., 5., 11., 14., ১৫., Q1) আছে কি না তা যাচাই করে।
    ব্যাখ্যার ভেতরের বা শেষ লাইনের লেখাকে কখনোই প্রশ্ন হতে দেয় না।
    """
    # ১. ভিজ্যুয়াল রো অবশ্যই বাম মার্জিনের কাছে শুরু হতে হবে
    if r["x0"] > col_left + MAX_Q_X_OFFSET:
        return False, None

    # ২. রো-তে যদি ২+ অপশন বৃত্ত থাকে তবে সেটি অপশন লাইন, প্রশ্ন শুরুর লাইন নয়
    if r["circles"] >= 2:
        return False, None

    # ৩. রো-এর প্রশস্ততা যথেষ্ট হতে হবে অথবা পাশাপাশি অন্তত ২টি সাব-বক্স (নম্বর + প্রশ্ন) থাকতে হবে
    if (r["x1"] - r["x0"]) < 38.0 and len(r["sub_boxes"]) < 2:
        return False, None

    # ৪. ন্যারো স্ট্রিপ ইংরেজি ওসিআর, মেইন ওসিআর এবং থিন ওসিআরের বাম প্রান্তের টোকেন পরীক্ষা করা
    candidates = []
    for ew in r["eng_words"][:2]:
        if ew[0] <= col_left + 28.0:
            candidates.append(ew[4])
    for mw in r["words"][:2]:
        if mw[0] <= col_left + 28.0:
            candidates.append(mw[4])
    for tw in r["thin_words"][:2]:
        if tw[0] <= col_left + 28.0:
            candidates.append(tw[4])

    for tok in candidates:
        t_clean = tok.strip("([{|। ")
        if not t_clean:
            continue
        # দশমিক সংখ্যা (যেমন 2.5 বা 10.0) বাদ দেওয়া
        if re.match(r"^[0-9০-৯]+\.[0-9০-৯]+$", t_clean):
            continue
        if OPT_TOKEN_RE.match(tok.strip()):
            continue

        m = NUM_RE.match(t_clean)
        if m:
            qn = _b2e_int(m.group(1))
            if qn is not None and 1 <= qn <= MAX_Q_NUMBER:
                return True, qn
        elif PURE_PREFIX_RE.match(t_clean) or SYMBOL_Q_RE.match(t_clean):
            return True, None
        elif re.match(r"^[0-9০-৯]{1,2}$", t_clean) and len(r["sub_boxes"]) >= 2:
            # ডট ছাড়া সংখ্যা হলে ১ম সাব-বক্সটি ছোট একটি আলাদা নম্বর ব্লক হতে হবে এবং প্রশ্নে '?' বা 'কত/কোনটি' থাকতে হবে
            first_sb = r["sub_boxes"][0]
            first_sb_w = first_sb["x1"] - first_sb["x0"]
            if first_sb["x0"] <= col_left + 16.0 and first_sb_w <= 20.0:
                comb_t = f"{r['text']} {r['thin_text']}"
                if "?" in comb_t or DOTLESS_KW_RE.search(comb_t):
                    qn = _b2e_int(t_clean)
                    if qn is not None and 1 <= qn <= MAX_Q_NUMBER:
                        return True, qn

    return False, None


def _classify_option_row(r, col_left, col_w, seen_options, options_seen_count):
    """
    একটি ভিজ্যুয়াল রো অপশন সারি কি না (`is_opt_row`) এবং এই সারিতে কয়টি অপশন আছে (`opt_count_in_row`)
    তা জ্যামিতি, অপশন বৃত্ত এবং টোকেন মিলিয়ে নির্ণয় করে।
    """
    txt = f"{r['text']} {r['thin_text']}".strip()
    circle_cnt = r["circles"]
    c_xs = r["circle_xs"]

    # বোর্ড ট্যাগ ([DU-A 22-23], [সি.বো. ১৯]) বাদে মূল সাব-বক্সগুলো বের করা
    sbs = r["sub_boxes"]
    if len(sbs) >= 2 and (BOARD_TAG_RE.search(txt) or BOARD_TAG_RE.search(r["eng_text"])):
        core_sbs = [sb for sb in sbs if sb["x0"] < col_left + col_w * 0.68]
        if not core_sbs:
            core_sbs = sbs[:1]
    else:
        core_sbs = sbs

    # কলামের মাঝামাঝি (২৮% থেকে ৭২% প্রস্থের মধ্যে) ২য় অপশন কলামের সাব-বক্স বা বৃত্ত আছে কি না
    has_mid_col_box = len(core_sbs) >= 2 and any(
        (col_left + col_w * 0.28) <= sb["x0"] <= (col_left + col_w * 0.74)
        for sb in core_sbs[1:]
    )
    has_mid_col_circle = any(
        (col_left + col_w * 0.28) <= cx <= (col_left + col_w * 0.78)
        for cx in c_xs
    )

    ocr_opt_matches = INLINE_OPT_TOKEN_RE.findall(r["text"])
    ocr_opt_cnt = len(ocr_opt_matches)
    first_tok_is_opt = bool(r["words"] and OPT_TOKEN_RE.match(r["words"][0][4]))

    is_indented = (col_left + 6.5 < r["x0"] < col_left + col_w * 0.44)

    # ১. যদি ২ বা ততোধিক গোল অপশন বৃত্ত থাকে -> নিশ্চিত অপশন সারি
    if circle_cnt >= 2:
        cnt = 4 if (circle_cnt >= 3 or len(core_sbs) >= 4) else 2
        return True, cnt

    # ২. যদি ওসিআর-এ স্পষ্ট ২+ অপশন টোকেন ((ক), (খ) ইত্যাদি) থাকে
    if ocr_opt_cnt >= 2:
        cnt = 4 if ocr_opt_cnt >= 3 else 2
        return True, cnt

    # ৩. যদি পাশাপাশি ২+ অপশন সাব-বক্স থাকে (যেখানে ২য় বক্সটি কলামের মাঝামাঝি শুরু হয়েছে)
    if has_mid_col_box:
        cnt = 4 if len(core_sbs) >= 4 else 2
        return True, cnt

    # ৪. যদি লাইনটি ইনডেন্টেড হয় এবং তাতে অপশন বৃত্ত বা অপশন টোকেন থাকে
    if is_indented and (circle_cnt >= 1 or first_tok_is_opt or has_mid_col_circle):
        cnt = 2 if (len(core_sbs) >= 2 or has_mid_col_circle) else 1
        return True, cnt

    # ৫. যদি ইতোমধ্যে ১ম সারির অপশন (যেমন ক, খ = ২টি অপশন) দেখা হয়ে থাকে এবং এখনো ৪টি অপশন পূর্ণ না হয় (< 4),
    # তবে ২য় সারির অপশন (গ, ঘ) বাম মার্জিনে শুরু হলেও এবং সাব-বক্স কাছাকাছি থাকলেও সেটিকে অপশন হিসেবে ধরা হবে!
    if seen_options and options_seen_count < 4:
        if circle_cnt >= 1 or first_tok_is_opt or has_mid_col_circle or len(core_sbs) >= 2:
            cnt = 2 if (len(core_sbs) >= 2 or has_mid_col_circle or circle_cnt >= 2) else 1
            return True, cnt

    return False, 0


def extract_column_questions_hybrid(gray_img, ink, rows, col_left, col_right, H, xray_log=None, page_no=1, col_no=1):
    """
    স্টেজ ৩ (Hybrid Layout + Option Counter + KB Tag Engine):
    ভিজ্যুয়াল রো, অপশন কাউন্টার এবং নলেজ-বেস ট্যাগ একসাথে মিলিয়ে প্রতিটি প্রশ্নের নিখুঁত (top_y, bot_y) বের করে।
    """
    if not rows:
        return []

    col_w = max(100.0, col_right - col_left)

    # ১. জলছাপ ও হেডার নয়েজ বাদ দেওয়া
    valid_rows = []
    for r in rows:
        comb_txt = f"{r['text']} {r['thin_text']} {r['eng_text']}".strip()
        if comb_txt and HEADER_NOISE_RE.search(comb_txt):
            if xray_log is not None:
                xray_log.append({
                    "পৃষ্ঠা": page_no, "কলাম": col_no, "Y": round(r["y0"], 1),
                    "সিদ্ধান্ত": "বাতিল (হেডার নয়েজ)", "লাইন": comb_txt[:70]
                })
            continue
        valid_rows.append(r)

    if not valid_rows:
        return []

    # ২. বাম মার্জিনে শুরু হওয়া আসল প্রশ্নগুলোর ইনডেক্স শনাক্ত করা
    q_start_indices = []
    for idx, r in enumerate(valid_rows):
        if _looks_like_left_answer_header(r, col_left, col_w):
            continue
        if ("=" in r["text"] or "⇒" in r["text"] or "∴" in r["text"]) and "?" not in r["text"] and not EQ_GUARD_KW_RE.search(r["text"]):
            continue

        is_q, qn = _row_starts_with_question_number(r, col_left)
        if is_q:
            if not q_start_indices or (r["y0"] - valid_rows[q_start_indices[-1]]["y0"]) >= MIN_START_GAP_PT:
                q_start_indices.append(idx)
                if xray_log is not None:
                    xray_log.append({
                        "পৃষ্ঠা": page_no, "কলাম": col_no, "Y": round(r["y0"], 1),
                        "সিদ্ধান্ত": f"শুরু (প্রশ্ন #{qn if qn else '?'})",
                        "লাইন": (r["text"] or r["eng_text"])[:70]
                    })

    # ৩. প্রতিটি প্রশ্নের শুরু থেকে অপশন গ্রিডের শেষ পর্যন্ত বক্স নির্ধারণ (উত্তর/ব্যাখ্যা বাদ দিয়ে)
    q_bounds = []
    for qi, start_idx in enumerate(q_start_indices):
        end_limit_idx = q_start_indices[qi + 1] if (qi + 1) < len(q_start_indices) else len(valid_rows)
        start_row = valid_rows[start_idx]
        q_top = start_row["y0"] - 2.0
        q_bot = start_row["y1"] + 2.0

        seen_options = False
        options_seen_count = 0
        opt_rows_count = 0
        collected_rows = [start_row]

        # যদি প্রশ্নের ১ম লাইনের ভেতরেই অপশন শুরু হয়ে গিয়ে থাকে (এক লাইনের ছোট প্রশ্ন)
        init_is_opt, init_opt_cnt = _classify_option_row(start_row, col_left, col_w, False, 0)
        if init_is_opt and start_row["circles"] >= 2:
            seen_options = True
            options_seen_count = init_opt_cnt
            opt_rows_count = 1

        for r_idx in range(start_idx + 1, end_limit_idx):
            r = valid_rows[r_idx]
            txt = f"{r['text']} {r['thin_text']}".strip()

            # ক) নলেজ-বেসের বুলেট স্টপ চেক
            if txt and (BULLET_STOP_RE.match(txt) or (r["words"] and BULLET_STOP_RE.match(r["words"][0][4]))):
                if xray_log is not None:
                    xray_log.append({
                        "পৃষ্ঠা": page_no, "কলাম": col_no, "Y": round(r["y0"], 1),
                        "সিদ্ধান্ত": "থামো (বুলেট চিহ্ন)", "লাইন": txt[:70]
                    })
                break

            # খ) নলেজ-বেসের সমাধান ট্যাগ বা বাম মার্জিনের 'উত্তর:' / 'ব্যাখ্যা:' হেডার চেক
            if _looks_like_left_answer_header(r, col_left, col_w):
                if xray_log is not None:
                    xray_log.append({
                        "পৃষ্ঠা": page_no, "কলাম": col_no, "Y": round(r["y0"], 1),
                        "সিদ্ধান্ত": "থামো (উত্তর/সমাধান ট্যাগ)", "লাইন": (txt or r["eng_text"])[:70]
                    })
                break

            # গ) সমীকরণ লাইন ('=', '⇒', '∴') চেক
            if txt and ("=" in txt or "⇒" in txt or "∴" in txt) and "?" not in txt:
                if not EQ_GUARD_KW_RE.search(txt) and r["circles"] < 2:
                    if xray_log is not None:
                        xray_log.append({
                            "পৃষ্ঠা": page_no, "কলাম": col_no, "Y": round(r["y0"], 1),
                            "সিদ্ধান্ত": "থামো (সমীকরণ লাইন)", "লাইন": txt[:70]
                        })
                    break

            # ঘ) যদি ইতোমধ্যে ৪টি অপশন (ক, খ, গ, ঘ) সম্পূর্ণ দেখা হয়ে গিয়ে থাকে, তবে অপশন ব্লক শেষ!
            if seen_options and (options_seen_count >= 4 or opt_rows_count >= 2):
                # ব্যতিক্রম: যদি ২য় অপশন সারিটির লম্বা কোনো অপশন লেখা ঠিক নিচের লাইনে ইনডেন্টেড হয়ে র‍্যাপ (Wrap) করে থাকে
                is_wrapped_opt_cont = (
                    (r["x0"] > col_left + 12.0)
                    and (r["y0"] - q_bot) <= 6.5
                    and r["circles"] == 0
                    and len(r["sub_boxes"]) <= 2
                )
                if is_wrapped_opt_cont:
                    q_bot = r["y1"] + 2.0
                    collected_rows.append(r)
                    continue

                if xray_log is not None:
                    xray_log.append({
                        "পৃষ্ঠা": page_no, "কলাম": col_no, "Y": round(r["y0"], 1),
                        "সিদ্ধান্ত": "থামো (৪টি অপশন সম্পন্ন, নিচে উত্তর/ব্যাখ্যা)",
                        "লাইন": (txt or r["eng_text"])[:70]
                    })
                break

            # ঙ) বর্তমান সারিটি অপশন সারি কি না তা পরীক্ষা করা
            is_opt_row, row_opt_cnt = _classify_option_row(r, col_left, col_w, seen_options, options_seen_count)

            if not seen_options:
                if is_opt_row:
                    seen_options = True
                    opt_rows_count = 1
                    options_seen_count += row_opt_cnt
                    q_bot = r["y1"] + 2.0
                    collected_rows.append(r)
                elif (r["y0"] - q_bot) <= 18.0 and (r_idx - start_idx) <= 6:
                    # প্রশ্নের উদ্দীপক, ২য়-৬ষ্ঠ লাইন (যেমন রোমান সংখ্যা i, ii, iii বা বোর্ড ট্যাগ লাইন)
                    q_bot = r["y1"] + 2.0
                    collected_rows.append(r)
                else:
                    break
            else:
                # অপশন চলাকালীন পরবর্তী অপশন সারি (যেমন গ, ঘ) যোগ করা
                if is_opt_row and (r["y0"] - q_bot) <= 19.0:
                    opt_rows_count += 1
                    options_seen_count += row_opt_cnt
                    q_bot = r["y1"] + 2.0
                    collected_rows.append(r)
                else:
                    if xray_log is not None:
                        xray_log.append({
                            "পৃষ্ঠা": page_no, "কলাম": col_no, "Y": round(r["y0"], 1),
                            "সিদ্ধান্ত": "থামো (অপশন শেষে বাম মার্জিনে উত্তর/ব্যাখ্যা)",
                            "লাইন": (txt or r["eng_text"])[:70]
                        })
                    break

        # ৪. প্যারাগ্রাফ গার্ড: বক্সটিতে যদি অপশন থাকে, অথবা প্রশ্নবোধক চিহ্ন/কীওয়ার্ড থাকে, তবেই তা বৈধ প্রশ্ন
        full_q_text = " ".join(f"{x['text']} {x['thin_text']}" for x in collected_rows).strip()
        total_circles_in_q = sum(x["circles"] for x in collected_rows)
        has_q_mark = "?" in full_q_text or "?" in " ".join(x["eng_text"] for x in collected_rows)
        has_kw = bool(VALID_CROP_KW_RE.search(full_q_text))

        if seen_options or total_circles_in_q >= 2 or has_q_mark or has_kw:
            q_bounds.append((q_top, q_bot))
        elif xray_log is not None:
            xray_log.append({
                "পৃষ্ঠা": page_no, "কলাম": col_no, "Y": round(q_top, 1),
                "সিদ্ধান্ত": "বাতিল (সাধারণ প্যারাগ্রাফ/নয়েজ)", "লাইন": full_q_text[:70]
            })

    return q_bounds


# ==============================================================================
# ৮. স্লাইড মোড: বক্সে ঘেরা প্রশ্ন শনাক্ত, Sol^n ও ফাঁকা জায়গা ছাঁটাই
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
    cur_rect = rect
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
                cur_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, max(rect.y0 + 18.0, new_bot))
                break
    except Exception:
        pass

    try:
        _, trimmed_bot = trim_vertical_whitespace(
            ink,
            (cur_rect.x0 + 6.0, cur_rect.x1 - 6.0),
            cur_rect.y0 + 4.0,
            cur_rect.y1 - 4.0,
            MAX_BLANK_GAP_PT
        )
        if trimmed_bot + 4.0 < cur_rect.y1 and (trimmed_bot - cur_rect.y0) >= 18.0:
            cur_rect = fitz.Rect(cur_rect.x0, cur_rect.y0, cur_rect.x1, trimmed_bot + 2.0)
    except Exception:
        pass
    return cur_rect


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


# ==============================================================================
# ৯. প্রোগ্রেস কার্ড ও মূল এক্সট্রাকশন পাইপলাইন
# ==============================================================================
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


def extract_questions_from_pdf(src_doc, page_indices, progress_slot, collect_xray=False):
    questions = []
    xray_logs = []
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
            hint_left_x = inner_x0 + (col_x_profile[0] / SC if len(col_x_profile) > 0 else 0)

            scan_x0 = min(cx0, div_fn(20.0) + 1.5, div_fn(H - 20.0) + 1.5) if (is_two_col and col_i == 1) else cx0
            scan_x1 = max(cx1, div_fn(20.0) - 1.5, div_fn(H - 20.0) - 1.5) if (is_two_col and col_i == 0) else cx1

            pdf_lines, ocr_lines, ocr_thin_lines, ocr_eng_lines = get_column_lines(
                page, gray, scan_x0, scan_x1, H,
                col_i=col_i, is_two_col=is_two_col, div_fn=div_fn, col_left_hint=hint_left_x
            )

            # স্টেজ ১: ভিজ্যুয়াল কন্টুর ও ওসিআর মিলিয়ে জ্যামিতিক রো এবং অপশন সাব-বক্স তৈরি
            visual_rows = build_hybrid_visual_rows(
                gray, ink, scan_x0, scan_x1, H, col_i, is_two_col, div_fn,
                pdf_lines, ocr_lines, ocr_thin_lines, ocr_eng_lines
            )
            if not visual_rows:
                continue

            wide_lefts = [r["x0"] for r in visual_rows if (r["x1"] - r["x0"]) > 28.0]
            col_left = float(np.percentile(wide_lefts, 12)) if wide_lefts else hint_left_x
            col_right = max(r["x1"] for r in visual_rows)

            log_target = xray_logs if collect_xray else None

            # স্টেজ ২ ও ৩: হাইব্রিড লেআউট পার্সার + অপশন কাউন্টার + নলেজ-বেস ট্যাগ ইঞ্জিন
            q_bounds = extract_column_questions_hybrid(
                gray, ink, visual_rows, col_left, col_right, H,
                xray_log=log_target, page_no=pno + 1, col_no=col_i + 1
            )

            for st_y, end_y in q_bounds:
                div_at_top = div_fn(st_y)
                div_at_bot = div_fn(end_y)
                if is_two_col and col_i == 0:
                    q_x0 = max(8.0, cx0 - 2.0, min(col_left - 5.0, 22.0))
                    q_x1 = min(div_at_top, div_at_bot) - 2.0 - RIGHT_EDGE_TRIM_PT
                elif is_two_col and col_i == 1:
                    q_x0 = max(div_at_top, div_at_bot) + 1.5
                    q_x1 = min(W - 8.0, col_right + 5.0) - RIGHT_EDGE_TRIM_PT
                else:
                    q_x0 = max(8.0, cx0 - 2.0, col_left - 5.0)
                    q_x1 = min(cx1, col_right + 5.0) - RIGHT_EDGE_TRIM_PT

                if q_x1 - q_x0 < 75:
                    continue

                top_y = snap_cut_line(ink, (q_x0, q_x1), max(10.0, st_y - 1.5), up_win=2.5, down_win=1.0)
                bot_y = snap_cut_line(ink, (q_x0, q_x1), min(H - 10.0, end_y + 1.5), up_win=1.5, down_win=2.0)
                top_y, bot_y = trim_vertical_whitespace(ink, (q_x0, q_x1), top_y, bot_y, MAX_BLANK_GAP_PT)

                if (bot_y - top_y) >= MIN_Q_HEIGHT_PT:
                    questions.append((pno, fitz.Rect(q_x0, top_y, q_x1, bot_y)))

        render_progress_card(
            progress_slot, idx + 1, total_to_scan, pno + 1, len(questions),
            f"পৃষ্ঠা {pno + 1} সম্পন্ন"
        )

    return questions, xray_logs


# ==============================================================================
# ১০. নতুন প্র্যাকটিস শিট পিডিএফ তৈরি (১ বা ২ কলাম, ডার্ক-মোড ইনভার্সন সহ)
# ==============================================================================
def build_output_pdf(src_doc, questions, cols_per_page=2, header_title="MCQ Practice Sheet", gap_pt=12, invert_dark=False):
    out_doc = fitz.open()
    PAGE_W, PAGE_H = 595.0, 842.0  # A4 Size
    MARGIN_X = 24.0
    TOP_Y = 52.0
    BOT_Y = PAGE_H - 28.0
    COL_GAP = 16.0

    if cols_per_page == 2:
        col_w = (PAGE_W - 2 * MARGIN_X - COL_GAP) / 2.0
        col_x_list = [MARGIN_X, MARGIN_X + col_w + COL_GAP]
    else:
        col_w = PAGE_W - 2 * MARGIN_X
        col_x_list = [MARGIN_X]

    def _new_sheet_page(page_num):
        p = out_doc.new_page(width=PAGE_W, height=PAGE_H)
        if header_title.strip():
            p.insert_text(
                fitz.Point(MARGIN_X, 30),
                header_title.strip(),
                fontsize=12,
                color=(0.15, 0.22, 0.20)
            )
        p.insert_text(
            fitz.Point(PAGE_W - MARGIN_X - 45, 30),
            f"Page {page_num}",
            fontsize=9,
            color=(0.45, 0.45, 0.45)
        )
        p.draw_line(
            fitz.Point(MARGIN_X, 38),
            fitz.Point(PAGE_W - MARGIN_X, 38),
            color=(0.80, 0.84, 0.82),
            width=0.7
        )
        if cols_per_page == 2:
            mid_x = PAGE_W / 2.0
            p.draw_line(
                fitz.Point(mid_x, TOP_Y),
                fitz.Point(mid_x, BOT_Y),
                color=(0.86, 0.88, 0.87),
                width=0.5
            )
        return p

    if not questions:
        _new_sheet_page(1)
        return out_doc.tobytes()

    sheet_page_no = 1
    cur_page = _new_sheet_page(sheet_page_no)
    cur_col = 0
    cur_y = TOP_Y

    for pno, rect in questions:
        if rect.width <= 5 or rect.height <= 5:
            continue

        scale = col_w / rect.width
        target_h = rect.height * scale

        if cur_y + target_h > BOT_Y and cur_y > TOP_Y + 5.0:
            cur_col += 1
            if cur_col >= cols_per_page:
                sheet_page_no += 1
                cur_page = _new_sheet_page(sheet_page_no)
                cur_col = 0
            cur_y = TOP_Y

        max_avail_h = BOT_Y - cur_y
        if target_h > max_avail_h:
            fit_scale = max_avail_h / rect.height
            draw_w = rect.width * fit_scale
            draw_h = max_avail_h
        else:
            draw_w = col_w
            draw_h = target_h

        x0_dest = col_x_list[cur_col]
        dest_rect = fitz.Rect(x0_dest, cur_y, x0_dest + draw_w, cur_y + draw_h)
        src_page = src_doc[pno]

        if invert_dark and clip_is_dark(src_page, rect):
            pix = src_page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=rect, colorspace=fitz.csRGB, alpha=False)
            img_np = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            inv_np = 255 - img_np
            ok, buf = cv2.imencode(".png", cv2.cvtColor(inv_np, cv2.COLOR_RGB2BGR))
            if ok:
                cur_page.insert_image(dest_rect, stream=buf.tobytes())
        else:
            cur_page.show_pdf_page(dest_rect, src_doc, pno, clip=rect)

        cur_y += draw_h + float(gap_pt)

    return out_doc.tobytes()


def render_page_preview_with_boxes(src_doc, pno, page_boxes, scale=1.35):
    page = src_doc[pno]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()

    for b_idx, rect in page_boxes:
        x0, y0 = int(rect.x0 * scale), int(rect.y0 * scale)
        x1, y1 = int(rect.x1 * scale), int(rect.y1 * scale)
        cv2.rectangle(img, (x0, y0), (x1, y1), (46, 160, 115), 2)
        label = f"#{b_idx + 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(img, (x0, max(0, y0 - th - 6)), (x0 + tw + 8, y0), (46, 160, 115), -1)
        cv2.putText(img, label, (x0 + 4, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ==============================================================================
# ১১. মেইন ইউজার ইন্টারফেস, ইন্টারেক্টিভ বক্স এডিটর ও লাইভ ট্রেনিং ওয়ার্কফ্লো
# ==============================================================================
st.markdown("""
<div class="app-hero-header" style="margin-bottom: 1.2rem;">
    <h2 style="margin:0; font-weight:700; letter-spacing:-0.02em;">SMART SCANNER <span style="font-size:0.55em; padding:3px 8px; border-radius:6px; background:#E8F3EF; color:#2E6F5E; vertical-align:middle;">HYBRID STUDIO</span></h2>
    <p style="margin:4px 0 0 0; font-size:0.9rem; opacity:0.78;">ভিজ্যুয়াল লেআউট পার্সার • অপশন গ্রিড কাউন্টার • নলেজ-বেস ট্যাগ ইঞ্জিন</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("পিডিএফ ফাইল আপলোড করুন", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_sig = f"{uploaded_file.name}_{len(file_bytes)}"

    if st.session_state.get("current_file_sig") != file_sig:
        st.session_state["current_file_sig"] = file_sig
        st.session_state["raw_pdf_bytes"] = file_bytes
        st.session_state.pop("initial_questions", None)
        st.session_state.pop("edited_questions", None)
        st.session_state.pop("xray_logs", None)
        st.session_state.pop("output_pdf_bytes", None)

    src_doc = fitz.open(stream=st.session_state["raw_pdf_bytes"], filetype="pdf")
    total_pages = len(src_doc)
    selected_pages = parse_page_range(PAGE_RANGE_TEXT, total_pages)

    col_act1, col_act2 = st.columns([2, 1])
    with col_act1:
        scan_clicked = st.button("🚀 স্মার্ট স্ক্যান শুরু করুন", type="primary", use_container_width=True)
    with col_act2:
        st.caption(f"মোট পৃষ্ঠা: {total_pages} • নির্বাচিত: {len(selected_pages)}টি")

    progress_slot = st.empty()

    if scan_clicked or "edited_questions" not in st.session_state:
        questions, xray_logs = extract_questions_from_pdf(
            src_doc, selected_pages, progress_slot, collect_xray=XRAY_MODE
        )
        progress_slot.empty()
        st.session_state["initial_questions"] = [(p, fitz.Rect(r)) for p, r in questions]
        st.session_state["edited_questions"] = [(p, fitz.Rect(r)) for p, r in questions]
        st.session_state["xray_logs"] = xray_logs
        st.session_state["output_pdf_bytes"] = build_output_pdf(
            src_doc,
            st.session_state["edited_questions"],
            cols_per_page=COLUMNS_PER_PAGE,
            header_title=HEADER_TITLE,
            gap_pt=QUESTION_GAP_PT,
            invert_dark=INVERT_DARK_OUT
        )

    edited_questions = st.session_state.get("edited_questions", [])

    # সাইডবারের ব্যাচ এডিট বাটন চাপলে নির্বাচিত বক্সগুলোর মাপ একসাথে পরিবর্তন করা
    if APPLY_BATCH_BTN and edited_questions:
        target_indices = set()
        if BATCH_TARGET == "সব বক্স":
            target_indices = set(range(len(edited_questions)))
        elif BATCH_TARGET == "নির্দিষ্ট বক্স":
            target_indices = parse_box_selection_string(BATCH_BOX_NUMS, len(edited_questions))
        elif BATCH_TARGET == "নির্দিষ্ট পেজ":
            target_pno = int(BATCH_PAGE_NO) - 1
            target_indices = {i for i, (p, _) in enumerate(edited_questions) if p == target_pno}
        elif BATCH_TARGET in ("বাম কলাম", "ডান কলাম"):
            for i, (p, r) in enumerate(edited_questions):
                pw = src_doc[p].rect.width
                cx = (r.x0 + r.x1) * 0.5
                if BATCH_TARGET == "বাম কলাম" and cx < (pw * 0.5):
                    target_indices.add(i)
                elif BATCH_TARGET == "ডান কলাম" and cx >= (pw * 0.5):
                    target_indices.add(i)

        updated_list = []
        for i, (pno, r) in enumerate(edited_questions):
            if i in target_indices:
                pw, ph = src_doc[pno].rect.width, src_doc[pno].rect.height
                nx0 = max(2.0, min(pw - 20.0, r.x0 + BATCH_DX))
                ny0 = max(2.0, min(ph - 20.0, r.y0 + BATCH_DY))
                nx1 = max(nx0 + 25.0, min(pw - 2.0, r.x1 + BATCH_DX + BATCH_DW))
                ny1 = max(ny0 + 12.0, min(ph - 2.0, r.y1 + BATCH_DY + BATCH_DH))
                updated_list.append((pno, fitz.Rect(nx0, ny0, nx1, ny1)))
            else:
                updated_list.append((pno, fitz.Rect(r)))

        st.session_state["edited_questions"] = updated_list
        st.session_state["output_pdf_bytes"] = build_output_pdf(
            src_doc,
            updated_list,
            cols_per_page=COLUMNS_PER_PAGE,
            header_title=HEADER_TITLE,
            gap_pt=QUESTION_GAP_PT,
            invert_dark=INVERT_DARK_OUT
        )
        st.toast(f"{len(target_indices)}টি বক্স আপডেট হয়েছে!")
        st.rerun()

    # ফলাফল সামারি ও ডাউনলোড বার
    if edited_questions:
        out_bytes = build_output_pdf(
            src_doc,
            edited_questions,
            cols_per_page=COLUMNS_PER_PAGE,
            header_title=HEADER_TITLE,
            gap_pt=QUESTION_GAP_PT,
            invert_dark=INVERT_DARK_OUT
        )
        st.session_state["output_pdf_bytes"] = out_bytes

        dl_col1, dl_col2 = st.columns([2, 1])
        with dl_col1:
            st.success(f"✅ মোট {len(edited_questions)}টি প্রশ্ন নিখুঁতভাবে শনাক্ত হয়েছে।")
        with dl_col2:
            st.download_button(
                label="⬇ প্র্যাকটিস শিট ডাউনলোড (PDF)",
                data=out_bytes,
                file_name=f"Smart_Sheet_{uploaded_file.name}",
                mime="application/pdf",
                use_container_width=True
            )

        tab_preview, tab_editor, tab_xray = st.tabs([
            "📄 আউটপুট শিট প্রিভিউ",
            "✂️ ভিজ্যুয়াল বক্স এডিটর ও এআই ট্রেনিং",
            "🔬 এক্স-রে লগ"
        ])

        with tab_preview:
            out_doc_preview = fitz.open(stream=out_bytes, filetype="pdf")
            prev_page_no = st.number_input(
                "আউটপুট পৃষ্ঠা দেখুন",
                min_value=1,
                max_value=max(1, len(out_doc_preview)),
                value=1,
                step=1,
                key="out_prev_pno"
            )
            p_pix = out_doc_preview[prev_page_no - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            p_img = np.frombuffer(p_pix.samples, np.uint8).reshape(p_pix.height, p_pix.width, 3)
            st.image(p_img, caption=f"আউটপুট পৃষ্ঠা {prev_page_no} / {len(out_doc_preview)}", use_container_width=True)

        with tab_editor:
            ed_c1, ed_c2 = st.columns([3, 2])
            with ed_c1:
                inspected_pno = st.selectbox(
                    "মূল পিডিএফ পৃষ্ঠা নির্বাচন করুন",
                    options=selected_pages,
                    format_func=lambda x: f"পৃষ্ঠা {x + 1}",
                    key="editor_page_sel"
                )
                page_boxes = [(idx, r) for idx, (p, r) in enumerate(edited_questions) if p == inspected_pno]
                annotated_img = render_page_preview_with_boxes(src_doc, inspected_pno, page_boxes)
                st.image(annotated_img, caption=f"মূল পৃষ্ঠা {inspected_pno + 1} (শনাক্তকৃত বক্স: {len(page_boxes)}টি)", use_container_width=True)

            with ed_c2:
                st.markdown("##### বক্স ফাইন-টিউন ও সংশোধন")
                if page_boxes:
                    box_options = [idx for idx, _ in page_boxes]
                    sel_box_idx = st.selectbox(
                        "বক্স নম্বর",
                        options=box_options,
                        format_func=lambda idx: f"বক্স #{idx + 1}",
                        key="single_box_sel"
                    )
                    cur_pno, cur_rect = edited_questions[sel_box_idx]
                    pw, ph = src_doc[cur_pno].rect.width, src_doc[cur_pno].rect.height

                    new_x0 = st.number_input("বাম (X0)", value=float(round(cur_rect.x0, 1)), step=2.0, key=f"x0_{sel_box_idx}")
                    new_y0 = st.number_input("উপর (Y0)", value=float(round(cur_rect.y0, 1)), step=2.0, key=f"y0_{sel_box_idx}")
                    new_x1 = st.number_input("ডান (X1)", value=float(round(cur_rect.x1, 1)), step=2.0, key=f"x1_{sel_box_idx}")
                    new_y1 = st.number_input("নিচ (Y1)", value=float(round(cur_rect.y1, 1)), step=2.0, key=f"y1_{sel_box_idx}")

                    b_act1, b_act2 = st.columns(2)
                    with b_act1:
                        if st.button("✓ বক্স আপডেট", key=f"upd_box_{sel_box_idx}", use_container_width=True):
                            edited_questions[sel_box_idx] = (
                                cur_pno,
                                fitz.Rect(
                                    max(0.0, new_x0),
                                    max(0.0, new_y0),
                                    min(pw, max(new_x0 + 20.0, new_x1)),
                                    min(ph, max(new_y0 + 10.0, new_y1))
                                )
                            )
                            st.session_state["edited_questions"] = edited_questions
                            st.toast(f"বক্স #{sel_box_idx + 1} আপডেট হয়েছে!")
                            st.rerun()
                    with b_act2:
                        if st.button("🗑 বক্স মুছুন", key=f"del_box_{sel_box_idx}", use_container_width=True):
                            edited_questions.pop(sel_box_idx)
                            st.session_state["edited_questions"] = edited_questions
                            st.toast(f"বক্স #{sel_box_idx + 1} মুছে ফেলা হয়েছে!")
                            st.rerun()

                    if st.session_state.get("is_admin_unlocked", False):
                        st.markdown("---")
                        if st.button("🧠 সংশোধন থেকে এআইকে শেখান", use_container_width=True):
                            try:
                                suggestions = analyze_user_corrections(
                                    src_doc,
                                    st.session_state.get("initial_questions", []),
                                    st.session_state.get("edited_questions", [])
                                )
                                if suggestions:
                                    learned_cnt = 0
                                    for sug in suggestions:
                                        cat = sug.get("category", "solution_tags")
                                        val = sug.get("value", "")
                                        if val:
                                            ok, _ = commit_rule_to_kb(cat, val, "Learned from box correction")
                                            if ok:
                                                learned_cnt += 1
                                    st.success(f"নতুন {learned_cnt}টি নিয়ম নলেজ-বেসে যুক্ত হয়েছে!")
                                else:
                                    st.info("নতুন কোনো অজানা ট্যাগ পাওয়া যায়নি।")
                            except Exception as e:
                                st.warning(f"লার্নিং ইঞ্জিন বার্তা: {e}")
                else:
                    st.info("এই পৃষ্ঠায় কোনো বক্স নেই।")

        with tab_xray:
            xray_data = st.session_state.get("xray_logs", [])
            if XRAY_MODE and xray_data:
                st.dataframe(pd.DataFrame(xray_data), use_container_width=True, height=420)
            else:
                st.info("এক্স-রে লগ দেখতে বাম পাশের সাইডবার থেকে 'এক্স-রে মোড' চালু করে পুনরায় স্ক্যান করুন।")
    else:
        st.warning("⚠️ কোনো প্রশ্ন শনাক্ত হয়নি। পৃষ্ঠা নম্বর বা মোড পরিবর্তন করে পুনরায় চেষ্টা করুন।")
else:
    st.info("👆 শুরু করতে আপনার এমসিকিউ বা প্রশ্নপত্রের পিডিএফ ফাইলটি উপরে আপলোড করুন।")
