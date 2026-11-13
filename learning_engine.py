# -*- coding: utf-8 -*-
"""
================================================================================
learning_engine.py — ডুপ্লিকেট-মুক্ত রিইনফোর্সমেন্ট লার্নিং এবং গিটহাব সিঙ্ক ইঞ্জিন
================================================================================
"""

import os
import re
import json
import base64
import requests
import fitz
import cv2
import numpy as np
import pytesseract
import streamlit as st

KB_FILENAME = "knowledge_base.json"
DEFAULT_REPO = "axolest-droid/just-test"
DEFAULT_BRANCH = "main"

CATEGORY_LABELS_BN = {
    "question_tags": "প্রশ্ন ট্যাগ (Question Tag)",
    "solution_tags": "সমাধান ট্যাগ (Solution Tag)",
    "header_noise_patterns": "জলছাপ/হেডার (Header Noise)",
    "bullet_stop_chars": "বুলেট চিহ্ন (Bullet Stop)",
    "valid_crop_keywords": "বৈধ প্রশ্নের শব্দ (Valid Keyword)"
}

EXCLUSIVE_CATEGORIES = ["question_tags", "solution_tags", "header_noise_patterns"]


def get_default_kb():
    return {
        "question_tags": [
            "Q", "Question", "Ques", "Prob", "Problem", "Ex", "Example",
            "প্রশ্ন", "এমসিকিউ", "সমস্যা", "উদাহরণ", "❑", "▫", "▪", "■", "□"
        ],
        "solution_tags": [
            "Sol", "Soln", "Solⁿ", "Sol^n", "So1", "So1n", "S0l", "S0ln", "5ol", "5oln",
            "Solution", "Ans", "Answer", "Shortcut", "Hint", "Hints", "Note", "[Note", "[Note:",
            "উত্তর", "উত্ত", "উওর", "উত্তব", "উত্তন", "উত্তম", "ডত্তর", "ডওর", "ডত্তব", "ত্তর", "ত্তব", "উঃ", "উ:", "ত্তঃ", "ত্ত:",
            "সমাধান", "সমাধা", "মাধান", "সলভ", "সল্‌ভ",
            "ব্যাখ্যা", "ব্যখ্যা", "ব্যাখা", "ব্যখা", "্যাখ্যা", "াখ্যা", "রাখ্যা", "য্যাখ্যা", "বাখ্যা", "ব্যাখ",
            "বিকল্প", "িকল্প", "শর্টকাট",
            "প্র্যাকটিস প্রবলেম", "প্র্যাকটিস", "প্রশ্ন-সলভ", "প্রশ্ন-সল্‌ভ",
            "Written", "লিখিত", "বিগত ভর্তি", "DËi:", "DËi", "mgvavb", "e¨vL¨v"
        ],
        "header_noise_patterns": [
            "Educationalpdf", "Educationaf", "Rhombus\\s*Publications",
            "পরিমাণগত\\s*রসায়ন", "ACS\\s*Varsity", "ACS\\s*Chemistry", "ACS\\s*Cliemistry"
        ],
        "bullet_stop_chars": ["➢", "➤", "➣", "•", "♦", "►"],
        "option_units": ["mol", "L", "g", "kg", "mL", "ml", "cm", "mm", "m", "gm", "sec", "min", "s", "×", "x", "10^", "10"],
        "dotless_question_keywords": [
            "কত", "কোনটি", "হচ্ছে", "থাকে", "থাকবে", "রয়েছে", "রয়েছে", "আছে",
            "ওজন", "ভর", "সংখ্যা", "আয়তন", "আয়তন", "মোল", "লিটার", "গ্রাম",
            "গ্যাস", "পরমাণু", "অণু", "STP", "NTP"
        ],
        "valid_crop_keywords": [
            "কত", "কোনটি", "কোন", "হচ্ছে", "থাকে", "থাকবে", "রয়েছে", "রয়েছে", "আছে",
            "নির্ণয়", "নির্ণয়", "ওজন", "ভর", "সংখ্যা", "আয়তন", "আয়তন", "ঘনমাত্রা",
            "মোল", "লিটার", "গ্যাস", "পরমাণু", "অণু", "বিক্রিয়া", "বিক্রিয়া", "লেখচিত্র",
            "what", "which", "find", "calculate", "value"
        ],
        "equation_question_guard_words": [
            "কত", "কোনটি", "হলে", "যদি", "নির্ণয়", "নির্ণয়", "সমান", "অর্ধেক", "দ্বিগুণ"
        ],
        "layout_thresholds": {
            "max_question_number": 160,
            "min_start_gap_pt": 16.0,
            "max_q_x_offset": 52.0,
            "max_sol_x_offset": 120.0,
            "max_internal_blank_gap_pt": 24.0
        }
    }


def _normalize_tag_key(val):
    """তুলনা করার জন্য ট্যাগ থেকে অতিরিক্ত স্ল্যাশ ও স্পেস সরিয়ে ছোট হাতের অক্ষরে রূপান্তর করে।"""
    s = str(val).strip()
    s = re.sub(r"\\([.\:\-\(\)\[\]\s])", r"\1", s)
    return s.strip().lower()


def _dedup_list_preserve_order(items):
    """একই তালিকার ভেতর থেকে ডুপ্লিকেট ট্যাগ (Case-insensitive) বাদ দেয়।"""
    seen = set()
    out = []
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        k = _normalize_tag_key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _read_txt_lines(filename):
    if not os.path.exists(filename):
        return []
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def load_knowledge_base():
    kb = get_default_kb()
    if os.path.exists(KB_FILENAME):
        try:
            with open(KB_FILENAME, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for k, v in loaded.items():
                if isinstance(v, list):
                    kb[k] = _dedup_list_preserve_order(v)
                elif isinstance(v, dict) and k in kb:
                    kb[k].update(v)
                else:
                    kb[k] = v
        except Exception:
            pass

    # টেক্সট ফাইলে ম্যানুয়ালি নতুন কোনো ট্যাগ যোগ করা হলে তা গ্রহণ করা
    txt_q = _read_txt_lines("question_tags.txt")
    txt_s = _read_txt_lines("solution_tags.txt")
    if txt_q:
        kb["question_tags"] = _dedup_list_preserve_order(txt_q + kb.get("question_tags", []))
    if txt_s:
        kb["solution_tags"] = _dedup_list_preserve_order(txt_s + kb.get("solution_tags", []))

    for k in list(kb.keys()):
        if isinstance(kb[k], list):
            kb[k] = _dedup_list_preserve_order(kb[k])

    return kb


def find_all_tag_conflicts(kb=None):
    """
    সবগুলো ফাইল চেক করে দেখে কোনো ট্যাগ একাধিক ক্যাটাগরিতে আছে কি না
    অথবা ফাইলগুলোর ভেতরে কোনো ডুপ্লিকেট লাইন আছে কি না।
    """
    if kb is None:
        kb = load_knowledge_base()

    key_to_cats = {}
    key_to_display = {}

    for cat in EXCLUSIVE_CATEGORIES:
        for item in kb.get(cat, []):
            nk = _normalize_tag_key(item)
            if not nk:
                continue
            key_to_cats.setdefault(nk, []).append(cat)
            if nk not in key_to_display:
                key_to_display[nk] = re.sub(r"\\([.\:\-\(\)\[\]\s])", r"\1", str(item).strip())

    conflicts = []
    for nk, cats in key_to_cats.items():
        unique_cats = list(dict.fromkeys(cats))
        if len(unique_cats) > 1:
            conflicts.append({
                "norm_key": nk,
                "display": key_to_display[nk],
                "categories": unique_cats
            })

    # কাঁচা ফাইলে অভ্যন্তরীণ ডুপ্লিকেট বা অসামঞ্জস্য আছে কি না তা গণনা করা
    internal_dup_count = 0
    for fn, cat_k in [("question_tags.txt", "question_tags"), ("solution_tags.txt", "solution_tags")]:
        raw_lines = _read_txt_lines(fn)
        dedup_lines = _dedup_list_preserve_order(raw_lines)
        internal_dup_count += max(0, len(raw_lines) - len(dedup_lines))

    return conflicts, internal_dup_count


def check_tag_conflict(category, new_value, kb=None):
    """একটি নতুন ট্যাগ অন্য কোনো ক্যাটাগরিতে আগে থেকেই আছে কি না তা পরীক্ষা করে।"""
    if kb is None:
        kb = load_knowledge_base()

    nk = _normalize_tag_key(new_value)
    other_cats = []
    already_in_same = False

    if category in EXCLUSIVE_CATEGORIES:
        for cat in EXCLUSIVE_CATEGORIES:
            existing_keys = {_normalize_tag_key(x) for x in kb.get(cat, [])}
            if nk in existing_keys:
                if cat == category:
                    already_in_same = True
                else:
                    other_cats.append(cat)
    else:
        existing_keys = {_normalize_tag_key(x) for x in kb.get(category, [])}
        if nk in existing_keys:
            already_in_same = True

    return {
        "conflict": len(other_cats) > 0,
        "other_cats": other_cats,
        "already_in_same": already_in_same
    }


def _get_github_credentials():
    token = None
    repo = DEFAULT_REPO
    branch = DEFAULT_BRANCH
    try:
        if "GITHUB_TOKEN" in st.secrets:
            token = st.secrets["GITHUB_TOKEN"]
        if "GITHUB_REPO" in st.secrets:
            repo = st.secrets["GITHUB_REPO"]
        if "GITHUB_BRANCH" in st.secrets:
            branch = st.secrets["GITHUB_BRANCH"]
    except Exception:
        pass

    if not token:
        token = os.environ.get("GITHUB_TOKEN", "")
    return token.strip() if token else None, repo.strip(), branch.strip()


def verify_admin_pin(input_pin):
    secret_pin = None
    try:
        if "ADMIN_PIN" in st.secrets:
            secret_pin = str(st.secrets["ADMIN_PIN"]).strip()
    except Exception:
        pass

    if not secret_pin:
        secret_pin = os.environ.get("ADMIN_PIN", "").strip()

    if not secret_pin:
        secret_pin = "1234"

    return str(input_pin).strip() == secret_pin


def push_file_to_github(filepath, content_str, commit_message):
    token, repo, branch = _get_github_credentials()
    if not token:
        return False, "গিটহাব টোকেন পাওয়া যায়নি (শুধুমাত্র লোকাল সেশনে সেভ হয়েছে)।"

    api_url = f"https://api.github.com/repos/{repo}/contents/{filepath}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    try:
        sha = None
        get_resp = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=10)
        if get_resp.status_code == 200:
            sha = get_resp.json().get("sha")

        encoded_content = base64.b64encode(content_str.encode("utf-8")).decode("ascii")
        payload = {
            "message": commit_message,
            "content": encoded_content,
            "branch": branch
        }
        if sha:
            payload["sha"] = sha

        put_resp = requests.put(api_url, headers=headers, json=payload, timeout=12)
        if put_resp.status_code in (200, 201):
            return True, f"গিটহাব রিপোজিটরিতে ({repo}) স্থায়ীভাবে সেভ হয়েছে!"
        else:
            err_msg = put_resp.json().get("message", f"HTTP {put_resp.status_code}")
            return False, f"গিটহাব এপিআই ত্রুটি: {err_msg}"
    except Exception as e:
        return False, f"গিটহাব সংযোগ ত্রুটি: {e}"


def sync_all_kb_files(kb, commit_note="Sync and deduplicate knowledge base"):
    """
    knowledge_base.json, question_tags.txt এবং solution_tags.txt ফাইল তিনটি
    ডুপ্লিকেটমুক্ত করে একসাথে লোকালি ও গিটহাবে সেভ করে।
    """
    for k in list(kb.keys()):
        if isinstance(kb[k], list):
            kb[k] = _dedup_list_preserve_order(kb[k])

    json_str = json.dumps(kb, ensure_ascii=False, indent=2)
    q_txt = "# নিচে প্রতি লাইনে একটি করে প্রশ্ন শুরুর ট্যাগ লিখুন\n" + "\n".join(kb.get("question_tags", [])) + "\n"
    s_txt = "# নিচে প্রতি লাইনে একটি করে সমাধান বা বাদ দেওয়ার ট্যাগ লিখুন\n" + "\n".join(kb.get("solution_tags", [])) + "\n"

    try:
        with open(KB_FILENAME, "w", encoding="utf-8") as f:
            f.write(json_str)
        with open("question_tags.txt", "w", encoding="utf-8") as f:
            f.write(q_txt)
        with open("solution_tags.txt", "w", encoding="utf-8") as f:
            f.write(s_txt)
    except Exception:
        pass

    push_file_to_github("question_tags.txt", q_txt, f"{commit_note} [sync question_tags.txt]")
    push_file_to_github("solution_tags.txt", s_txt, f"{commit_note} [sync solution_tags.txt]")
    ok, msg = push_file_to_github(KB_FILENAME, json_str, commit_note)
    return ok, msg


def resolve_tag_conflicts(resolutions_map, commit_note="Resolved duplicate tag conflicts"):
    """
    ব্যবহারকারীর নির্বাচন অনুযায়ী ডুপ্লিকেট ট্যাগগুলো নির্দিষ্ট একটি ক্যাটাগরিতে রেখে
    বাকি সব ক্যাটাগরি থেকে মুছে ফাইলগুলো ডুপ্লিকেটমুক্ত করে।
    """
    kb = load_knowledge_base()

    for norm_k, chosen_cat in resolutions_map.items():
        raw_display = None
        for cat in EXCLUSIVE_CATEGORIES:
            new_list = []
            for item in kb.get(cat, []):
                if _normalize_tag_key(item) == norm_k:
                    if raw_display is None:
                        raw_display = re.sub(r"\\([.\:\-\(\)\[\]\s])", r"\1", str(item).strip())
                else:
                    new_list.append(item)
            kb[cat] = new_list

        if chosen_cat in EXCLUSIVE_CATEGORIES and raw_display:
            val_to_add = re.escape(raw_display) if chosen_cat == "header_noise_patterns" else raw_display
            kb[chosen_cat].append(val_to_add)

    return sync_all_kb_files(kb, commit_note)


def commit_rule_to_kb(category, new_value, commit_note="Auto-learn new PDF pattern", force_move=False):
    val = str(new_value).strip()
    if not val:
        return False, "ফাঁকা মান সেভ করা যাবে না।"

    kb = load_knowledge_base()
    if category not in kb or not isinstance(kb[category], list):
        return False, f"অজানা ক্যাটাগরি: {category}"

    raw_clean = re.sub(r"\\([.\:\-\(\)\[\]\s])", r"\1", val).strip()
    nk = _normalize_tag_key(raw_clean)

    conflict_info = check_tag_conflict(category, raw_clean, kb)
    if conflict_info["conflict"] and not force_move:
        other_names = ", ".join(CATEGORY_LABELS_BN.get(c, c) for c in conflict_info["other_cats"])
        other_keys = ",".join(conflict_info["other_cats"])
        return (
            False,
            f"CONFLICT|{other_keys}|⚠️ '{raw_clean}' ট্যাগটি ইতিমধ্যে [{other_names}]-এ আছে! আপনি কি এটিকে কোন ক্যাটাগরিতে রাখতে চান তা নির্বাচন করবেন?"
        )

    # যদি force_move=True হয়, তবে অন্য সব সাংঘর্ষিক ক্যাটাগরি থেকে আগে মুছে ফেলা হবে
    if category in EXCLUSIVE_CATEGORIES:
        for other_c in EXCLUSIVE_CATEGORIES:
            if other_c != category:
                kb[other_c] = [x for x in kb.get(other_c, []) if _normalize_tag_key(x) != nk]

    val_to_save = re.escape(raw_clean) if category == "header_noise_patterns" else raw_clean
    if not any(_normalize_tag_key(x) == nk for x in kb[category]):
        kb[category].append(val_to_save)
    elif not force_move:
        return True, f"'{raw_clean}' ইতিমধ্যে '{CATEGORY_LABELS_BN.get(category, category)}'-এ যুক্ত আছে।"

    return sync_all_kb_files(kb, f"{commit_note} [{category}: {raw_clean}]")


def _extract_lines_from_clip(page, clip_rect):
    words = []
    for wx0, wy0, wx1, wy1, txt, *_ in page.get_text("words"):
        if clip_rect.x0 - 3 <= wx0 <= clip_rect.x1 + 3 and clip_rect.y0 - 3 <= wy0 <= clip_rect.y1 + 3:
            if txt.strip():
                words.append((wx0, wy0, wx1, wy1, txt.strip()))

    if not words:
        try:
            sc = 2.5
            pix = page.get_pixmap(matrix=fitz.Matrix(sc, sc), clip=clip_rect, colorspace=fitz.csGRAY, alpha=False)
            g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
            if g.mean() < 110:
                g = 255 - g
            bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 13)
            data = pytesseract.image_to_data(bw, lang="eng+ben", config="--psm 6", output_type=pytesseract.Output.DICT)
            for i in range(len(data["text"])):
                t = data["text"][i].strip()
                conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip("-").isdigit() else -1
                if t and conf >= 15:
                    wx0 = clip_rect.x0 + data["left"][i] / sc
                    wy0 = clip_rect.y0 + data["top"][i] / sc
                    wx1 = wx0 + data["width"][i] / sc
                    wy1 = wy0 + data["height"][i] / sc
                    words.append((wx0, wy0, wx1, wy1, t))
        except Exception:
            pass

    if not words:
        return []

    sorted_w = sorted(words, key=lambda w: ((w[1] + w[3]) * 0.5, w[0]))
    lines = []
    cur = [sorted_w[0]]
    cur_cy = (sorted_w[0][1] + sorted_w[0][3]) * 0.5
    for w in sorted_w[1:]:
        cy = (w[1] + w[3]) * 0.5
        if abs(cy - cur_cy) <= 6.5:
            cur.append(w)
            cur_cy = sum((x[1] + x[3]) * 0.5 for x in cur) / len(cur)
        else:
            cur.sort(key=lambda x: x[0])
            lines.append(" ".join(x[4] for x in cur))
            cur = [w]
            cur_cy = cy
    cur.sort(key=lambda x: x[0])
    lines.append(" ".join(x[4] for x in cur))
    return [ln.strip() for ln in lines if ln.strip()]


def _rect_iou(r1, r2):
    inter = r1 & r2
    if inter.is_empty:
        return 0.0
    union = r1.get_area() + r2.get_area() - inter.get_area()
    return inter.get_area() / max(1.0, union)


def _sort_page_rects_for_numbering(rects, W, H):
    if W > 1.2 * H:
        rects_sorted = sorted(rects, key=lambda r: (r.y0, r.x0))
        rows, out = [], []
        for r in rects_sorted:
            if rows and abs(r.y0 - rows[-1][0].y0) < 28:
                rows[-1].append(r)
            else:
                rows.append([r])
        for row in rows:
            out.extend(sorted(row, key=lambda r: r.x0))
        return out
    return sorted(
        rects,
        key=lambda q: (0 if (q.width > 0.6 * W or (q.x0 + q.x1) / 2.0 < W / 2.0) else 1, q.y0)
    )


def analyze_user_corrections(raw_pdf, old_rows_df, new_rows_df):
    suggestions = []
    if old_rows_df is None or new_rows_df is None:
        return suggestions

    kb = load_knowledge_base()
    existing_q = {_normalize_tag_key(t) for t in kb.get("question_tags", [])}
    existing_s = {_normalize_tag_key(t) for t in kb.get("solution_tags", [])}
    existing_noise = {_normalize_tag_key(t) for t in kb.get("header_noise_patterns", [])}

    doc = fitz.open(stream=raw_pdf, filetype="pdf")

    old_raw_by_page = {}
    for _, r in old_rows_df.iterrows():
        if bool(r.get("keep", True)):
            p = int(r["page"]) - 1
            old_raw_by_page.setdefault(p, []).append(fitz.Rect(float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])))

    new_raw_by_page = {}
    for _, r in new_rows_df.iterrows():
        if bool(r.get("keep", True)):
            p = int(r["page"]) - 1
            new_raw_by_page.setdefault(p, []).append(fitz.Rect(float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])))

    all_pages = sorted(set(old_raw_by_page.keys()) | set(new_raw_by_page.keys()))

    old_by_page = {}
    old_serial = 1
    for pno in sorted(old_raw_by_page.keys()):
        if 0 <= pno < len(doc):
            W, H = doc[pno].rect.width, doc[pno].rect.height
            sorted_olds = _sort_page_rects_for_numbering(old_raw_by_page[pno], W, H)
            old_by_page[pno] = []
            for r in sorted_olds:
                old_by_page[pno].append((old_serial, r))
                old_serial += 1

    new_by_page = {}
    new_serial = 1
    for pno in sorted(new_raw_by_page.keys()):
        if 0 <= pno < len(doc):
            W, H = doc[pno].rect.width, doc[pno].rect.height
            sorted_news = _sort_page_rects_for_numbering(new_raw_by_page[pno], W, H)
            new_by_page[pno] = []
            for r in sorted_news:
                new_by_page[pno].append((new_serial, r))
                new_serial += 1

    for pno in all_pages:
        if not (0 <= pno < len(doc)):
            continue
        page = doc[pno]
        olds = old_by_page.get(pno, [])
        news = new_by_page.get(pno, [])

        matched_old = set()

        for new_box_no, nr in news:
            best_i, best_iou = -1, 0.0
            for i, (_old_no, or_rect) in enumerate(olds):
                iou = _rect_iou(nr, or_rect)
                horiz_overlap = min(nr.x1, or_rect.x1) - max(nr.x0, or_rect.x0)
                if horiz_overlap > 0.6 * min(nr.width, or_rect.width) and abs(nr.y0 - or_rect.y0) < 15.0:
                    iou = max(iou, 0.55)
                if iou > best_iou:
                    best_iou, best_i = iou, i

            if best_iou < 0.25:
                top_strip = fitz.Rect(nr.x0, nr.y0, nr.x1, min(nr.y1, nr.y0 + 28.0))
                lines = _extract_lines_from_clip(page, top_strip)
                if lines:
                    first_line = lines[0]
                    first_tok = first_line.split()[0].strip("([{.।:;,-")
                    cleaned_prefix = re.sub(r"[0-9০-৯]+.*$", "", first_tok).strip(".:-()[] ")
                    nk = _normalize_tag_key(cleaned_prefix)
                    if cleaned_prefix and len(cleaned_prefix) <= 18 and nk not in existing_q:
                        suggestions.append({
                            "type": "question_tags",
                            "pattern": cleaned_prefix,
                            "page": pno + 1,
                            "box_no": new_box_no,
                            "sample": first_line[:65],
                            "prompt": f"পৃষ্ঠা {pno + 1}-এর **বক্স #{new_box_no}**-এ নতুন প্রশ্নের বক্স যোগ করা হয়েছে যার শুরুতে **'{cleaned_prefix}'** আছে।"
                        })
                        existing_q.add(nk)

            elif best_i != -1:
                matched_old.add(best_i)
                _old_no, or_rect = olds[best_i]
                if abs(nr.y0 - or_rect.y0) <= 12.0 and (or_rect.y1 - nr.y1) >= 12.0:
                    trimmed_rect = fitz.Rect(or_rect.x0, nr.y1 - 2.0, or_rect.x1, min(or_rect.y1, nr.y1 + 32.0))
                    lines = _extract_lines_from_clip(page, trimmed_rect)
                    if lines:
                        first_cut_line = lines[0]
                        first_tok = first_cut_line.split()[0].strip("([{•*-➢➤.@©\"'` ").rstrip(".:;ঃ-)]} ")
                        nk = _normalize_tag_key(first_tok)
                        if first_tok and 1 <= len(first_tok) <= 20 and nk not in existing_s:
                            suggestions.append({
                                "type": "solution_tags",
                                "pattern": first_tok,
                                "page": pno + 1,
                                "box_no": new_box_no,
                                "sample": first_cut_line[:65],
                                "prompt": f"পৃষ্ঠা {pno + 1}-এর **বক্স #{new_box_no}**-এর নিচের অংশ ছেঁটে ফেলা হয়েছে যা **'{first_tok}'** দিয়ে শুরু হয়েছিল।"
                            })
                            existing_s.add(nk)

        for i, (old_box_no, or_rect) in enumerate(olds):
            if i not in matched_old:
                any_overlap = any(_rect_iou(or_rect, nr) > 0.2 for _n_no, nr in news)
                if not any_overlap:
                    lines = _extract_lines_from_clip(page, or_rect)
                    if lines and len(lines) <= 2:
                        noise_candidate = lines[0].strip()
                        nk = _normalize_tag_key(noise_candidate)
                        if 3 <= len(noise_candidate) <= 45 and nk not in existing_noise:
                            suggestions.append({
                                "type": "header_noise_patterns",
                                "pattern": re.escape(noise_candidate),
                                "display_pattern": noise_candidate,
                                "page": pno + 1,
                                "box_no": old_box_no,
                                "sample": noise_candidate,
                                "prompt": f"পৃষ্ঠা {pno + 1}-এর **পুরোনো বক্স #{old_box_no}** (**'{noise_candidate}'**) মুছে ফেলা হয়েছে।"
                            })
                            existing_noise.add(nk)

    doc.close()
    return suggestions
