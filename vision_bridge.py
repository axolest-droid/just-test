# -*- coding: utf-8 -*-
"""
vision_bridge.py  --  app.py <-> question_vision.py স্মার্ট হাইব্রিড ব্রিজ।

১. ডিজিটাল টেক্সট পিডিএফ (যেমন s5.pdf) শনাক্ত করলে ভিশন ইঞ্জিন বাইপাস করে
   app.py-এর নেটিভ টেক্সট-লেয়ার ইঞ্জিনে পাঠায়।
২. ফটো বা স্ক্যান করা বইয়ের পেজে (Rhombus: s2, s3, s4 এবং Udvash: s1, s6)
   স্বয়ংক্রিয়ভাবে সঠিক লেআউট মোড নির্বাচন করে প্রশ্ন ও সোজা করা ছবি রিটার্ন করে।
"""
import json
import os
import re
import numpy as np
import cv2

PROFILE_FILE = "vision_profile.json"


def load_profile(path=PROFILE_FILE):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        return {"read_numbers": "none", "mode": "auto"}


def box_key(pno, x0, y0, x1, y1):
    """এডিটর টেবিলের মতো একই রাউন্ডিং ব্যবহার করে যাতে আনটাচড বক্সের ছবি খুঁজে পাওয়া যায়।"""
    return (int(pno), round(float(x0), 1), round(float(y0), 1), round(float(x1), 1), round(float(y1), 1))


def is_digital_text_page(page):
    """
    পৃষ্ঠাটি s5.pdf-এর মতো খাঁটি ডিজিটাল টেক্সট পেজ কি না তা পরীক্ষা করে।
    যদি পৃষ্ঠায় কোনো বড় ব্যাকগ্রাউন্ড স্ক্যান ইমেজ না থাকে এবং পর্যাপ্ত ইউনিকোড টেক্সট থাকে,
    তবে নেটিভ টেক্সট ইঞ্জিন সবচেয়ে নিখুঁত কাজ করে।
    """
    try:
        words = page.get_text("words")
        if len(words) < 35:
            return False
        imgs = page.get_images(full=True)
        if not imgs:
            return True
        W, H = float(page.rect.width), float(page.rect.height)
        page_area = max(1.0, W * H)
        for im in imgs:
            xref = im[0]
            rects = page.get_image_rects(xref)
            for r in rects:
                if (r.width * r.height) > 0.45 * page_area:
                    return False
        return True
    except Exception:
        return False


def run_on_images(grays, zooms, sizes_pt, page_numbers, profile=None, progress=None):
    """pure part (no fitz): grays = uint8 page images, zooms = px per pt."""
    from question_vision import process_pages
    prof = dict(profile or load_profile())
    prof.pop("target_page_width_px", None)
    results = process_pages(grays, prof, 1, progress)
    items = []
    for (qs, _dbg), z, (Wpt, Hpt), pno in zip(results, zooms, sizes_pt, page_numbers):
        for q in qs:
            x0, y0, x1, y1 = [float(v) / z for v in q.rect_px]
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(Wpt, x1), min(Hpt, y1)
            if x1 - x0 < 18 or y1 - y0 < 6:
                continue
            ok, buf = cv2.imencode(".png", q.crop)
            if not ok:
                continue
            items.append(dict(
                pno=pno,
                rect=(round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
                png=buf.tobytes(),
                w_pt=q.crop.shape[1] / z,
                h_pt=q.crop.shape[0] / z,
                conf=q.conf,
                flags=q.flags,
                num=q.num
            ))
    weak = [f"পৃষ্ঠা {it['pno'] + 1} / বক্স #{i + 1}" for i, it in enumerate(items) if it["conf"] < 0.9]
    note = f"ভিশন ইঞ্জিন: {len(items)} টি প্রশ্ন পাওয়া গেছে।"
    if weak:
        note += " দেখে নেওয়া ভালো: " + ", ".join(weak[:15]) + (" ..." if len(weak) > 15 else "")
    return items, note


def extract(src_doc, page_indices, fitz, progress=None, profile=None):
    """returns (questions[(pno, fitz.Rect)], imgs{key:(png, w_pt, h_pt)}, note)"""
    prof = dict(profile or load_profile())
    target_w = float(prof.get("target_page_width_px", 1272))

    # যদি সবগুলো নির্বাচিত পৃষ্ঠা s5.pdf-এর মতো খাঁটি ডিজিটাল টেক্সট পেজ হয়,
    # তবে ভিশন ইঞ্জিন স্কিপ করে নেটিভ টেক্সট-লেয়ার ইঞ্জিনে পাঠানো হবে।
    digital_count = sum(1 for pno in page_indices if is_digital_text_page(src_doc[pno]))
    if digital_count == len(page_indices) and len(page_indices) > 0:
        return [], {}, "ডিজিটাল টেক্সট পিডিএফ শনাক্ত হয়েছে — নেটিভ টেক্সট ইঞ্জিন ব্যবহৃত হয়েছে।"

    grays, zooms, sizes, pnos = [], [], [], []
    for pno in page_indices:
        page = src_doc[pno]
        W, H = float(page.rect.width), float(page.rect.height)
        z = target_w / max(1.0, W)
        pix = page.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY, alpha=False)
        grays.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy())
        zooms.append(z)
        sizes.append((W, H))
        pnos.append(pno)

    items, note = run_on_images(grays, zooms, sizes, pnos, prof, progress)
    questions, imgs = [], {}
    for it in items:
        x0, y0, x1, y1 = it["rect"]
        questions.append((it["pno"], fitz.Rect(x0, y0, x1, y1)))
        imgs[box_key(it["pno"], x0, y0, x1, y1)] = (it["png"], it["w_pt"], it["h_pt"])
    return questions, imgs, note
