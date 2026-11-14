# -*- coding: utf-8 -*-
"""
vision_bridge.py  --  app.py <-> question_vision.py bridge.

app.py works with (page_index, fitz.Rect) boxes.  The vision engine also gives a *straightened* picture of
every question (photo tilt / page curl removed).  We hand app.py the boxes AND keep the straightened pictures,
so the practice sheet uses the clean picture as long as the box was not moved in the editor.
"""
import json, os
import numpy as np, cv2

PROFILE_FILE = "vision_profile.json"

def load_profile(path=PROFILE_FILE):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        return {"read_numbers": "none"}

def box_key(pno, x0, y0, x1, y1):
    """same rounding the editor table uses, so an untouched box finds its picture again."""
    return (int(pno), round(float(x0), 1), round(float(y0), 1), round(float(x1), 1), round(float(y1), 1))

def run_on_images(grays, zooms, sizes_pt, page_numbers, profile=None, progress=None):
    """pure part (no fitz): grays = uint8 page images, zooms = px per pt.
    returns (items, note) ; item = dict(pno, rect, png, w_pt, h_pt, conf, flags, num)"""
    from question_vision import process_pages
    prof = dict(profile or load_profile()); prof.pop("target_page_width_px", None)
    results = process_pages(grays, prof, 1, progress)
    items = []
    for (qs, _dbg), z, (Wpt, Hpt), pno in zip(results, zooms, sizes_pt, page_numbers):
        for q in qs:
            x0, y0, x1, y1 = [float(v) / z for v in q.rect_px]
            x0, y0 = max(0.0, x0), max(0.0, y0); x1, y1 = min(Wpt, x1), min(Hpt, y1)
            if x1 - x0 < 18 or y1 - y0 < 6: continue
            ok, buf = cv2.imencode(".png", q.crop)
            if not ok: continue
            items.append(dict(pno=pno, rect=(round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
                              png=buf.tobytes(), w_pt=q.crop.shape[1] / z, h_pt=q.crop.shape[0] / z,
                              conf=q.conf, flags=q.flags, num=q.num))
    weak = [f"পৃষ্ঠা {it['pno'] + 1} / বক্স #{i + 1}" for i, it in enumerate(items) if it["conf"] < 0.9]
    note = f"ভিশন ইঞ্জিন: {len(items)} টি প্রশ্ন পাওয়া গেছে।"
    if weak:
        note += " দেখে নেওয়া ভালো: " + ", ".join(weak[:15]) + (" ..." if len(weak) > 15 else "")
    return items, note

def extract(src_doc, page_indices, fitz, progress=None, profile=None):
    """returns (questions[(pno, fitz.Rect)], imgs{key:(png, w_pt, h_pt)}, note)"""
    prof = dict(profile or load_profile())
    target_w = float(prof.get("target_page_width_px", 1272))
    grays, zooms, sizes, pnos = [], [], [], []
    for pno in page_indices:
        page = src_doc[pno]; W, H = float(page.rect.width), float(page.rect.height)
        z = target_w / max(1.0, W)
        pix = page.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY, alpha=False)
        grays.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy())
        zooms.append(z); sizes.append((W, H)); pnos.append(pno)
    items, note = run_on_images(grays, zooms, sizes, pnos, prof, progress)
    questions, imgs = [], {}
    for it in items:
        x0, y0, x1, y1 = it["rect"]
        questions.append((it["pno"], fitz.Rect(x0, y0, x1, y1)))
        imgs[box_key(it["pno"], x0, y0, x1, y1)] = (it["png"], it["w_pt"], it["h_pt"])
    return questions, imgs, note
