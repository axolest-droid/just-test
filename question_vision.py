# -*- coding: utf-8 -*-
"""
question_vision.py  --  Dual-Mode Vision Question Extractor (Rhombus Flush-Margin + Udvash Hanging-Indent).

১. Rhombus Mode (s2, s3, s4):
   পেজ সোজা করে বাম মার্জিনের বারবার আসা চওড়া লেবেল ('উত্তর:', 'ব্যাখ্যা:', 'সমাধান:') চিনে নেয়
   এবং লেবেলের পরের প্রথম ছোট মার্জিন টোকেনকে প্রশ্ন হিসেবে নেয়।
২. Udvash / Hanging-Indent Mode (s1, s6):
   - পৃষ্ঠার উপরের থিওরি টেবিল এবং মাঝখানের পূর্ণ-প্রস্থের সেকশন ব্যানার
     ('বিগত ভর্তি পরীক্ষার MCQ প্রশ্ন-সল্ভ' / 'Written প্রশ্ন-সল্ভ') শনাক্ত করে পেজকে জোনে ভাগ করে।
   - প্রতিটি কলামে হ্যাঙ্গিং আউটডেন্টে থাকা প্রশ্ন নম্বর (01., 02., ..., 41.) এবং ইনলাইনে থাকা প্রশ্ন নম্বর শনাক্ত করে।
   - ইনডেন্টেড মার্জিনে থাকা 'Soln:' / 'Sol":' / 'সমাধান:' লাইনে প্রশ্ন থামিয়ে দেয়।
   - কোনো প্রশ্নে 'Soln:' না থাকলে (যেমন s6-এর 02, 11, 17, 22) পরের প্রশ্নের ঠিক আগে থামিয়ে দেয়।
   - কলামের শুরুতে আগের কলামের বাকি সমাধান থাকলে তা স্বয়ংক্রিয়ভাবে বাদ দেয়।
"""
import re
import random
from dataclasses import dataclass, field
import numpy as np
import cv2
import pytesseract

# --------------------------------------------------------------------------- profile
PROFILE = dict(
    mode='auto',          # 'auto' | 'rhombus' | 'udvash'
    margin_tol=12,        # px a line may sit from the fitted margin and still count as "margin"
    num_max_w=48,         # widest first-token that may still be a number marker
    pad_top=6, pad_bot=4, # px kept above the number / below the last question line
    min_q_h=40,           # smaller boxes are flagged
    max_q_h=720,          # bigger boxes are flagged
    label_thr=0.7, label_frac=.25,   # how alike repeated label glyphs must be / how common a cluster must be
    header_px=60,         # ignore this many px at the top of every column
    footer_px=65,         # ignore this many px at the bottom of every column
    hanging_indent_min_px=13,
    read_numbers='digits',
)

# --------------------------------------------------------------------------- image prep
def normalize(g):
    bg = cv2.medianBlur(cv2.resize(g, None, fx=.25, fy=.25), 31)
    bg = cv2.resize(bg, (g.shape[1], g.shape[0]))
    n = cv2.divide(g, bg, scale=255)
    return n, (n < 190).astype(np.uint8)


def _bands(ink, x0, x1, y0, y1, minh=8):
    prof = ink[y0:y1, x0:x1].sum(1)
    on = prof > max(2, .004 * (x1 - x0))
    out = []
    s = None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        if not v and s is not None:
            if i - s >= minh:
                out.append((s + y0, i + y0))
            s = None
    if s is not None:
        out.append((s + y0, y1))
    return out


def _first_token(ink, y0, y1, x0, x1, gap=9):
    xs = np.where(ink[y0:y1, x0:x1].sum(0) > 0)[0]
    if len(xs) == 0:
        return None
    s = e = xs[0]
    for x in xs[1:]:
        if x - e > gap:
            break
        e = x
    return x0 + s, x0 + e


# --------------------------------------------------------------------------- layout
def find_slanted_divider(ink):
    """(slope, x_top) with divider x(y)=x_top+slope*y, or None for a single column."""
    H, W = ink.shape
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(H * .04))))
    lo, hi = int(W * .3), int(W * .7)
    yy, xx = np.nonzero(v[:, lo - 80:hi + 80])
    xx = xx + lo - 80
    best = (0, None, None)
    if len(xx) > 0:
        for s in np.arange(-0.08, 0.0801, 0.004):
            x0 = np.round(xx - s * yy).astype(int)
            k = np.convolve(np.bincount(x0 - x0.min()), np.ones(5), 'same')
            i = int(np.argmax(k))
            if k[i] > best[0]:
                best = (k[i], s, i + x0.min())
    # যদি পেজের উপরের অর্ধেকে থিওরি বক্স থাকে (যেমন s6 পৃষ্ঠা ১), তবে নিচের অর্ধেকেও ডিভাইডার থাকায় থ্রেশহোল্ড 0.22*H রাখা হয়েছে
    if best[0] >= H * .22 and abs(best[2] - W * 0.5) < W * 0.12:
        return (best[1], best[2])
    return None


def straighten(n, ink, div):
    if div is None:
        return n, ink
    H, W = ink.shape
    M = np.float32([[1, -div[0], 0], [0, 1, 0]])
    return (
        cv2.warpAffine(n, M, (W, H), borderValue=255),
        cv2.warpAffine(ink, M, (W, H), borderValue=0)
    )


def strip_rules(ink):
    """remove the long thin vertical divider and thin horizontal / dotted rules."""
    H, W = ink.shape
    out = ink.copy()
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(H * .08))))
    out &= (cv2.dilate(v, np.ones((1, 3), np.uint8)) == 0).astype(np.uint8)
    j = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)))
    hr = cv2.morphologyEx(j, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * .16), 1)))
    hr &= (cv2.morphologyEx(j, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * .16), 6))) == 0).astype(np.uint8)
    out &= (cv2.dilate(hr, np.ones((3, 1), np.uint8)) == 0).astype(np.uint8)
    return out


def _best_angle(ink, rng=6, step=.25):
    small = cv2.resize(ink.astype(np.float32), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    c = (small.shape[1] / 2, small.shape[0] / 2)
    best = (-1, 0)
    for a in np.arange(-rng, rng + 1e-6, step):
        r = cv2.warpAffine(small, cv2.getRotationMatrix2D(c, a, 1), (small.shape[1], small.shape[0]))
        s = np.sum(np.diff(r.sum(1)) ** 2)
        if s > best[0]:
            best = (s, a)
    return best[1]


def dewarp(arrs, ink_ref, win=380, step=190):
    """first-order page-curl fix: text-line angle is measured every ~200px down the column."""
    H, W = ink_ref.shape
    ys = sorted(set(list(range(0, max(1, H - win), step)) + [max(0, H - win)]))
    cs, th = [], []
    for y in ys:
        seg = ink_ref[y:y + win]
        cs.append(y + win / 2)
        th.append(np.deg2rad(_best_angle(seg) if seg.sum() > 200 else 0.))
    th = np.array(th)
    if len(th) >= 3:
        th = np.array([np.median(th[max(0, i - 1):i + 2]) for i in range(len(th))])
    thy = np.interp(np.arange(H), cs, th)
    xx = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    yy = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    my = (yy + thy[:, None] * (xx - W / 2)).astype(np.float32)
    return [
        cv2.remap(
            a, xx, my,
            cv2.INTER_NEAREST if a.dtype == bool else cv2.INTER_LINEAR,
            borderValue=255 if a.max() > 1 else 0
        ) for a in arrs
    ], thy


def header_end(raw_ink):
    H, W = raw_ink.shape
    j = cv2.dilate(raw_ink, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    r = cv2.morphologyEx(j, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * .45), 1)))
    r &= (cv2.morphologyEx(j, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * .45), 6))) == 0).astype(np.uint8)
    rows = np.where(r[:int(H * .16)].sum(1) > 0)[0]
    rows = rows[rows > 50]
    if len(rows) == 0:
        return 95
    end = rows[0]
    for y in rows[1:]:
        if y - end > 12:
            break
        end = y
    return int(end) + 6


def footer_start(raw_ink):
    """পৃষ্ঠার নিচের ফুটার রেখা বা লোগো শনাক্ত করে তার উপরের সীমানা রিটার্ন করে।"""
    H, W = raw_ink.shape
    j = cv2.dilate(raw_ink, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    r = cv2.morphologyEx(j, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * .40), 1)))
    r &= (cv2.morphologyEx(j, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * .40), 6))) == 0).astype(np.uint8)
    rows = np.where(r[int(H * .88):].sum(1) > 0)[0]
    if len(rows) > 0:
        return int(H * .88) + int(rows[0]) - 6
    return H - 78


# --------------------------------------------------------------------------- margin (Rhombus)
def _fit_margin(pts, tol=9, iters=500, seed=1):
    rnd = random.Random(seed)
    pts = np.array(pts, float)
    best = None
    for _ in range(iters):
        i, j = rnd.sample(range(len(pts)), 2)
        if abs(pts[i, 0] - pts[j, 0]) < 150:
            continue
        a = (pts[j, 1] - pts[i, 1]) / (pts[j, 0] - pts[i, 0])
        if abs(a) > .1:
            continue
        b = pts[i, 1] - a * pts[i, 0]
        res = pts[:, 1] - (a * pts[:, 0] + b)
        sc = (abs(res) <= tol).sum() - 3 * (res < -tol).sum()
        if best is None or sc > best[0]:
            best = (sc, a, b)
    if best is None:
        return 0., float(pts[:, 1].min())
    _, a, b = best
    m = abs(pts[:, 1] - (a * pts[:, 0] + b)) <= tol
    return tuple(np.polyfit(pts[m, 0], pts[m, 1], 1)) if m.sum() >= 3 else (a, b)


def _margin_fn(ink):
    pts = []
    for a, b in _bands(ink, 0, ink.shape[1], 100, ink.shape[0] - 70):
        xs = np.where(ink[a:b].sum(0) > 0)[0]
        if len(xs):
            pts.append(((a + b) / 2, xs[0]))
    if len(pts) < 4:
        x = min([p[1] for p in pts] or [0])
        return (lambda y: x), (0., x)
    a, b = _fit_margin(pts)
    return (lambda y: a * y + b), (a, b)


def _gather(img, mx, lo=-8, hi=76):
    H, W = img.shape
    w = hi - lo
    S = np.zeros((H, w), img.dtype) if img.max() <= 1 else np.full((H, w), 255, img.dtype)
    for y in range(H):
        a = int(round(mx(y))) + lo
        s, e = max(0, a), min(W, a + w)
        if e > s:
            S[y, s - a:e - a] = img[y, s:e]
    return S


# --------------------------------------------------------------------------- digit OCR (tiny crops only)
def _ocr_digits(crop, psm):
    crop = cv2.copyMakeBorder(
        cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC),
        12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255
    )
    d = pytesseract.image_to_data(
        crop,
        config=f'--psm {psm} -c tessedit_char_whitelist=0123456789.',
        output_type=pytesseract.Output.DICT
    )
    return [(t.strip(), int(c)) for t, c in zip(d['text'], d['conf']) if t.strip() and str(c).lstrip('-').isdigit()]


def _variants(crop):
    yield crop
    _, o = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield o
    yield cv2.erode(o, np.ones((2, 2), np.uint8))
    yield cv2.dilate(o, np.ones((2, 2), np.uint8))


def read_number(Sg, y0, y1, x_start, widths=(26, 30, 34, 38, 42, 48)):
    for w in (34, 44):
        crop = Sg[max(0, y0 - 3):y1 + 3, max(0, x_start - 4):x_start + w]
        if crop.size == 0:
            continue
        for t, cf in _ocr_digits(crop, 7):
            m = re.fullmatch(r'(\d{1,3})\.', t)
            if m and cf >= 85:
                return (int(m.group(1)), cf)
    votes = {}
    for w in widths:
        crop = Sg[max(0, y0 - 3):y1 + 3, max(0, x_start - 4):x_start + w]
        if crop.size == 0:
            continue
        for v in _variants(crop):
            for psm in (7, 8):
                for t, cf in _ocr_digits(v, psm):
                    m = re.fullmatch(r'(\d{1,3})(\.?)', t)
                    if m and cf >= 25 and (m.group(2) or len(m.group(1)) >= 2):
                        k = int(m.group(1))
                        votes[k] = votes.get(k, 0) + cf + (15 if m.group(2) else 0) + 10 * (len(m.group(1)) - 1)
    if not votes:
        return None
    k = max(votes, key=votes.get)
    return (k, votes[k])


# --------------------------------------------------------------------------- results
@dataclass
class Marker:
    col: int
    y0: int
    y1: int
    tw: int
    kind: str
    num: int = None
    conf: float = 0
    guessed: bool = False


@dataclass
class Question:
    page: int
    col: int
    num: int
    box: tuple
    crop: np.ndarray = None
    conf: float = 1.0
    flags: list = field(default_factory=list)
    poly: list = None
    rect_px: tuple = None


# --------------------------------------------------------------------------- Rhombus pipeline (s2, s3, s4 - UNTOUCHED)
def _margin_bands(n_c, i_c, raw_c, hdr, ci, P):
    H, Wc = i_c.shape
    mx, _ = _margin_fn(np.where(np.arange(H)[:, None] < hdr, 0, i_c).astype(np.uint8))
    Si = _gather(i_c, mx)
    Sg = _gather(n_c, mx)
    Si[:hdr] = 0
    out = []
    hl = cv2.morphologyEx(raw_c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(Wc * .5), 1)))
    for y0, y1 in _bands(Si, 0, Si.shape[1], 0, H - 60, minh=9):
        xs = np.where(Si[y0:y1].sum(0) > 0)[0]
        if len(xs) == 0 or xs[0] > 8 + P['margin_tol']:
            continue
        ft = _first_token(Si, y0, y1, 0, Si.shape[1])
        tw = ft[1] - ft[0] + 1
        if Si[y0:y1, ft[0]:ft[1] + 1].mean() > .85 and tw >= 14:
            continue
        sigs = []
        rr = np.where(Si[y0:y1, ft[0]:ft[0] + 44].sum(1) > 0)[0]
        r0, r1 = (y0 + rr[0], y0 + rr[-1] + 1) if len(rr) >= 6 else (y0, y1)
        for w in (44, 56, 68):
            patch = Sg[r0:r1, max(0, ft[0] - 2):ft[0] + min(tw, w) + 2]
            sg = cv2.GaussianBlur(cv2.resize(patch, (48, 24), interpolation=cv2.INTER_AREA).astype(np.float32), (3, 3), 0)
            sigs.append((sg - sg.mean()) / (sg.std() + 1e-6))
        out.append(dict(
            col=ci, y0=y0, y1=y1, tw=tw, ft0=ft[0], sigs=sigs,
            box=bool(hl[max(0, y0 - 3):y0 + 8].sum() > 0 and hl[max(0, y1 - 8):y1 + 3].sum() > 0 and y1 - y0 > 14),
            Sg=Sg
        ))
    return out


def learn_labels(all_bands, thr=0.8, min_n=3, frac=.3):
    wide = [b for b in all_bands if b['tw'] > 40 and not b['box']]
    reps = []
    for b in wide:
        for r in reps:
            if max(float((a * c).mean()) for a, c in zip(r['rep'], b['sigs'])) >= thr:
                r['members'].append(b)
                break
        else:
            reps.append(dict(rep=b['sigs'], members=[b]))
    reps = [r for r in reps if len(r['members']) >= min_n]
    if not reps:
        return set()
    top = max(len(r['members']) for r in reps)
    return {id(b) for r in reps if len(r['members']) >= frac * top for b in r['members']}


def _grammar(bands, labels, P):
    out = []
    state = 'start'
    for b in sorted(bands, key=lambda b: b['y0']):
        if b['box']:
            out.append(Marker(b['col'], b['y0'], b['y1'], b['tw'], 'hdr'))
            state = 'start'
            continue
        if id(b) in labels:
            out.append(Marker(b['col'], b['y0'], b['y1'], b['tw'], 'label'))
            state = 'sol'
            continue
        if state in ('start', 'sol') and 8 <= b['tw'] <= P['num_max_w']:
            m = Marker(b['col'], b['y0'], b['y1'], b['tw'], 'num')
            m.band = b
            out.append(m)
            state = 'q'
    return out


def _number(markers, P, start):
    k = start - 1
    prev = None
    for m in sorted([m for m in markers if m.kind == 'num'], key=lambda m: (m.col, m.y0)):
        r = read_number(m.band['Sg'], m.y0, m.y1, m.band['ft0']) if P['read_numbers'] == 'digits' else None
        if r and (prev is None or 0 < r[0] - prev <= 3):
            m.num, m.conf = r[0], r[1]
        else:
            m.num, m.guessed = k + 1, P['read_numbers'] == 'digits'
        k = prev = m.num


def _prepare(gray, P):
    n, ink = normalize(gray)
    div = find_slanted_divider(ink)
    n2, ink2 = straighten(n, ink, div)
    clean = strip_rules(ink2)
    H, W = gray.shape
    cols = [(30, int(div[1]) - 1), (int(div[1]) + 2, W - 30)] if div else [(30, W - 30)]
    cdata, bands, geom, raw_cols = [], [], [], []
    for ci, (x0, x1) in enumerate(cols):
        (n_c, i_c, raw_c), thy = dewarp(
            [n2[:, x0:x1].copy(), clean[:, x0:x1].copy(), ink2[:, x0:x1].copy()],
            clean[:, x0:x1]
        )
        i_c = (i_c > 0).astype(np.uint8)
        raw_c = (raw_c > 0).astype(np.uint8)
        hdr = max(P['header_px'], header_end(raw_c))
        cdata.append((n_c, i_c, hdr))
        raw_cols.append(raw_c)
        geom.append((x0, thy))
        bands.append(_margin_bands(n_c, i_c, raw_c, hdr, ci, P))
    return dict(
        cols=cols, div=div, cdata=cdata, raw_cols=raw_cols, bands=bands,
        geom=geom, shear=(div[0] if div else 0.0), size=gray.shape,
        n2=n2, ink2=ink2
    )


def _questions(page_no, pg, markers, P):
    qs = []
    for ci, (n_c, i_c, hdr) in enumerate(pg['cdata']):
        ms = sorted([m for m in markers if m.col == ci], key=lambda m: m.y0)
        Hc, Wc = i_c.shape
        ftr = footer_start(pg['raw_cols'][ci]) if 'raw_cols' in pg else (Hc - 70)
        for k, m in enumerate(ms):
            if m.kind != 'num':
                continue
            nxt = [x for x in ms[k + 1:]]
            stop = nxt[0].y0 - P['pad_bot'] if nxt else min(Hc - 70, ftr)
            flags = []
            if not nxt or nxt[0].kind != 'label':
                flags.append('no solution label found')
            top = max(0, m.y0 - P['pad_top'])
            bot = max(top + 1, stop)
            # নিচের অতিরিক্ত সাদা জায়গা ছাঁটাই করা
            ink_rows = np.where(i_c[top:bot].sum(1) > 2)[0]
            if len(ink_rows) > 0:
                bot = min(bot, top + int(ink_rows[-1]) + P['pad_bot'] + 2)
            xs = np.where(i_c[top:bot].sum(0) > 0)[0]
            bx0 = max(0, (xs[0] if len(xs) else 0) - 6)
            bx1 = min(Wc, (xs[-1] if len(xs) else Wc) + 8)
            if bot - top < P['min_q_h']:
                flags.append('very small box')
            if bot - top > P['max_q_h']:
                flags.append('very tall box')
            if m.guessed:
                flags.append('number inferred from sequence')
            crop = n_c[top:bot, bx0:bx1].copy()
            soft = []
            if nxt and nxt[0].kind == 'label':
                lb = nxt[0]
                zone0 = int(bx0 + .55 * (bx1 - bx0))
                lim = min(Hc - 1, lb.y1 + 12)
                rowink = i_c[bot:lim, zone0:bx1].sum(1) > 0
                straddle = int((i_c[max(0, bot - 3), zone0:bx1] & i_c[min(Hc - 1, bot + 3), zone0:bx1]).sum())
                if straddle >= 3 and lb.y1 - lb.y0 <= 40 and rowink[:max(1, lb.y1 - bot + 8)].sum() > 2:
                    last = 0
                    blank = 0
                    for r, v in enumerate(rowink):
                        if v:
                            last, blank = r, 0
                        else:
                            blank += 1
                            if blank >= 4 and r > lb.y1 - bot:
                                break
                    nb = min(Hc - 1, bot + last + 6)
                    cols_ink = np.where(i_c[lb.y0:lb.y1, bx0:bx1].sum(0) > 0)[0]
                    e = cols_ink[0] if len(cols_ink) else 0
                    for c in cols_ink[1:]:
                        if c - e > 40:
                            break
                        e = c
                    crop = n_c[top:nb, bx0:bx1].copy()
                    crop[max(0, lb.y0 - 3 - top):lb.y1 + 3 - top, :e + 8] = 255
                    bot = nb
                    soft.append('curved edge: label text masked')
            x0g, thy = pg['geom'][ci]
            poly = []
            for xr, yr in ((bx0, top), (bx1, top), (bx1, bot), (bx0, bot)):
                ys = yr + float(thy[min(len(thy) - 1, max(0, int(yr)))]) * (xr - Wc / 2.0)
                poly.append((xr + x0g + pg['shear'] * ys, ys))
            px = [p[0] for p in poly]
            py = [p[1] for p in poly]
            rect_px = (min(px), min(py), max(px), max(py))
            hard = [f for f in flags if f != 'very small box'] + (['very small box'] if bot - top < P['min_q_h'] // 2 else [])
            qs.append(Question(
                page_no, ci, m.num, (bx0, top, bx1, bot), crop,
                max(.05, 1.0 - .35 * len(hard) - .3 * len(soft)),
                flags + soft, poly, rect_px
            ))
    qs.sort(key=lambda q: (q.col, q.box[1]))
    return qs


# --------------------------------------------------------------------------- Udvash / Hanging-Indent Detector & Extractor (s1, s6)
_UDV_NUM_RE = re.compile(r"^[\(\[]?([0-9০-৯]{1,2})[\.\)\]\:\ঃ](?![0-9০-৯])")
_UDV_SOL_RE = re.compile(
    r"^(?:[\(\[\{•\*\-\.\:\;\,\'\"\`\~\|∀]*)"
    r"(?:[sS58\$৪][oO0aAeE9][lL1Ii\|!t](?:[nNmMrRuU23\^\°\*\~\"\'\`\’\”\ⁿ\-\_\:\%]+)?|"
    r"oln|ol\"|সমাধান|উত্তর|ব্যাখ্যা)(?:$|[\s\:\ঃ\.\;\,\-\(\[\{\"'\`\’\”\ⁿ\^\°\*\/\|])",
    re.IGNORECASE
)
_UDV_BANNER_RE = re.compile(r"(?:বিগত\s*ভর্তি\s*পরীক্ষার|প্রশ্ন[\-\—\–]সল্?ভ|প্রশ্নব্যাংক|অধ্যায়[\-\—\–][০-৯0-9])")


def _detect_fullwidth_banners(ink2, hdr, ftr):
    """
    s6.pdf-এর পৃষ্ঠা ১ ও ৩-এ থাকা পূর্ণ-প্রস্থের সেকশন ব্যানার
    ('বিগত ভর্তি পরীক্ষার MCQ প্রশ্ন-সল্ভ' / 'Written প্রশ্ন-সল্ভ') শনাক্ত করে।
    """
    H, W = ink2.shape
    if ftr <= hdr + 100:
        return []
    # বাম ও ডান উভয় কলাম জুড়ে বিস্তৃত অনুভূমিক বক্স বা ব্যানার খোঁজা
    dil = cv2.dilate(ink2[hdr:ftr, int(W * 0.15):int(W * 0.85)], cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    row_cov = dil.mean(axis=1)
    # যেসব রো-তে বাম, মধ্য (ডিভাইডার অঞ্চল) ও ডান—সব জায়গায় কালি বা বর্ডার আছে
    mid_band = ink2[hdr:ftr, int(W * 0.44):int(W * 0.56)]
    mid_dil = cv2.dilate(mid_band, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))).mean(axis=1)

    banners = []
    in_b = None
    for r in range(len(row_cov)):
        if row_cov[r] > 0.42 and mid_dil[r] > 0.25:
            if in_b is None:
                in_b = r
        else:
            if in_b is not None:
                bh = r - in_b
                if 16 <= bh <= 95:
                    banners.append((hdr + in_b - 4, hdr + r + 4))
                in_b = None
    if in_b is not None and 16 <= (len(row_cov) - in_b) <= 95:
        banners.append((hdr + in_b - 4, hdr + len(row_cov) + 4))
    return banners


def _is_udvash_document(pgs):
    """
    ডকুমেন্টটি উদ্ভাস / হ্যাঙ্গিং-ইনডেন্ট ফরম্যাটের (s1, s6) নাকি রম্বস ফরম্যাটের (s2, s3, s4) তা নির্ণয় করে।
    উদ্ভাস বইয়ে প্রশ্ন নম্বরগুলো (01., 10.) মূল টেক্সট মার্জিন থেকে ১৫-৩৫ পিক্সেল বামে বাইরে থাকে।
    """
    hanging_hits = 0
    total_cols = 0
    for pg in pgs:
        for ci, (n_c, i_c, hdr) in enumerate(pg['cdata']):
            Hc, Wc = i_c.shape
            ftr = footer_start(pg['raw_cols'][ci])
            bs = _bands(i_c, 0, Wc, hdr, ftr, minh=9)
            if len(bs) < 6:
                continue
            total_cols += 1
            x0s = []
            narrow_left = []
            for y0, y1 in bs:
                if y1 - y0 > 42:
                    continue
                ft = _first_token(i_c, y0, y1, 0, Wc, gap=8)
                if ft is None:
                    continue
                tw = ft[1] - ft[0] + 1
                x0s.append(ft[0])
                if 10 <= tw <= 42:
                    narrow_left.append((y0, y1, ft[0], tw))
            if len(x0s) < 6:
                continue
            body_x = float(np.percentile(x0s, 55))
            outdented = [item for item in narrow_left if (body_x - item[2]) >= 13.0 and item[2] < Wc * 0.25]
            if len(outdented) >= 2:
                hanging_hits += 1
    return hanging_hits >= max(1, total_cols // 2)


def _extract_udvash_column_zone(n_c, i_c, y_top, y_bot, ci, P):
    """
    একটি কলামের নির্দিষ্ট জোনের (y_top থেকে y_bot) ভেতর থেকে উদ্ভাস ফরম্যাটের
    প্রশ্ন শুরুর অবস্থান এবং Soln: স্টপ অবস্থান বের করে।
    """
    Hc, Wc = i_c.shape
    if y_bot - y_top < 45:
        return []

    roi_gray = n_c[y_top:y_bot, :]
    roi_ink = i_c[y_top:y_bot, :]

    # ১. ভিজ্যুয়াল লাইন এবং বাম মার্জিনের হ্যাঙ্গিং আউটডেন্ট বিশ্লেষণ
    line_bands = _bands(roi_ink, 0, Wc, 0, y_bot - y_top, minh=9)
    if not line_bands:
        return []

    x0_list = []
    band_infos = []
    for by0, by1 in line_bands:
        ft = _first_token(roi_ink, by0, by1, 0, Wc, gap=8)
        if ft is None:
            continue
        tw = ft[1] - ft[0] + 1
        if by1 - by0 <= 38:
            x0_list.append(ft[0])
        band_infos.append((by0 + y_top, by1 + y_top, ft[0], ft[1], tw))

    body_margin_x = float(np.percentile(x0_list, 55)) if len(x0_list) >= 4 else 55.0
    outer_margin_max = max(32.0, body_margin_x - P.get('hanging_indent_min_px', 13))

    # ২. কলামের ওপর দ্রুত OCR চালিয়ে প্রশ্ন নম্বর (01., 10.) এবং Soln: ট্যাগ শনাক্ত করা
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_gray)
    bw = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 14)
    ocr_d = pytesseract.image_to_data(bw, lang="eng+ben", config="--psm 6", output_type=pytesseract.Output.DICT)

    words = []
    for i in range(len(ocr_d["text"])):
        t = ocr_d["text"][i].strip()
        cf = int(ocr_d["conf"][i]) if str(ocr_d["conf"][i]).lstrip("-").isdigit() else -1
        if not t or cf < 12:
            continue
        wx0 = int(ocr_d["left"][i])
        wy0 = int(ocr_d["top"][i]) + y_top
        wx1 = wx0 + int(ocr_d["width"][i])
        wy1 = wy0 + int(ocr_d["height"][i])
        words.append((wx0, wy0, wx1, wy1, t, cf))

    # বাম স্ট্রিপে শুধুমাত্র ইংরেজি ডিজিট ও Soln এর জন্য হালকা OCR পাস
    strip_w = min(Wc, int(max(body_margin_x + 75, 130)))
    eng_d = pytesseract.image_to_data(bw[:, :strip_w], lang="eng", config="--psm 6", output_type=pytesseract.Output.DICT)
    eng_words = []
    for i in range(len(eng_d["text"])):
        t = eng_d["text"][i].strip()
        cf = int(eng_d["conf"][i]) if str(eng_d["conf"][i]).lstrip("-").isdigit() else -1
        if not t or cf < 14:
            continue
        wx0 = int(eng_d["left"][i])
        wy0 = int(eng_d["top"][i]) + y_top
        wx1 = wx0 + int(eng_d["width"][i])
        wy1 = wy0 + int(eng_d["height"][i])
        eng_words.append((wx0, wy0, wx1, wy1, t, cf))

    def _group_lines(wlist):
        if not wlist:
            return []
        sw = sorted(wlist, key=lambda w: ((w[1] + w[3]) * 0.5, w[0]))
        lns = []
        cur = [sw[0]]
        cy = (sw[0][1] + sw[0][3]) * 0.5
        for w in sw[1:]:
            wcy = (w[1] + w[3]) * 0.5
            if abs(wcy - cy) <= 12.0:
                cur.append(w)
                cy = sum((x[1] + x[3]) * 0.5 for x in cur) / len(cur)
            else:
                cur.sort(key=lambda x: x[0])
                lns.append(cur)
                cur = [w]
                cy = wcy
        cur.sort(key=lambda x: x[0])
        lns.append(cur)
        return lns

    ocr_lines = _group_lines(words)
    eng_lines = _group_lines(eng_words)

    q_starts = []  # (y0, q_num_or_None)
    sol_stops = [] # y0

    # ৩. ভিজ্যুয়াল হ্যাঙ্গিং নম্বর এবং OCR লাইন মিলিয়ে প্রশ্ন শুরু ও Soln স্টপ বের করা
    for by0, by1, fx0, fx1, tw in band_infos:
        bh = by1 - by0
        if bh > 42:
            continue
        # হ্যাঙ্গিং আউটডেন্টে থাকা ছোট টোকেন চেক করা (যেমন 01., 10.)
        if fx0 <= outer_margin_max and 10 <= tw <= 44:
            # পাশে একই লাইনে প্রশ্ন টেক্সট আছে কি না দেখা
            right_ink = i_c[by0:by1, min(Wc, fx1 + 6):min(Wc, fx1 + 190)].sum()
            if right_ink > 25:
                crop = n_c[max(0, by0 - 3):min(Hc, by1 + 3), max(0, fx0 - 4):min(Wc, fx1 + 10)]
                if crop.size > 0:
                    toks = _ocr_digits(crop, 7) or _ocr_digits(crop, 8)
                    for t, cf in toks:
                        m = re.fullmatch(r"(\d{1,2})\.?", t)
                        if m and int(m.group(1)) > 0 and (t.endswith(".") or len(m.group(1)) == 2 or cf >= 55):
                            q_starts.append((by0, int(m.group(1))))
                            break

    # ৪. OCR ও Eng লাইন থেকে প্রশ্ন নম্বর এবং Soln: শনাক্ত করা
    for lns in (ocr_lines, eng_lines):
        for ln in lns:
            wx0, wy0, wx1, wy1, t0, _ = ln[0]
            joined = " ".join(w[4] for w in ln)
            if _UDV_BANNER_RE.search(joined):
                continue

            # প্রথম বা দ্বিতীয় টোকেনে Soln: আছে কি না চেক করা
            comb01 = (t0 + (ln[1][4] if len(ln) > 1 else "")).strip()
            if wx0 <= body_margin_x + 175:
                if _UDV_SOL_RE.match(t0) or _UDV_SOL_RE.match(comb01) or re.search(r"(?:^|\s)(?:Sol[\"\'\^\ⁿnN]*\s*[\:\.]|সমাধান\s*[\:\ঃ])", joined):
                    sol_stops.append(wy0)
                    continue

            # প্রশ্ন নম্বর (হ্যাঙ্গিং মার্জিনে অথবা চিত্রের ডান পাশে যেমন s6 Written 01.)
            for idx_w, w in enumerate(ln[:3]):
                c_x0, c_y0, c_x1, c_y1, c_t, c_cf = w
                if c_x0 > outer_margin_max + 8 and not (idx_w == 0 and c_x0 < Wc * 0.55 and re.match(r"^0[1-9]\.$", c_t)):
                    continue
                m = _UDV_NUM_RE.match(c_t)
                if m:
                    num_str = m.group(1).translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
                    qnum = int(num_str) if num_str.isdigit() else None
                    if qnum and 1 <= qnum <= 99:
                        # যদি এটি সমীকরণের অংশ না হয় এবং পরে শব্দ থাকে
                        rest = " ".join(x[4] for x in ln[idx_w + 1:])
                        if len(rest) >= 3 and not re.match(r"^[\=\+\-\×\/0-9\s]+$", rest):
                            q_starts.append((c_y0, qnum))
                            break

    # ডুপ্লিকেট প্রশ্ন শুরু এবং স্টপ মার্জ করা
    q_starts.sort(key=lambda x: x[0])
    merged_starts = []
    for y0, qn in q_starts:
        if not merged_starts or (y0 - merged_starts[-1][0]) > 26:
            merged_starts.append([y0, qn])
        elif merged_starts[-1][1] is None and qn is not None:
            merged_starts[-1][1] = qn

    sol_stops = sorted(set(sol_stops))

    # ৫. প্রতিটি প্রশ্নের জন্য বক্স তৈরি করা (প্রশ্ন শুরু থেকে প্রথম Soln: অথবা পরের প্রশ্ন পর্যন্ত)
    boxes = []
    for idx, (st_y, qn) in enumerate(merged_starts):
        nxt_q_y = merged_starts[idx + 1][0] - P['pad_bot'] if idx + 1 < len(merged_starts) else (y_bot - 4)
        cand_sols = [sy for sy in sol_stops if (st_y + 18) <= sy < nxt_q_y]
        stop_y = (cand_sols[0] - P['pad_bot']) if cand_sols else nxt_q_y

        top = max(y_top, st_y - P['pad_top'])
        bot = min(y_bot, max(top + 20, stop_y))

        # বক্সের ভেতরের নিচের ফাঁকা জায়গা ট্রিম করা
        ink_rows = np.where(i_c[top:bot, :].sum(1) > 2)[0]
        if len(ink_rows) == 0:
            continue
        bot = min(bot, top + int(ink_rows[-1]) + P['pad_bot'] + 2)
        if bot - top < P['min_q_h']:
            continue

        xs = np.where(i_c[top:bot, :].sum(0) > 0)[0]
        bx0 = max(0, (int(xs[0]) if len(xs) else 0) - 6)
        bx1 = min(Wc, (int(xs[-1]) if len(xs) else Wc) + 8)
        if bx1 - bx0 < 120:
            continue

        boxes.append((bx0, top, bx1, bot, qn, bool(cand_sols)))

    return boxes


def _process_udvash_page(page_no, pg, P, start_num=1):
    """উদ্ভাস পেজের জন্য জোন ও কলাম ভিত্তিক প্রশ্ন এক্সট্রাকশন।"""
    ink2 = pg['ink2']
    H, W = ink2.shape
    hdr = max(P['header_px'], min(c[2] for c in pg['cdata']))
    ftr = min(footer_start(rc) for rc in pg['raw_cols'])

    banners = _detect_fullwidth_banners(ink2, hdr, ftr)
    zones = []
    cur_y = hdr
    for by0, by1 in banners:
        if by0 - cur_y >= 65:
            zones.append((cur_y, by0))
        cur_y = by1
    if ftr - cur_y >= 65:
        zones.append((cur_y, ftr))
    if not zones:
        zones = [(hdr, ftr)]

    qs = []
    markers = []
    q_counter = start_num

    for zy0, zy1 in zones:
        for ci, (n_c, i_c, _chdr) in enumerate(pg['cdata']):
            Hc, Wc = i_c.shape
            c_boxes = _extract_udvash_column_zone(n_c, i_c, max(0, zy0), min(Hc, zy1), ci, P)
            x0g, thy = pg['geom'][ci]
            for bx0, top, bx1, bot, qn, has_sol in c_boxes:
                num_val = qn if qn is not None else q_counter
                q_counter = num_val + 1
                crop = n_c[top:bot, bx0:bx1].copy()
                poly = []
                for xr, yr in ((bx0, top), (bx1, top), (bx1, bot), (bx0, bot)):
                    ys = yr + float(thy[min(len(thy) - 1, max(0, int(yr)))]) * (xr - Wc / 2.0)
                    poly.append((xr + x0g + pg['shear'] * ys, ys))
                px = [p[0] for p in poly]
                py = [p[1] for p in poly]
                rect_px = (min(px), min(py), max(px), max(py))
                flags = [] if has_sol else ['no solution label found']
                conf = 1.0 if has_sol else 0.85
                qs.append(Question(page_no, ci, num_val, (bx0, top, bx1, bot), crop, conf, flags, poly, rect_px))
                markers.append(Marker(ci, top, top + 18, 24, 'num', num=num_val, conf=conf))

    return qs, dict(cols=pg['cols'], div=pg['div'], cdata=pg['cdata'], markers=markers)


# --------------------------------------------------------------------------- Main Entry Points
def process_pages(grays, profile=None, start_num=1, progress=None):
    """whole document at once. Automatically routes between Rhombus and Udvash modes."""
    P = {**PROFILE, **(profile or {})}
    pgs = []
    for i, g in enumerate(grays):
        pgs.append(_prepare(g, P))
        if progress:
            progress(i + 1, len(grays))

    mode = P.get('mode', 'auto')
    use_udvash = (mode == 'udvash') or (mode == 'auto' and _is_udvash_document(pgs))

    if use_udvash:
        res = []
        k = start_num
        for pno, pg in enumerate(pgs, 1):
            qs, dbg = _process_udvash_page(pno, pg, P, k)
            if qs:
                k = qs[-1].num + 1
            res.append((qs, dbg))
        return res

    # Rhombus pipeline (s2, s3, s4)
    labels = learn_labels([b for pg in pgs for col in pg['bands'] for b in col], P['label_thr'], 3, P['label_frac'])
    res = []
    k = start_num
    for pno, pg in enumerate(pgs, 1):
        markers = [m for col in pg['bands'] for m in _grammar(col, labels, P)]
        _number(markers, P, k)
        nums = [m.num for m in markers if m.kind == 'num']
        k = (max(nums) + 1) if nums else k
        qs = _questions(pno, pg, markers, P)
        res.append((qs, dict(cols=pg['cols'], div=pg['div'], cdata=pg['cdata'], markers=markers)))
    return res


def process_page(gray, page_no=1, profile=None, start_num=1):
    (qs, dbg), = process_pages([gray], profile, start_num)
    for q in qs:
        q.page = page_no
    return qs, dbg


def overlay(dbg, qs):
    pics = []
    for ci, (n_c, i_c, hdr) in enumerate(dbg['cdata']):
        v = cv2.cvtColor(n_c, cv2.COLOR_GRAY2BGR)
        for m in dbg['markers']:
            if m.col != ci:
                continue
            c = {'num': (0, 0, 255), 'cand': (0, 140, 255), 'label': (255, 120, 0), 'hdr': (200, 0, 200)}[m.kind]
            cv2.rectangle(v, (0, m.y0), (16, m.y1), c, -1)
        for q in qs:
            if q.col != ci:
                continue
            x0, y0, x1, y1 = q.box
            c = (0, 170, 0) if q.conf >= .85 else (0, 0, 230)
            cv2.rectangle(v, (x0, y0), (x1, y1), c, 3)
            cv2.putText(v, f"Q{q.num}", (x0 + 22, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, .8, c, 2)
        pics.append(v)
    return pics


def render_pdf_pages(pdf_path, dpi=150):
    try:
        import fitz
        for i, p in enumerate(fitz.open(pdf_path)):
            pix = p.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            yield i + 1, np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
    except ImportError:
        import pypdfium2 as pdfium
        for i, p in enumerate(pdfium.PdfDocument(pdf_path)):
            yield i + 1, np.array(p.render(scale=dpi / 72).to_pil().convert('L'))


def build_sheet(questions, out_pdf, gap=24, width=1000):
    from PIL import Image
    pages, cur, y = [], Image.new('RGB', (width + 80, 1500), 'white'), 40
    for q in questions:
        im = Image.fromarray(q.crop).convert('RGB')
        s = min(1.0, width / im.width)
        im = im.resize((int(im.width * s), int(im.height * s)))
        if y + im.height > 1460:
            pages.append(cur)
            cur, y = Image.new('RGB', (width + 80, 1500), 'white'), 40
        cur.paste(im, (40, y))
        y += im.height + gap
    pages.append(cur)
    pages[0].save(out_pdf, save_all=True, append_images=pages[1:])


if __name__ == '__main__':
    import sys
    import os
    pdf, outdir = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else 'digits'
    os.makedirs(outdir, exist_ok=True)
    allq = []
    pages = list(render_pdf_pages(pdf))
    for (pno, _), (qs, dbg) in zip(pages, process_pages([g for _, g in pages], {'read_numbers': mode})):
        for q in qs:
            q.page = pno
        allq += qs
        for ci, pic in enumerate(overlay(dbg, qs)):
            cv2.imwrite(f'{outdir}/page{pno}_col{ci + 1}_check.png', pic)
        print(f'page {pno}: found', [(q.col, q.num, round(q.conf, 2)) for q in qs])
    for q in allq:
        cv2.imwrite(f'{outdir}/Q{q.page}_{q.col + 1}_{q.num}.png', q.crop)
    if allq:
        build_sheet(allq, f'{outdir}/practice_sheet.pdf')
