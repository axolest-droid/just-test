# -*- coding: utf-8 -*-
"""
question_vision.py  --  OCR-light question finder for photographed / scanned book pages.

Idea: do NOT read the page. Find the *left-margin markers* instead.
  1. straighten the page (slanted column divider, curved/tilted text lines),
  2. walk down each column's left margin (it may drift because of perspective),
  3. for every margin line decide: question NUMBER (tiny digit-only OCR)  or
     solution LABEL (a margin line that is not a number, e.g. "উত্তর:" / "ব্যাখ্যা:"),
  4. use the number sequence (11,12,13,...) to repair OCR misses / reject strays,
  5. question box = number line  ->  just before the first solution label.

Needs only: numpy, opencv, pytesseract + tesseract *eng* (no Bengali model needed).
Engine works on a page IMAGE and returns rectified crops, so it is independent of
PyMuPDF; use `render_pdf_pages()` or your own renderer.
"""
import re, random
from dataclasses import dataclass, field
import numpy as np, cv2, pytesseract

# --------------------------------------------------------------------------- profile
PROFILE = dict(
    margin_tol=12,        # px a line may sit from the fitted margin and still count as "margin"
    num_max_w=48,         # widest first-token that may still be a number marker
    pad_top=6, pad_bot=4, # px kept above the number / below the last question line
    min_q_h=45,           # smaller boxes are flagged
    max_q_h=700,          # bigger boxes are flagged
    label_thr=0.7, label_frac=.25,   # how alike repeated label glyphs must be / how common a cluster must be
    header_px=60,         # ignore this many px at the top of every column (rules/icons are handled by shape, not position)
    read_numbers='digits',# 'digits' = tiny eng-digit OCR | 'none' = narrow margin token is a question start (Bengali digits etc.)
)

# --------------------------------------------------------------------------- image prep
def normalize(g):
    bg = cv2.medianBlur(cv2.resize(g, None, fx=.25, fy=.25), 31)
    bg = cv2.resize(bg, (g.shape[1], g.shape[0]))
    n = cv2.divide(g, bg, scale=255)            # shadow / uneven light removed
    return n, (n < 190).astype(np.uint8)

def _bands(ink, x0, x1, y0, y1, minh=8):
    prof = ink[y0:y1, x0:x1].sum(1); on = prof > max(2, .004 * (x1 - x0)); out = []; s = None
    for i, v in enumerate(on):
        if v and s is None: s = i
        if not v and s is not None:
            if i - s >= minh: out.append((s + y0, i + y0))
            s = None
    if s is not None: out.append((s + y0, y1))
    return out

def _first_token(ink, y0, y1, x0, x1, gap=9):
    xs = np.where(ink[y0:y1, x0:x1].sum(0) > 0)[0]
    if len(xs) == 0: return None
    s = e = xs[0]
    for x in xs[1:]:
        if x - e > gap: break
        e = x
    return x0 + s, x0 + e

# --------------------------------------------------------------------------- layout
def find_slanted_divider(ink):
    """(slope, x_top) with divider x(y)=x_top+slope*y, or None for a single column."""
    H, W = ink.shape
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(H * .04))))
    lo, hi = int(W * .3), int(W * .7)
    yy, xx = np.nonzero(v[:, lo - 80:hi + 80]); xx = xx + lo - 80
    best = (0, None, None)
    for s in np.arange(-0.08, 0.0801, 0.004):
        x0 = np.round(xx - s * yy).astype(int)
        k = np.convolve(np.bincount(x0 - x0.min()), np.ones(5), 'same')
        i = int(np.argmax(k))
        if k[i] > best[0]: best = (k[i], s, i + x0.min())
    return None if best[0] < H * .35 else (best[1], best[2])

def straighten(n, ink, div):
    if div is None: return n, ink
    H, W = ink.shape; M = np.float32([[1, -div[0], 0], [0, 1, 0]])
    return (cv2.warpAffine(n, M, (W, H), borderValue=255), cv2.warpAffine(ink, M, (W, H), borderValue=0))

def strip_rules(ink):
    """remove the long thin vertical divider and thin horizontal / dotted rules."""
    H, W = ink.shape; out = ink.copy()
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(H * .1))))
    out &= (cv2.dilate(v, np.ones((1, 3), np.uint8)) == 0).astype(np.uint8)
    j = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)))
    hr = cv2.morphologyEx(j, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * .16), 1)))
    hr &= (cv2.morphologyEx(j, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * .16), 6))) == 0).astype(np.uint8)
    out &= (cv2.dilate(hr, np.ones((3, 1), np.uint8)) == 0).astype(np.uint8)
    return out

def _best_angle(ink, rng=6, step=.25):
    small = cv2.resize(ink.astype(np.float32), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    c = (small.shape[1] / 2, small.shape[0] / 2); best = (-1, 0)
    for a in np.arange(-rng, rng + 1e-6, step):
        r = cv2.warpAffine(small, cv2.getRotationMatrix2D(c, a, 1), (small.shape[1], small.shape[0]))
        s = np.sum(np.diff(r.sum(1)) ** 2)
        if s > best[0]: best = (s, a)
    return best[1]

def dewarp(arrs, ink_ref, win=380, step=190):
    """first-order page-curl fix: text-line angle is measured every ~200px down the column."""
    H, W = ink_ref.shape
    ys = sorted(set(list(range(0, max(1, H - win), step)) + [max(0, H - win)])); cs, th = [], []
    for y in ys:
        seg = ink_ref[y:y + win]
        cs.append(y + win / 2); th.append(np.deg2rad(_best_angle(seg) if seg.sum() > 200 else 0.))
    th = np.array(th)
    if len(th) >= 3: th = np.array([np.median(th[max(0, i - 1):i + 2]) for i in range(len(th))])
    thy = np.interp(np.arange(H), cs, th)
    xx = np.tile(np.arange(W, dtype=np.float32), (H, 1)); yy = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    my = (yy + thy[:, None] * (xx - W / 2)).astype(np.float32)
    return [cv2.remap(a, xx, my, cv2.INTER_NEAREST if a.dtype == bool else cv2.INTER_LINEAR,
                      borderValue=255 if a.max() > 1 else 0) for a in arrs], thy

def header_end(raw_ink):
    H, W = raw_ink.shape
    j = cv2.dilate(raw_ink, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    r = cv2.morphologyEx(j, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * .45), 1)))
    r &= (cv2.morphologyEx(j, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * .45), 6))) == 0).astype(np.uint8)
    rows = np.where(r[:int(H * .16)].sum(1) > 0)[0]; rows = rows[rows > 60]
    if len(rows) == 0: return 100
    end = rows[0]
    for y in rows[1:]:                    # first rule only (later thin rules can sit inside the first question)
        if y - end > 12: break
        end = y
    return int(end) + 6

# --------------------------------------------------------------------------- margin
def _fit_margin(pts, tol=9, iters=500, seed=1):
    """robust line x=a*y+b through the LEFT-most ink of many lines (lower envelope)."""
    rnd = random.Random(seed); pts = np.array(pts, float); best = None
    for _ in range(iters):
        i, j = rnd.sample(range(len(pts)), 2)
        if abs(pts[i, 0] - pts[j, 0]) < 150: continue
        a = (pts[j, 1] - pts[i, 1]) / (pts[j, 0] - pts[i, 0])
        if abs(a) > .1: continue
        b = pts[i, 1] - a * pts[i, 0]; res = pts[:, 1] - (a * pts[:, 0] + b)
        sc = (abs(res) <= tol).sum() - 3 * (res < -tol).sum()
        if best is None or sc > best[0]: best = (sc, a, b)
    if best is None: return 0., float(pts[:, 1].min())
    _, a, b = best; m = abs(pts[:, 1] - (a * pts[:, 0] + b)) <= tol
    return tuple(np.polyfit(pts[m, 0], pts[m, 1], 1)) if m.sum() >= 3 else (a, b)

def _margin_fn(ink):
    pts = []
    for a, b in _bands(ink, 0, ink.shape[1], 100, ink.shape[0] - 70):
        xs = np.where(ink[a:b].sum(0) > 0)[0]
        if len(xs): pts.append(((a + b) / 2, xs[0]))
    if len(pts) < 4: x = min([p[1] for p in pts] or [0]); return (lambda y: x), (0., x)
    a, b = _fit_margin(pts); return (lambda y: a * y + b), (a, b)

def _gather(img, mx, lo=-8, hi=76):
    """cut a strip that follows the (tilted) margin."""
    H, W = img.shape; w = hi - lo; S = np.zeros((H, w), img.dtype) if img.max() <= 1 else np.full((H, w), 255, img.dtype)
    for y in range(H):
        a = int(round(mx(y))) + lo; s, e = max(0, a), min(W, a + w)
        if e > s: S[y, s - a:e - a] = img[y, s:e]
    return S

# --------------------------------------------------------------------------- digit OCR (tiny crops only)
def _ocr_digits(crop, psm):
    crop = cv2.copyMakeBorder(cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC), 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    d = pytesseract.image_to_data(crop, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789.', output_type=pytesseract.Output.DICT)
    return [(t.strip(), int(c)) for t, c in zip(d['text'], d['conf']) if t.strip() and str(c).lstrip('-').isdigit()]

def _variants(crop):
    yield crop
    _, o = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); yield o
    yield cv2.erode(o, np.ones((2, 2), np.uint8))          # thicker ink
    yield cv2.dilate(o, np.ones((2, 2), np.uint8))         # thinner ink

def read_number(Sg, y0, y1, x_start, widths=(26, 30, 34, 38, 42, 48)):
    """vote over several crops/preprocessings; returns (number, score) or None."""
    for w in (34, 44):                                       # fast path: clear reading -> done
        crop = Sg[max(0, y0 - 3):y1 + 3, max(0, x_start - 4):x_start + w]
        if crop.size == 0: continue
        for t, cf in _ocr_digits(crop, 7):
            m = re.fullmatch(r'(\d{1,3})\.', t)
            if m and cf >= 85: return (int(m.group(1)), cf)
    votes = {}
    for w in widths:
        crop = Sg[max(0, y0 - 3):y1 + 3, max(0, x_start - 4):x_start + w]
        if crop.size == 0: continue
        for v in _variants(crop):
            for psm in (7, 8):
                for t, cf in _ocr_digits(v, psm):
                    m = re.fullmatch(r'(\d{1,3})(\.?)', t)
                    if m and cf >= 25 and (m.group(2) or len(m.group(1)) >= 2):
                        k = int(m.group(1)); votes[k] = votes.get(k, 0) + cf + (15 if m.group(2) else 0) + 10 * (len(m.group(1)) - 1)
    if not votes: return None
    k = max(votes, key=votes.get); return (k, votes[k])

# --------------------------------------------------------------------------- results
@dataclass
class Marker:
    col: int; y0: int; y1: int; tw: int; kind: str          # kind: 'num' | 'cand' | 'label'
    num: int = None; conf: float = 0; guessed: bool = False

@dataclass
class Question:
    page: int; col: int; num: int; box: tuple               # box = (x0,y0,x1,y1) inside that column's rectified image
    crop: np.ndarray = None; conf: float = 1.0; flags: list = field(default_factory=list)
    poly: list = None; rect_px: tuple = None      # same box in ORIGINAL page pixel coordinates

def _margin_bands(n_c, i_c, raw_c, hdr, ci, P):
    """every line that starts at the (tilted) left margin: width of first token, shape signature, box-header flag."""
    H, Wc = i_c.shape; mx, _ = _margin_fn(np.where(np.arange(H)[:, None] < hdr, 0, i_c).astype(np.uint8))
    Si = _gather(i_c, mx); Sg = _gather(n_c, mx); Si[:hdr] = 0; out = []
    hl = cv2.morphologyEx(raw_c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(Wc * .5), 1)))
    for y0, y1 in _bands(Si, 0, Si.shape[1], 0, H - 60, minh=9):
        xs = np.where(Si[y0:y1].sum(0) > 0)[0]
        if len(xs) == 0 or xs[0] > 8 + P['margin_tol']: continue
        ft = _first_token(Si, y0, y1, 0, Si.shape[1]); tw = ft[1] - ft[0] + 1
        if Si[y0:y1, ft[0]:ft[1] + 1].mean() > .85 and tw >= 14: continue     # solid square / badge / bullet block, not text
        sigs = []
        rr = np.where(Si[y0:y1, ft[0]:ft[0] + 44].sum(1) > 0)[0]            # rows of the leading glyphs only (a tall fraction may share the band)
        r0, r1 = (y0 + rr[0], y0 + rr[-1] + 1) if len(rr) >= 6 else (y0, y1)
        for w in (44, 56, 68):                       # label may be followed closely by content: try several widths
            patch = Sg[r0:r1, max(0, ft[0] - 2):ft[0] + min(tw, w) + 2]
            sg = cv2.GaussianBlur(cv2.resize(patch, (48, 24), interpolation=cv2.INTER_AREA).astype(np.float32), (3, 3), 0)
            sigs.append((sg - sg.mean()) / (sg.std() + 1e-6))
        out.append(dict(col=ci, y0=y0, y1=y1, tw=tw, ft0=ft[0], sigs=sigs, box=bool(hl[max(0, y0 - 3):y0 + 8].sum() > 0 and hl[max(0, y1 - 8):y1 + 3].sum() > 0 and y1 - y0 > 14), Sg=Sg))
    return out

def learn_labels(all_bands, thr=0.8, min_n=3, frac=.3):
    """solution labels (উত্তর: / ব্যাখ্যা: / সমাধান: ...) are the wide margin tokens that look the SAME again and again."""
    wide = [b for b in all_bands if b['tw'] > 40 and not b['box']]; reps = []
    for b in wide:
        for r in reps:
            if max(float((a * c).mean()) for a, c in zip(r['rep'], b['sigs'])) >= thr: r['members'].append(b); break
        else: reps.append(dict(rep=b['sigs'], members=[b]))
    reps = [r for r in reps if len(r['members']) >= min_n]
    if not reps: return set()
    top = max(len(r['members']) for r in reps)
    return {id(b) for r in reps if len(r['members']) >= frac * top for b in r['members']}

def _grammar(bands, labels, P):
    """state machine down one column:  [start] -narrow-> QUESTION -label-> SOLUTION -narrow-> QUESTION ..."""
    out = []; state = 'start'
    for b in sorted(bands, key=lambda b: b['y0']):
        if b['box']: out.append(Marker(b['col'], b['y0'], b['y1'], b['tw'], 'hdr')); state = 'start'; continue
        if id(b) in labels:
            out.append(Marker(b['col'], b['y0'], b['y1'], b['tw'], 'label')); state = 'sol'; continue
        if state in ('start', 'sol') and 8 <= b['tw'] <= P['num_max_w']:
            m = Marker(b['col'], b['y0'], b['y1'], b['tw'], 'num'); m.band = b; out.append(m); state = 'q'
    return out

def _number(markers, P, start):
    k = start - 1; prev = None
    for m in sorted([m for m in markers if m.kind == 'num'], key=lambda m: (m.col, m.y0)):
        r = read_number(m.band['Sg'], m.y0, m.y1, m.band['ft0']) if P['read_numbers'] == 'digits' else None
        if r and (prev is None or 0 < r[0] - prev <= 3): m.num, m.conf = r[0], r[1]
        else: m.num, m.guessed = k + 1, P['read_numbers'] == 'digits'
        k = prev = m.num

def _prepare(gray, P):
    n, ink = normalize(gray); div = find_slanted_divider(ink)
    n2, ink2 = straighten(n, ink, div); clean = strip_rules(ink2); H, W = gray.shape
    cols = [(30, int(div[1]) - 1), (int(div[1]) + 2, W - 30)] if div else [(30, W - 30)]
    cdata, bands, geom = [], [], []
    for ci, (x0, x1) in enumerate(cols):
        (n_c, i_c, raw_c), thy = dewarp([n2[:, x0:x1].copy(), clean[:, x0:x1].copy(), ink2[:, x0:x1].copy()], clean[:, x0:x1])
        i_c = (i_c > 0).astype(np.uint8); raw_c = (raw_c > 0).astype(np.uint8)
        hdr = P['header_px']; cdata.append((n_c, i_c, hdr)); geom.append((x0, thy))
        bands.append(_margin_bands(n_c, i_c, raw_c, hdr, ci, P))
    return dict(cols=cols, div=div, cdata=cdata, bands=bands, geom=geom, shear=(div[0] if div else 0.0), size=gray.shape)

def _questions(page_no, pg, markers, P):
    qs = []
    for ci, (n_c, i_c, hdr) in enumerate(pg['cdata']):
        ms = sorted([m for m in markers if m.col == ci], key=lambda m: m.y0); Hc, Wc = i_c.shape
        for k, m in enumerate(ms):
            if m.kind != 'num': continue
            nxt = [x for x in ms[k + 1:]]
            stop = nxt[0].y0 - P['pad_bot'] if nxt else Hc - 70; flags = []
            if not nxt or nxt[0].kind != 'label': flags.append('no solution label found')
            top = max(0, m.y0 - P['pad_top']); bot = max(top + 1, stop)
            xs = np.where(i_c[top:bot].sum(0) > 0)[0]
            bx0 = max(0, (xs[0] if len(xs) else 0) - 6); bx1 = min(Wc, (xs[-1] if len(xs) else Wc) + 8)
            if bot - top < P['min_q_h']: flags.append('very small box')
            if bot - top > P['max_q_h']: flags.append('very tall box')
            if m.guessed: flags.append('number inferred from sequence')
            crop = n_c[top:bot, bx0:bx1].copy(); soft = []
            if nxt and nxt[0].kind == 'label':      # curved edge: option lower than the label on the far side
                lb = nxt[0]; zone0 = int(bx0 + .55 * (bx1 - bx0))
                lim = min(Hc - 1, lb.y1 + 12); rowink = i_c[bot:lim, zone0:bx1].sum(1) > 0
                straddle = int((i_c[max(0, bot - 3), zone0:bx1] & i_c[min(Hc - 1, bot + 3), zone0:bx1]).sum())   # a glyph cut in half by the stop line
                if straddle >= 3 and lb.y1 - lb.y0 <= 40 and rowink[:max(1, lb.y1 - bot + 8)].sum() > 2:
                    last = 0; blank = 0
                    for r, v in enumerate(rowink):
                        if v: last, blank = r, 0
                        else:
                            blank += 1
                            if blank >= 4 and r > lb.y1 - bot: break
                    nb = min(Hc - 1, bot + last + 6); cols_ink = np.where(i_c[lb.y0:lb.y1, bx0:bx1].sum(0) > 0)[0]
                    e = cols_ink[0] if len(cols_ink) else 0
                    for c in cols_ink[1:]:
                        if c - e > 40: break
                        e = c
                    crop = n_c[top:nb, bx0:bx1].copy(); crop[max(0, lb.y0 - 3 - top):lb.y1 + 3 - top, :e + 8] = 255
                    bot = nb; soft.append('curved edge: label text masked')
            x0g, thy = pg['geom'][ci]; poly = []
            for xr, yr in ((bx0, top), (bx1, top), (bx1, bot), (bx0, bot)):       # rectified -> original page pixels
                ys = yr + float(thy[min(len(thy) - 1, max(0, int(yr)))]) * (xr - Wc / 2.0)
                poly.append((xr + x0g + pg['shear'] * ys, ys))
            px = [p[0] for p in poly]; py = [p[1] for p in poly]; rect_px = (min(px), min(py), max(px), max(py))
            hard = [f for f in flags if f != 'very small box'] + (['very small box'] if bot - top < P['min_q_h'] // 2 else [])
            qs.append(Question(page_no, ci, m.num, (bx0, top, bx1, bot), crop, max(.05, 1.0 - .35 * len(hard) - .3 * len(soft)), flags + soft, poly, rect_px))
    qs.sort(key=lambda q: (q.col, q.box[1]))
    return qs

def process_pages(grays, profile=None, start_num=1, progress=None):
    """whole document at once (labels are learned from ALL pages). grays: list of uint8 page images."""
    P = {**PROFILE, **(profile or {})}; pgs = []
    for i, g in enumerate(grays):
        pgs.append(_prepare(g, P))
        if progress: progress(i + 1, len(grays))
    labels = learn_labels([b for pg in pgs for col in pg['bands'] for b in col], P['label_thr'], 3, P['label_frac'])
    res = []; k = start_num
    for pno, pg in enumerate(pgs, 1):
        markers = [m for col in pg['bands'] for m in _grammar(col, labels, P)]
        _number(markers, P, k); nums = [m.num for m in markers if m.kind == 'num']; k = (max(nums) + 1) if nums else k
        qs = _questions(pno, pg, markers, P); res.append((qs, dict(cols=pg['cols'], div=pg['div'], cdata=pg['cdata'], markers=markers)))
    return res

def process_page(gray, page_no=1, profile=None, start_num=1):
    """single page convenience wrapper"""
    (qs, dbg), = process_pages([gray], profile, start_num)
    for q in qs: q.page = page_no
    return qs, dbg

def overlay(dbg, qs):
    """one picture per column showing markers and final boxes (for quick human checking)."""
    pics = []
    for ci, (n_c, i_c, hdr) in enumerate(dbg['cdata']):
        v = cv2.cvtColor(n_c, cv2.COLOR_GRAY2BGR)
        for m in dbg['markers']:
            if m.col != ci: continue
            c = {'num': (0, 0, 255), 'cand': (0, 140, 255), 'label': (255, 120, 0), 'hdr': (200, 0, 200)}[m.kind]
            cv2.rectangle(v, (0, m.y0), (16, m.y1), c, -1)
        for q in qs:
            if q.col != ci: continue
            x0, y0, x1, y1 = q.box; c = (0, 170, 0) if q.conf >= .9 else (0, 0, 230)
            cv2.rectangle(v, (x0, y0), (x1, y1), c, 3); cv2.putText(v, f"Q{q.num}", (x0 + 22, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, .8, c, 2)
        pics.append(v)
    return pics

# --------------------------------------------------------------------------- PDF helpers / CLI
def render_pdf_pages(pdf_path, dpi=150):
    try:
        import fitz
        for i, p in enumerate(fitz.open(pdf_path)):
            pix = p.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY); yield i + 1, np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
    except ImportError:
        import pypdfium2 as pdfium
        for i, p in enumerate(pdfium.PdfDocument(pdf_path)):
            yield i + 1, np.array(p.render(scale=dpi / 72).to_pil().convert('L'))

def build_sheet(questions, out_pdf, gap=24, width=1000):
    from PIL import Image
    pages, cur, y = [], Image.new('RGB', (width + 80, 1500), 'white'), 40
    for q in questions:
        im = Image.fromarray(q.crop).convert('RGB'); s = min(1.0, width / im.width); im = im.resize((int(im.width * s), int(im.height * s)))
        if y + im.height > 1460: pages.append(cur); cur, y = Image.new('RGB', (width + 80, 1500), 'white'), 40
        cur.paste(im, (40, y)); y += im.height + gap
    pages.append(cur); pages[0].save(out_pdf, save_all=True, append_images=pages[1:])

if __name__ == '__main__':
    import sys, os
    pdf, outdir = sys.argv[1], sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else 'digits'
    os.makedirs(outdir, exist_ok=True); allq = []
    pages = list(render_pdf_pages(pdf))
    for (pno, _), (qs, dbg) in zip(pages, process_pages([g for _, g in pages], {'read_numbers': mode})):
        for q in qs: q.page = pno
        allq += qs
        for ci, pic in enumerate(overlay(dbg, qs)): cv2.imwrite(f'{outdir}/page{pno}_col{ci + 1}_check.png', pic)
        print(f'page {pno}: found', [(q.col, q.num, round(q.conf, 2)) for q in qs])
    for q in allq: cv2.imwrite(f'{outdir}/Q{q.page}_{q.col + 1}_{q.num}.png', q.crop)
    if allq: build_sheet(allq, f'{outdir}/practice_sheet.pdf')
