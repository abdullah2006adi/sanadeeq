"""يصنع نسخة طولية (للجوال) من كل صورة صناديق جاهزة: صفّان بدل خمسة، والترقيم من اليمين لليسار.
يكتب boards/boardN-tall.jpg ويطبع أماكن الصناديق بالنسبة المئوية."""
import json, sys, numpy as np, cv2
ROOT = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/sanadeeq'
OUT = sys.argv[2] if len(sys.argv) > 2 else ROOT + '/boards'
cfg = {b['id']: b for b in json.load(open(ROOT + '/config.json'))['boards']}

def px(r, W, H): x, y, w, h = r; return np.array([x * W / 100, y * H / 100, w * W / 100, h * H / 100])

def vstack(parts, ov=18):
    """يلصق الشرائح فوق بعض مع دمج ناعم عند الحدود"""
    out = parts[0].astype(np.float32)
    for p in parts[1:]:
        p = p.astype(np.float32); o = min(ov, out.shape[0] // 3, p.shape[0] // 3)
        a = np.linspace(0, 1, o)[:, None, None]
        mid = out[-o:] * (1 - a) + p[:o] * a
        out = np.concatenate([out[:-o], mid, p[o:]], 0)
    return out.clip(0, 255).astype(np.uint8)

def feather_paste(dst, src, x, y, f):
    h, w = src.shape[:2]
    m = np.ones((h, w), np.float32)
    for k in range(f):
        v = (k + 1) / (f + 1); m[k, :] = np.minimum(m[k, :], v); m[h - 1 - k, :] = np.minimum(m[h - 1 - k, :], v)
        m[:, k] = np.minimum(m[:, k], v); m[:, w - 1 - k] = np.minimum(m[:, w - 1 - k], v)
    m = m[..., None]
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32); mm = m[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (d * (1 - mm) + s * mm).astype(np.uint8)

def make(n):
    b = cfg[f'board{n}']; img = cv2.imread(f"{ROOT}/{b['src']}"); H, W = img.shape[:2]
    R = [px(r, W, H) for r in b['rects']]            # R[k] = مكان الصندوق رقم k+1
    row1, row2 = R[:5], R[5:]
    t1 = min(r[1] for r in row1); b1 = max(r[1] + r[3] for r in row1); t2 = min(r[1] for r in row2)
    d = max(4, (t2 - b1) / 2 + 0.06 * (t2 - t1))     # مسافة فوق الصندوق تشمل ما يعلوه من زينة
    # شريحة عمودية حول الصندوقين 2 و3 (في الوسط)
    def bounds(row):
        A, B, C, D = row[3], row[2], row[1], row[0]  # 4 يسار … 1 يمين
        return (A[0] + A[2] + B[0]) / 2, (C[0] + C[2] + D[0]) / 2
    xl1, xr1 = bounds(row1); xl2, xr2 = bounds(row2)
    sw = int(round(max(xr1 - xl1, xr2 - xl2)))
    def band(y0, y1, row):
        xl, xr = bounds(row); cx = (xl + xr) / 2
        x0 = int(round(cx - sw / 2)); return img[int(y0):int(y1), x0:x0 + sw].copy(), x0, int(y0)
    by0 = t1 - d; by1 = t2 - d
    top_h = int(by0)
    # الجزء العلوي: يُصغَّر ليتسع عرضه للعنوان والزينة
    wt = min(W, max(sw, int(0.62 * W))); xt = (W - wt) // 2
    top = cv2.resize(img[:top_h, xt:xt + wt], (sw, max(1, round(top_h * sw / wt))), interpolation=cv2.INTER_AREA) if top_h > 8 else None
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]   # (يمين، يسار)
    parts, offs, rects = [], [], []
    if n in CELLS: return cells(n, img, R, t1, t2, d, by0, by1, top_h, pairs)
    if top is not None: parts.append(top)
    for k, (ri, li) in enumerate(pairs):
        last = k == 4
        row = row2 if last else row1
        y0, y1 = (t2 - d, H) if last else (by0, by1)
        piece, x0, yy0 = band(y0, y1, row)
        slots = {ri: row[1], li: row[2]}              # مكانا الصندوقين 2 و3 في هذا الصف
        for num, slot in slots.items():
            s = R[num]
            pad = 0.07
            sx0, sy0 = int(s[0] - pad * s[2]), int(s[1] - pad * s[3]); sx1, sy1 = int(s[0] + s[2] * (1 + pad)), int(s[1] + s[3] * (1 + pad))
            crop = img[max(0, sy0):sy1, sx0:sx1]
            sc = (slot[2] / s[2] + slot[3] / s[3]) / 2
            crop = cv2.resize(crop, (round(crop.shape[1] * sc), round(crop.shape[0] * sc)), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
            cxs, cys = slot[0] + slot[2] / 2 - x0, slot[1] + slot[3] / 2 - yy0
            px0, py0 = int(round(cxs - crop.shape[1] / 2)), int(round(cys - crop.shape[0] / 2))
            feather_paste(piece, crop, px0, py0, max(4, int(0.06 * crop.shape[1])))
            rects.append((k, num, cxs - slot[2] / 2, cys - slot[3] / 2, slot[2], slot[3]))
        parts.append(piece)
    ov = 18
    out = vstack(parts, ov)
    # حساب مواضع الصناديق في الصورة النهائية
    ys, y = [], 0
    for i, p in enumerate(parts):
        ys.append(y); y += p.shape[0] - ov
    base = 1 if top is not None else 0
    TH, TW = out.shape[:2]
    res = [None] * 10
    for k, num, x, yv, w, h in rects:
        res[num] = [round(100 * x / TW, 2), round(100 * (ys[base + k] + yv) / TH, 2), round(100 * w / TW, 2), round(100 * h / TH, 2)]
    out, res = widen(out, res); TH, TW = out.shape[:2]
    scale = 1080 / TW
    out = cv2.resize(out, (1080, round(TH * scale)), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(f"{OUT}/board{n}-tall.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return res, out.shape[:2]

ASPECT = 9 / 16
def widen(out, res):
    """يوسّع الصورة إلى 9:16 بمدّ الخلفية على الجانبين (انعكاس + ضباب) بلا صناديق إضافية"""
    TH, TW = out.shape[:2]; NW = int(round(TH * ASPECT))
    if NW <= TW: return out, res
    e = (NW - TW) // 2; NW = TW + 2 * e
    ref = cv2.copyMakeBorder(out, 0, 0, e, e, cv2.BORDER_REFLECT_101).astype(np.float32)
    soft = cv2.GaussianBlur(ref, (0, 0), 22) * 0.6
    x = np.arange(NW, dtype=np.float32)
    dist = np.maximum(e - x, x - (e + TW - 1)).clip(0) / max(1, e)       # 0 داخل الأصل، 1 عند الحافة
    w = np.clip(dist * 2.2, 0, 1)[None, :, None]
    # حافة الصورة الأصلية تذوب تدريجياً في الامتداد
    res2 = [[round((r[0] * TW / 100 + e) / NW * 100, 2), r[1], round(r[2] * TW / NW, 2), r[3]] for r in res]
    return (ref * (1 - w) + soft * w).clip(0, 255).astype(np.uint8), res2

CELLS = {1}
def cells(n, img, R, t1, t2, d, by0, by1, top_h, pairs):
    """للصور التي لكل صندوق فيها خانة مستقلة (رفوف بأعمدة): تُقصّ الخانات كما هي وتُرصّ"""
    H, W = img.shape[:2]
    def xb(num):
        row = R[:5] if num < 5 else R[5:]; j = num % 5
        c = [r[0] + r[2] / 2 for r in row]   # مراكز من اليمين لليسار
        gap = lambda a, b: (row[a][0] + row[b][0] + row[b][2]) / 2  # بين a (يمين) و b (يسار)
        right = gap(j - 1, j) if j > 0 else row[j][0] + row[j][2] + (row[j][0] - (row[j + 1][0] + row[j + 1][2])) / 2
        left = gap(j, j + 1) if j < 4 else row[j][0] - (row[j - 1][0] - (row[j][0] + row[j][2])) / 2
        return int(round(left)), int(round(right))
    cw = int(round(np.median([xb(k)[1] - xb(k)[0] for k in range(10)])))
    bh = int(by1 - by0)
    wt = W; top = cv2.resize(img[:int(by0)], (2 * cw, max(1, round(by0 * 2 * cw / W))), interpolation=cv2.INTER_AREA) if by0 > 8 else None
    parts = [top] if top is not None else []
    rects = []
    for k, (ri, li) in enumerate(pairs):
        last = k == 4
        row_cells = []
        for num in (li, ri):
            l, r = xb(num)
            y0 = int(by0 if num < 5 else t2 - d)
            y1 = H if last else y0 + bh
            c = cv2.resize(img[y0:y1, l:r], (cw, y1 - y0), interpolation=cv2.INTER_AREA)
            s = R[num]; sx = cw / (r - l)
            rects.append((k, num, (s[0] - l) * sx + (cw if num == ri else 0), s[1] - y0, s[2] * sx, s[3]))
            row_cells.append(c)
        hh = min(c.shape[0] for c in row_cells)
        parts.append(np.concatenate([c[:hh] for c in row_cells], 1))
    ov = 18
    out = vstack(parts, ov)
    ys, y = [], 0
    for p in parts: ys.append(y); y += p.shape[0] - ov
    base = 1 if top is not None else 0
    TH, TW = out.shape[:2]; res = [None] * 10
    for k, num, x, yv, w, h in rects:
        res[num] = [round(100 * x / TW, 2), round(100 * (ys[base + k] + yv) / TH, 2), round(100 * w / TW, 2), round(100 * h / TH, 2)]
    out, res = widen(out, res); TH, TW = out.shape[:2]
    out = cv2.resize(out, (1080, round(TH * 1080 / TW)), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(f"{OUT}/board{n}-tall.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return res, out.shape[:2]

if __name__ == '__main__':
    allr = {}
    for n in range(1, 6):
        r, sh = make(n); allr[f'board{n}'] = r; print(n, sh, sh[1] / sh[0])
    json.dump(allr, open('/home/claude/tall/rects.json', 'w'))
