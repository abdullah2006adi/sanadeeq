"""يقلب أرقام الصناديق المرسومة لتصير من اليمين لليسار، ويضع كل رقم في وسط لوحته تماماً.
مكان اللوحة يُحسب من تناظرها (اللوحات متناظرة) بعد محو الأرقام."""
import json, numpy as np, cv2
def glyph(img, r, win, loose=False):
    H, W = img.shape[:2]; (fx0, fx1, fy0, fy1), pol = win
    x, y, w, h = [v / 100 for v in r]
    X0, X1, Y0, Y1 = int((x + fx0 * w) * W), int((x + fx1 * w) * W), max(0, int((y + fy0 * h) * H)), int((y + fy1 * h) * H)
    g = cv2.cvtColor(img[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, 0, 255, (cv2.THRESH_BINARY_INV if pol == 'dark' else cv2.THRESH_BINARY) + cv2.THRESH_OTSU)
    k, lab, st, _ = cv2.connectedComponentsWithStats(m); hh, ww = m.shape
    if loose:   # نافذة ضيقة حول الرقم: يُقبل ما يلمس الحافة العليا
        cand = [i for i in range(1, k) if st[i][4] >= 25 and st[i][0] > 0 and st[i][0] + st[i][2] < ww
                and st[i][1] + st[i][3] < hh and st[i][3] >= 0.18 * hh]
    else:
        cand = [i for i in range(1, k) if st[i][4] >= 40 and st[i][0] > 0 and st[i][1] > 0 and st[i][0] + st[i][2] < ww
                and st[i][1] + st[i][3] < hh and 0.15 * hh <= st[i][3] <= 0.6 * hh and st[i][2] <= 0.5 * ww]
    main = max(cand, key=lambda i: st[i][4]); mx, my, mw, mh, _ = st[main]
    keep = np.zeros_like(m)
    for i in cand:
        bx, by, bw, bh, _ = st[i]
        ov = min(by + bh, my + mh) - max(by, my); gap = max(bx - (mx + mw), mx - (bx + bw), 0)
        if i == main or (ov > 0.7 * mh and abs(bh - mh) < 0.25 * mh and gap < 0.6 * mh): keep[lab == i] = 255
    full = np.zeros((H, W), np.uint8); full[Y0:Y1, X0:X1] = keep
    ys, xs = np.nonzero(full)
    return full, (xs.min(), ys.min(), xs.max(), ys.max())
def plate_center(gray, cx, cy, gh):
    """مركز اللوحة: النقطة التي تكون اللوحة حولها متناظرة يميناً ويساراً، وأعلى وأسفل"""
    hw, hh = int(1.05 * gh), int(0.8 * gh); rng = int(0.45 * gh)
    def best(axis):
        bestv, bestc = 1e18, 0
        for d in range(-rng, rng + 1):
            x, y = (cx + d, cy) if axis == 0 else (cx, cy + d)
            P = gray[int(y - hh):int(y + hh), int(x - hw):int(x + hw)].astype(np.float32)
            if P.size == 0: continue
            Q = P[:, ::-1] if axis == 0 else P[::-1, :]
            v = np.abs(P - Q).mean()
            if v < bestv: bestv, bestc = v, d
        return bestc
    dx = best(0); dy = best(1)
    return cx + dx, cy + dy
def renumber(img, rects, win, src_of, loose=False):
    """src_of[t] = رقم الصندوق الذي يُنقل رقمه إلى المكان t"""
    orig = img.copy(); G = [glyph(orig, r, win, loose) for r in rects]
    allm = np.zeros(img.shape[:2], np.uint8)
    for m, _ in G: allm |= m
    img = cv2.inpaint(img, cv2.dilate(allm, np.ones((11, 11), np.uint8)), 9, cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for t, s in enumerate(src_of):
        m, (sx0, sy0, sx1, sy1) = G[s]; _, (tx0, ty0, tx1, ty1) = G[t]
        gh = max(sy1 - sy0, ty1 - ty0)
        pcx, pcy = plate_center(gray, (tx0 + tx1) / 2, (ty0 + ty1) / 2, gh)
        p = 5; bx0, by0, bx1, by1 = sx0 - p, sy0 - p, sx1 + p, sy1 + p
        patch = orig[by0:by1 + 1, bx0:bx1 + 1].astype(np.float32)
        al = cv2.GaussianBlur(cv2.dilate(m, np.ones((3, 3), np.uint8))[by0:by1 + 1, bx0:bx1 + 1], (3, 3), 0).astype(np.float32)[..., None] / 255
        ph, pw = patch.shape[:2]
        X = int(round(pcx - (sx1 - sx0) / 2)) - p
        Y = int(round(pcy - (sy1 - sy0) / 2)) - p
        roi = img[Y:Y + ph, X:X + pw].astype(np.float32)
        img[Y:Y + ph, X:X + pw] = (roi * (1 - al) + patch * al).clip(0, 255).astype(np.uint8)
    return img
