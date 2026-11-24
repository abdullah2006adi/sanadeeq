"""الصور الطولية من المعلم: يحدد أماكن الصناديق ويقلب الأرقام لتصير من اليمين لليسار (1 يمين، 2 يسار)."""
import json, numpy as np, cv2
# [يسار x0,x1] [يمين x0,x1] ثم صفوف [y0,y1] بالنسبة المئوية
GEO = {
 1: ((17, 39.3), (61, 84.3), [(16, 28), (33.6, 44.5), (49, 62), (66, 77.5), (81, 94)]),
 2: ((15.5, 45.5), (56.5, 86.5), [(19.5, 33.5), (35, 48.5), (50, 63.5), (65, 78), (80, 93)]),
 3: ((19.3, 46.3), (55.5, 83), [(14, 28), (29, 43), (44, 58), (59, 73), (74.5, 89)]),
 4: ((16.3, 46.3), (55, 85.2), [(8, 25), (25, 42.5), (43.5, 60.5), (61.5, 78.5), (80, 97.5)]),
 5: ((17, 46.7), (54, 84.8), [(24, 36.5), (37.5, 50), (51, 63.5), (64.5, 77), (78, 91)]),
}
WIN = {2: ((0.30, 0.70, 0.52, 0.88), 'dark'), 3: ((0.25, 0.75, -0.12, 0.42), 'light'),
       4: ((0.15, 0.85, -0.14, 0.44), 'dark'), 5: ((0.25, 0.75, 0.05, 0.65), 'light')}
def rects(n):
    L, R, rows = GEO[n]; out = []
    for y0, y1 in rows:   # الرقم الفردي يمين بعد القلب
        out += [[R[0], y0, R[1] - R[0], y1 - y0], [L[0], y0, L[1] - L[0], y1 - y0]]
    return out
def glyph(img, r, n):
    H, W = img.shape[:2]; (fx0, fx1, fy0, fy1), pol = WIN[n]
    x, y, w, h = [v / 100 for v in r]
    X0, X1, Y0, Y1 = int((x + fx0 * w) * W), int((x + fx1 * w) * W), int((y + fy0 * h) * H), int((y + fy1 * h) * H)
    g = cv2.cvtColor(img[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, 0, 255, (cv2.THRESH_BINARY_INV if pol == 'dark' else cv2.THRESH_BINARY) + cv2.THRESH_OTSU)
    k, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m); hh, ww = m.shape
    cand = []
    for i in range(1, k):
        bx, by, bw, bh, area = st[i]
        if area < 40 or bx == 0 or by == 0 or bx + bw >= ww or by + bh >= hh: continue
        if bh < 0.15 * hh or bh > 0.6 * hh or bw > 0.5 * ww: continue
        cand.append(i)
    # الرقم: أكبر جزء، ومعه ما يحاذيه على السطر نفسه (مثل «1» و«0» في 10)
    main = max(cand, key=lambda i: st[i][4])
    mx, my, mw, mh, _ = st[main]
    for i in cand:
        bx, by, bw, bh, _ = st[i]
        ov = min(by + bh, my + mh) - max(by, my)
        gap = max(bx - (mx + mw), mx - (bx + bw), 0)
        if i == main or (ov > 0.7 * mh and abs(bh - mh) < 0.25 * mh and gap < 0.6 * mh):
            keep[lab == i] = 255
    full = np.zeros((H, W), np.uint8); full[Y0:Y1, X0:X1] = keep
    ys, xs = np.nonzero(full)
    return full, (xs.min(), ys.min(), xs.max(), ys.max()), (X0, Y0, X1, Y1)
def swap(n, img):
    R = rects(n); orig = img.copy()
    G = [glyph(orig, r, n) for r in R]
    allm = np.zeros(img.shape[:2], np.uint8)
    for m, _, _ in G: allm |= m
    img = cv2.inpaint(img, cv2.dilate(allm, np.ones((11, 11), np.uint8)), 9, cv2.INPAINT_TELEA)
    blank = img.copy()                       # اللوحات بلا أرقام: تُستعمل لمطابقة مكان اللوحة بدقة
    gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    for k in range(5):
        a, b = 2 * k, 2 * k + 1
        for t, s in ((a, b), (b, a)):
            m, (sx0, sy0, sx1, sy1), _ = G[s]; _, (tx0, ty0, tx1, ty1), _ = G[t]
            gh = sy1 - sy0; cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            tw, th = int(gh * 2.4), int(gh * 1.7)
            X0, Y0 = int(cx - tw / 2), int(cy - th / 2)
            tpl = gray[Y0:Y0 + th, X0:X0 + tw]
            tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2; rad = int(gh * 0.8)
            SX0, SY0 = int(tcx - tw / 2 - rad), int(tcy - th / 2 - rad)
            area = gray[SY0:SY0 + th + 2 * rad, SX0:SX0 + tw + 2 * rad]
            res = cv2.matchTemplate(area, tpl, cv2.TM_CCOEFF_NORMED); _, sc, _, loc = cv2.minMaxLoc(res)
            dx, dy = SX0 + loc[0] - X0, SY0 + loc[1] - Y0          # إزاحة اللوحة من المصدر إلى الهدف
            p = 5; bx0, by0, bx1, by1 = sx0 - p, sy0 - p, sx1 + p, sy1 + p
            patch = orig[by0:by1 + 1, bx0:bx1 + 1].astype(np.float32)
            al = cv2.GaussianBlur(cv2.dilate(m, np.ones((3, 3), np.uint8))[by0:by1 + 1, bx0:bx1 + 1], (3, 3), 0).astype(np.float32)[..., None] / 255
            ph, pw = patch.shape[:2]; X, Y = bx0 + dx, by0 + dy
            roi = img[Y:Y + ph, X:X + pw].astype(np.float32)
            img[Y:Y + ph, X:X + pw] = (roi * (1 - al) + patch * al).clip(0, 255).astype(np.uint8)
            print(n, t, 'match %.2f' % sc)
    return img, G
if __name__ == '__main__':
    import sys
    allr = {}
    for n in range(1, 6):
        img = cv2.imread(f'in{n}.jpg')
        dbg = img.copy()
        if n in WIN:
            img, G = swap(n, img)
            for m, (x0, y0, x1, y1), (X0, Y0, X1, Y1) in G:
                cv2.rectangle(dbg, (X0, Y0), (X1, Y1), (0, 255, 255), 2); cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.imwrite(f'out/board{n}-tall.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 86])
        H, W = img.shape[:2]
        for k, r in enumerate(rects(n)):
            x, y, w, h = r; cv2.rectangle(dbg, (int(x * W / 100), int(y * H / 100)), (int((x + w) * W / 100), int((y + h) * H / 100)), (0, 255, 0), 3)
        cv2.imwrite(f'dbg{n}.jpg', cv2.resize(dbg, None, fx=0.3, fy=0.3))
        allr[str(n)] = [[round(v, 2) for v in r] for r in rects(n)]
    json.dump(allr, open('rects.json', 'w'))
