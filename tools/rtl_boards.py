import json, sys, numpy as np, cv2
cfg = {b['id']: b for b in json.load(open('/home/claude/sanadeeq/config.json'))['boards']}
# window (x0,x1,y0,y1 as fraction of rect), glyph polarity
WIN = {2: ((0.30, 0.70, 0.60, 0.90), 'dark'), 3: ((0.37, 0.65, 0.04, 0.27), 'light'),
       4: ((0.35, 0.68, 0.0, 0.26), 'dark'), 5: ((0.39, 0.645, 0.25, 0.52), 'light')}
def glyph(img, rect, n):
    H, W = img.shape[:2]; (fx0, fx1, fy0, fy1), pol = WIN[n]
    x, y, w, h = [v / 100 for v in rect]
    X0, X1, Y0, Y1 = int((x + fx0 * w) * W), int((x + fx1 * w) * W), int((y + fy0 * h) * H), int((y + fy1 * h) * H)
    g = cv2.cvtColor(img[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, 0, 255, (cv2.THRESH_BINARY_INV if pol == 'dark' else cv2.THRESH_BINARY) + cv2.THRESH_OTSU)
    k, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m); hh, ww = m.shape
    for i in range(1, k):
        bx, by, bw, bh, area = st[i]
        if area < 25 or bx == 0 or bx + bw >= ww or by + bh >= hh: continue
        if bh < 0.18 * hh: continue
        keep[lab == i] = 255
    full = np.zeros((H, W), np.uint8); full[Y0:Y1, X0:X1] = keep
    ys, xs = np.nonzero(full)
    return full, (xs.min(), ys.min(), xs.max(), ys.max())
def run(n, out):
    b = cfg[f'board{n}']; img = cv2.imread(f"/home/claude/sanadeeq/{b['src']}"); orig = img.copy()
    R = b['rects']; G = [glyph(orig, r, n) for r in R]
    # erase all glyphs
    allm = np.zeros(img.shape[:2], np.uint8)
    for m, _ in G: allm |= m
    allm = cv2.dilate(allm, np.ones((7, 7), np.uint8))
    img = cv2.inpaint(img, allm, 6, cv2.INPAINT_TELEA)
    for row in (0, 5):
        for c in range(5):
            t, s = row + c, row + 4 - c
            m, (sx0, sy0, sx1, sy1) = G[s]; _, (tx0, ty0, tx1, ty1) = G[t]
            p = 4; sx0 -= p; sy0 -= p; sx1 += p; sy1 += p
            patch = orig[sy0:sy1 + 1, sx0:sx1 + 1].astype(np.float32)
            a = cv2.dilate(m, np.ones((3, 3), np.uint8))[sy0:sy1 + 1, sx0:sx1 + 1]
            a = cv2.GaussianBlur(a, (3, 3), 0).astype(np.float32) / 255
            sc = (R[t][2] / R[s][2] + R[t][3] / R[s][3]) / 2
            nw, nh = max(1, round(patch.shape[1] * sc)), max(1, round(patch.shape[0] * sc))
            patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_CUBIC); a = cv2.resize(a, (nw, nh))[..., None]
            cx = (tx0 + tx1) / 2; cy = (ty0 + ty1) / 2
            # keep the vertical center of the source digits relative to the plate
            X = int(round(cx - nw / 2)); Y = int(round(cy - nh / 2))
            roi = img[Y:Y + nh, X:X + nw].astype(np.float32)
            img[Y:Y + nh, X:X + nw] = (roi * (1 - a) + patch * a).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return G
for n in map(int, sys.argv[1:]):
    run(n, f'/home/claude/rtl/board{n}.jpg'); print('done', n)
