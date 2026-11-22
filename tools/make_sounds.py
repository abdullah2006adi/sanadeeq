"""Synthesise the built-in sound library for الصناديق السرية (original sounds, no samples)."""
import numpy as np, json, subprocess, os, sys
from scipy.signal import fftconvolve, butter, sosfilt, lfilter, resample_poly

SR = 44100
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/sanadeeq/sounds"
WAVDIR = "/home/claude/sfx/wav"; os.makedirs(WAVDIR, exist_ok=True)
rng = np.random.default_rng(20261122)

# ---------- basic helpers ----------
def T(d, sr=SR): return np.arange(int(round(d * sr))) / sr
def N(name):
    k = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
    return 440 * 2 ** ((k[name[:-1]] + (int(name[-1]) - 4) * 12) / 12)
def sos(kind, f, order=2): return butter(order, f, btype=kind, fs=SR, output='sos')
def bp(x, lo, hi, o=2): return sosfilt(sos('bandpass', [lo, min(hi, SR / 2 * 0.95)], o), x)
def hp(x, f, o=2): return sosfilt(sos('highpass', f, o), x)
def lp(x, f, o=2): return sosfilt(sos('lowpass', min(f, SR / 2 * 0.95), o), x)
def stereo(d): return np.zeros((2, int(round((d + 2.5) * SR))))   # spare room so tails are never cut
def put(buf, sig, at, pan=0.0, gain=1.0):
    """add mono sig into stereo buf at time `at` with constant-power pan (-1 left .. +1 right)"""
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    i = int(round(at * SR)); j = min(buf.shape[1], i + len(sig)); n = j - i
    if n <= 0: return
    seg = np.array(sig[:n], dtype=float)
    f = min(int(0.04 * SR), max(1, n // 4))                 # soft end, so no note is ever cut off with a click
    seg[-f:] *= np.cos(np.linspace(0, np.pi / 2, f)) ** 2
    buf[0, i:j] += seg * l * gain; buf[1, i:j] += seg * r * gain
def put2(buf, st, at, gain=1.0):
    i = int(round(at * SR)); j = min(buf.shape[1], i + st.shape[1]); buf[:, i:j] += st[:, :j - i] * gain
def mix(y, z, g=1.0):
    n = min(len(y), len(z)); y[:n] += z[:n] * g; return y
def att(t, a): return np.clip(t / a, 0, 1) if a > 0 else 1.0

# ---------- instruments ----------
def glock(f, dur=1.8, bright=1.0, decay=1.0):
    t = T(dur)
    parts = [(1.0, 1.0, 0.95), (2.756, 0.42 * bright, 0.32), (5.404, 0.2 * bright, 0.14), (8.933, 0.09 * bright, 0.07)]
    y = np.zeros_like(t)
    for r, a, tau in parts:
        if f * r < SR * 0.45: y += a * np.sin(2 * np.pi * f * r * t + rng.uniform(0, 6.28)) * np.exp(-t / (tau * decay))
    strike = hp(rng.standard_normal(len(t)) * np.exp(-t / 0.0015), 2500) * 0.12
    return (y + strike) * att(t, 0.0012)

def musicbox(f, dur=1.6):
    t = T(dur)
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.75) + 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.25)
         + 0.12 * np.sin(2 * np.pi * 5.9 * f * t) * np.exp(-t / 0.05))
    return y * att(t, 0.001)

def fmbell(f, dur=2.2, ratio=1.4, index=2.6, tau=1.0, itau=0.35):
    t = T(dur); I = index * np.exp(-t / itau)
    return np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t)) * np.exp(-t / tau) * att(t, 0.002)

def ping(f, tau=0.05):
    t = T(tau * 7); return np.sin(2 * np.pi * f * t) * np.exp(-t / tau) * att(t, 0.0006)

def sparkles(buf, start, dur, n, fmin=3500, fmax=9500, amp=0.12, skew=1.6):
    for _ in range(n):
        at = start + dur * rng.random() ** skew
        put(buf, ping(rng.uniform(fmin, fmax), rng.uniform(0.018, 0.07)), at, rng.uniform(-0.85, 0.85), amp * rng.uniform(0.35, 1))

def shimmer(dur, amp=0.035, lo=5500, hi=13000):
    t = T(dur); n = bp(rng.standard_normal(len(t)), lo, hi)
    return n * (0.55 + 0.45 * np.sin(2 * np.pi * 11 * t + rng.uniform(0, 6))) * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2 * amp

def brass(f, dur, rel=0.18, vib=True, attack=0.028):
    t = T(dur + rel); y = np.zeros_like(t)
    cutoff = 500 + 3300 * (1 - np.exp(-t / 0.025)) * (0.68 + 0.32 * np.exp(-t / 0.12))
    fm = f * (1 + (0.004 * np.sin(2 * np.pi * 5.6 * t) * np.clip((t - 0.25) / 0.2, 0, 1) if vib else 0))
    phase = 2 * np.pi * np.cumsum(fm) / SR
    for k in range(1, 30):
        if k * f > 11000: break
        w = (1 / k ** 1.05) / (1 + (k * f / cutoff) ** 4)
        y += w * np.sin(k * phase + rng.uniform(0, 6.28))
    env = att(t, attack) * (0.86 + 0.14 * np.exp(-t / 0.09)) * np.clip((dur + rel - t) / rel, 0, 1) ** 1.5
    return y * env

def snare(vel=1.0):
    t = T(0.25)
    noise = bp(rng.standard_normal(len(t)), 1300, 7500) * np.exp(-t / 0.055)
    tone = 0.45 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.045) + 0.2 * np.sin(2 * np.pi * 330 * t) * np.exp(-t / 0.03)
    return (noise * 0.9 + tone) * att(t, 0.0008) * vel

def crash(dur=1.8):
    t = T(dur)
    n = hp(rng.standard_normal(len(t)), 3200, 4) * np.exp(-t / 0.75)
    metal = sum(np.sin(2 * np.pi * f * t + 3 * np.sin(2 * np.pi * f * 1.48 * t)) for f in (3150, 4270, 5630, 7020)) * 0.05 * np.exp(-t / 0.5)
    return (n * 0.6 + metal) * att(t, 0.001)

def kick(f0=70, dur=0.5):
    t = T(dur); f = f0 * (1 + 1.6 * np.exp(-t / 0.03))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16) * att(t, 0.001)

def ks_pluck(f, dur=1.6, g=0.996, bright=0.6):
    up = 4; sr = SR * up; Nn = int(round(sr / f))
    x = np.zeros(int(dur * sr)); burst = rng.standard_normal(Nn)
    burst = lfilter([bright, 1 - bright], [1], burst)            # soften the attack
    x[:Nn] = burst
    a = np.zeros(Nn + 2); a[0] = 1; a[Nn] = -g / 2; a[Nn + 1] = -g / 2
    y = lfilter([1.0], a, x)
    y = resample_poly(y, 1, up)
    return y / (np.abs(y).max() + 1e-9)

def square(f, dur, duty_soft=5000):
    t = T(dur); y = np.zeros_like(t)
    for k in range(1, 40, 2):
        if k * f > 12000: break
        y += np.sin(2 * np.pi * k * f * t) / k / (1 + (k * f / duty_soft) ** 2)
    return y

def clap_templates(n=28):
    out = []
    for _ in range(n):
        t = T(0.09); c = rng.uniform(850, 2300)
        y = bp(rng.standard_normal(len(t)), c / 1.7, c * 1.7) * np.exp(-t / rng.uniform(0.008, 0.016)) * att(t, 0.0004)
        out.append(y / (np.abs(y).max() + 1e-9))
    return out
CLAPS = clap_templates()

def svf_bandpass(x, fc, q=2.2):
    y = np.zeros_like(x); low = band = 0.0; qi = 1 / q
    for i in range(len(x)):
        f = 2 * np.sin(np.pi * fc[i] / SR)
        high = x[i] - low - qi * band
        band += f * high; low += f * band; y[i] = band
    return y

# ---------- reverb ----------
def make_ir(rt60, dur, predelay=0.012, damp=5000):
    t = T(dur); tau = rt60 / 6.91; chans = []
    for _ in range(2):
        n = rng.standard_normal(len(t)) * np.exp(-t / tau)
        w = np.clip(t / dur * 1.6, 0, 1); n = n * (1 - w) + lp(n, damp * 0.4) * w
        n[: int(predelay * SR)] = 0
        for k in range(7):
            n[int((predelay + rng.uniform(0.004, 0.05)) * SR)] += rng.uniform(0.15, 0.45) * rng.choice([-1, 1])
        chans.append(n / np.sqrt(np.sum(n ** 2)))
    return np.stack(chans)
ROOM, HALL = make_ir(0.7, 1.2), make_ir(1.7, 2.6, predelay=0.02)
def reverb(st, wet, ir=HALL):
    out = np.pad(st, ((0, 0), (0, ir.shape[1] - 1)))
    for c in range(2): out[c] += fftconvolve(st[c], ir[c]) * wet
    return out

# ---------- finishing ----------
def biquad(kind, f0, q, gain_db=0.0):
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f0 / SR; c, sn = np.cos(w), np.sin(w); al = sn / (2 * q)
    if kind == "highshelf":
        b0 = A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al); b1 = -2 * A * ((A - 1) + (A + 1) * c); b2 = A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)
        a0 = (A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al; a1 = 2 * ((A - 1) - (A + 1) * c); a2 = (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al
    else:  # highpass
        b0 = (1 + c) / 2; b1 = -(1 + c); b2 = (1 + c) / 2; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al
    return np.array([b0, b1, b2]) / a0, np.array([1, a1 / a0, a2 / a0])
KW = [biquad("highshelf", 1500, 1 / np.sqrt(2), 4.0), biquad("highpass", 38, 0.5)]   # ITU-R BS.1770 K-weighting
def max_momentary_lufs(st):
    y = st.copy()
    for b, a in KW: y = lfilter(b, a, y, axis=1)
    w = min(y.shape[1], int(0.4 * SR)); ms = 0
    for c in range(2):
        cs = np.cumsum(np.pad(y[c] ** 2, (1, 0))); ms = ms + (cs[w:] - cs[:-w]) / w
    return -0.691 + 10 * np.log10(ms.max() + 1e-12)

def finalize(st, target_lufs=-12.0, ceiling=0.89):
    st = sosfilt(sos('highpass', 25, 2), st, axis=1)          # remove DC without shifting the silent tail
    g = 10 ** ((target_lufs - max_momentary_lufs(st)) / 20)
    g = min(g, 1.35 / (np.abs(st).max() + 1e-9))               # allow at most ~2.6 dB of soft limiting
    st = st * g
    th = 0.72; a = np.abs(st)
    st = np.where(a > th, np.sign(st) * (th + (1 - th) * np.tanh((a - th) / (1 - th))), st) * ceiling
    env = np.abs(st).max(0); idx = np.where(env > 10 ** (-52 / 20))[0]
    st = st[:, : (idx[-1] + int(0.05 * SR)) if len(idx) else st.shape[1]]
    fi, fo = int(0.002 * SR), int(0.12 * SR)
    st[:, :fi] *= np.linspace(0, 1, fi); st[:, -fo:] *= np.linspace(1, 0, fo) ** 2
    return st

# ---------- the sounds ----------
def s_sparkle():
    b = stereo(1.3)
    for k, n in enumerate(['E6', 'G#6', 'B6', 'E7', 'G#7']):
        put(b, glock(N(n), 1.2, 0.9), 0.055 * k, -0.5 + 0.25 * k, 0.55 - 0.05 * k)
    sparkles(b, 0.04, 0.9, 26, amp=0.13); put(b, shimmer(1.0), 0.0, 0.3); put(b, shimmer(1.0), 0.05, -0.3)
    return reverb(b, 0.38)

def s_chime():
    b = stereo(1.6)
    put(b, glock(N('C5'), 1.6, 0.6), 0, 0, 0.35)
    for k, n in enumerate(['C6', 'E6', 'G6', 'C7']): put(b, glock(N(n), 1.5), 0.09 * k, -0.3 + 0.2 * k, 0.5)
    for k, n in enumerate(['E6', 'G6', 'C7']): put(b, glock(N(n), 1.4, 0.7), 0.42 + 0.012 * k, 0.2 - 0.2 * k, 0.28)
    sparkles(b, 0.35, 0.6, 8, amp=0.07)
    return reverb(b, 0.32)

def clink(f0):
    t = T(0.35); y = np.zeros_like(t)
    for r, a, tau in [(1, 1, 0.09), (1.47, .7, .07), (2.09, .5, .05), (2.57, .35, .04), (3.14, .25, .03)]:
        if f0 * r < SR * 0.45: y += a * np.sin(2 * np.pi * f0 * r * t + rng.uniform(0, 6.28)) * np.exp(-t / (tau * rng.uniform(0.7, 1.3)))
    y += hp(rng.standard_normal(len(t)) * np.exp(-t / 0.002), 4000) * 0.3
    return y * att(t, 0.0005)

def s_coins():
    b = stereo(1.5)
    times = np.sort(np.clip(0.02 + rng.gamma(1.5, 0.13, 30), 0, 1.05))
    for i, at in enumerate(times):
        put(b, clink(rng.uniform(2300, 4300)), at, rng.uniform(-0.8, 0.8), 0.5 * (1.0 - 0.55 * i / len(times)) * rng.uniform(0.5, 1))
    put(b, fmbell(N('B6'), 0.8, 3.5, 1.2, 0.25), 0.03, 0.2, 0.18); put(b, fmbell(N('E7'), 0.8, 3.5, 1.2, 0.2), 0.07, -0.2, 0.14)
    return reverb(b, 0.18, ROOM)

def s_chest():
    b = stereo(2.0); d = 0.55; t = T(d)
    f = 55 + 150 * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.5
    f *= 1 + 0.08 * rng.standard_normal(len(t)).cumsum() / np.sqrt(np.arange(1, len(t) + 1)) * 0.3
    ph = np.cumsum(f) / SR; imp = np.zeros_like(t); imp[np.where(np.diff(np.floor(ph)) > 0)[0]] = 1
    imp *= rng.uniform(0.7, 1.3, len(t))
    wood = sum(w * bp(imp, fc / 1.12, fc * 1.12) for fc, w in [(420, 1.0), (950, 0.8), (1800, 0.5), (3100, 0.3)])
    creak = hp(wood, 150) * np.clip(t / 0.04, 0, 1) * np.clip((d - t) / 0.08, 0, 1)
    put(b, creak / (np.abs(creak).max() + 1e-9), 0.04, 0.15, 0.55)
    tt = T(0.08); latch = mix(hp(rng.standard_normal(len(tt)) * np.exp(-tt / 0.003), 2000) * 0.5, ping(3200, 0.015), 0.4)
    put(b, latch, 0.0, -0.1, 0.6)
    tt = T(0.4); thump = np.sin(2 * np.pi * np.cumsum(95 - 25 * tt / 0.4) / SR) * np.exp(-tt / 0.08) + lp(rng.standard_normal(len(tt)), 600) * np.exp(-tt / 0.03) * 0.4
    put(b, thump, 0.6, 0, 0.6)
    for k, n in enumerate(['G5', 'B5', 'D6', 'G6']): put(b, glock(N(n), 1.3), 0.64 + 0.03 * k, -0.3 + 0.2 * k, 0.32)
    sparkles(b, 0.64, 0.75, 14, amp=0.1); put(b, shimmer(0.9, 0.03), 0.62, 0)
    return reverb(b, 0.22, ROOM)

def s_lock():
    b = stereo(1.4)
    def click(big=False):
        tt = T(0.12); y = hp(rng.standard_normal(len(tt)) * np.exp(-tt / 0.0025), 1500) * 0.6
        mix(y, ping(rng.uniform(2500, 4500), rng.uniform(0.01, 0.025)), 0.5)
        if big: y += np.sin(2 * np.pi * 300 * tt) * np.exp(-tt / 0.015) * 0.6
        return y
    put(b, click(), 0.0, 0.1, 0.45); put(b, click(), 0.11, 0.1, 0.45); put(b, click(True), 0.26, 0.05, 0.7)
    put(b, ping(5200, 0.06), 0.34, -0.1, 0.25)
    put(b, fmbell(N('C6'), 1.2, 3.5, 1.4, 0.45, 0.2), 0.46, -0.2, 0.3); put(b, fmbell(N('G6'), 1.1, 3.5, 1.4, 0.4, 0.2), 0.52, 0.2, 0.26)
    sparkles(b, 0.46, 0.5, 8, amp=0.07)
    return reverb(b, 0.16, ROOM)

def s_drumroll():
    b = stereo(2.4); at, k = 0.0, 0
    while at < 0.66:
        rate = 14 + 16 * (at / 0.66) ** 1.3; vel = 0.28 + 0.6 * (at / 0.66) ** 1.5
        put(b, snare(vel * rng.uniform(0.85, 1.1)), at, 0.15 if k % 2 else -0.15, 0.7); at += 1 / rate; k += 1
    put(b, kick(), 0.7, 0, 0.8); put(b, crash(), 0.7, 0.1, 0.55); put(b, snare(1.0), 0.7, 0, 0.5)
    for k, n in enumerate(['C6', 'E6', 'G6', 'C7']): put(b, glock(N(n), 1.5), 0.7 + 0.008 * k, -0.3 + 0.2 * k, 0.4)
    sparkles(b, 0.72, 0.8, 14, amp=0.1)
    return reverb(b, 0.24)

def s_tada():
    b = stereo(2.0)
    for n, p in [('G4', -0.3), ('B4', 0.0), ('D5', 0.3)]: put(b, brass(N(n) * 2 ** (rng.uniform(-3, 3) / 1200), 0.14, 0.05, vib=False, attack=0.012), rng.uniform(0, 0.008), p, 0.42)
    for n, p in [('C5', -0.35), ('E5', -0.1), ('G5', 0.15), ('C6', 0.35)]:
        put(b, brass(N(n) * 2 ** (rng.uniform(-3, 3) / 1200), 1.05, 0.25), 0.2 + rng.uniform(0, 0.012), p, 0.28)
    put(b, kick(65, 0.6), 0.2, 0, 0.5); put(b, crash(1.0), 0.2, 0, 0.07)
    sparkles(b, 0.25, 1.0, 10, amp=0.06)
    return reverb(b, 0.3)

def s_harp():
    b = stereo(2.6); notes = ['C5', 'D5', 'E5', 'G5', 'A5', 'C6', 'D6', 'E6', 'G6', 'A6', 'C7']
    for k, n in enumerate(notes):
        y = ks_pluck(N(n), 1.8, 0.997 if k < 6 else 0.995, 0.5); y *= np.exp(-T(len(y) / SR) / 0.9)
        put(b, y, 0.045 * k, -0.7 + 1.4 * k / (len(notes) - 1), 0.34)
    sparkles(b, 0.45, 0.6, 6, amp=0.05)
    return reverb(b, 0.36)

def s_musicbox():
    b = stereo(2.6)
    melody = [('G6', 0.0), ('C7', 0.16), ('E7', 0.32), ('D7', 0.48), ('C7', 0.64), ('G6', 0.8), ('C7', 1.04)]
    for n, at in melody: put(b, musicbox(N(n), 1.4), at, rng.uniform(-0.25, 0.25), 0.5)
    put(b, musicbox(N('C6'), 1.6), 0.0, -0.2, 0.3); put(b, musicbox(N('G5'), 1.6), 0.48, 0.2, 0.28); put(b, musicbox(N('C6'), 1.6), 1.04, 0, 0.3)
    return reverb(b, 0.33)

def s_pop():
    b = stereo(0.8)
    for at, f0, f1, g in [(0.0, 330, 1300, 0.8), (0.085, 520, 2000, 0.55)]:
        t = T(0.12); f = f0 * (f1 / f0) ** np.clip(t / 0.025, 0, 1)
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.03) * att(t, 0.0008)
        y += hp(rng.standard_normal(len(t)) * np.exp(-t / 0.001), 2000) * 0.2
        put(b, y, at, rng.uniform(-0.2, 0.2), g)
    put(b, ping(2600, 0.05), 0.15, -0.3, 0.2); put(b, ping(3500, 0.04), 0.2, 0.3, 0.16); sparkles(b, 0.15, 0.3, 6, amp=0.07)
    return reverb(b, 0.12, ROOM)

def s_powerup():
    b = stereo(1.2); notes = ['C5', 'E5', 'G5', 'C6', 'E6', 'G6', 'C7']
    for k, n in enumerate(notes):
        last = k == len(notes) - 1; d = 0.36 if last else 0.065; t = T(d)
        y = square(N(n), d) * att(t, 0.003) * np.clip((d - t) / 0.02, 0, 1)
        if last: y *= np.exp(-t / 0.22) * (1 + 0.15 * np.sin(2 * np.pi * 9 * t))
        put(b, y, 0.055 * k, 0, 0.22)
    sparkles(b, 0.38, 0.5, 8, amp=0.07)
    return reverb(b, 0.12, ROOM)

def s_whoosh():
    b = stereo(1.5); d = 0.62; t = T(d)
    fc = np.where(t < 0.45, 250 * (2800 / 250) ** (t / 0.45), 2800 * (1800 / 2800) ** ((t - 0.45) / 0.17))
    y = svf_bandpass(rng.standard_normal(len(t)), fc, 2.0)
    env = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.6
    y = y / (np.abs(y).max() + 1e-9) * env
    for i in range(0, len(t), 512):   # sweep right → left
        put(b, y[i:i + 512], i / SR, 0.75 - 1.5 * i / len(t), 0.75)
    for k, n in enumerate(['B6', 'E7']): put(b, glock(N(n), 1.0, 0.8), 0.45 + 0.07 * k, -0.4 + 0.3 * k, 0.32)
    sparkles(b, 0.45, 0.6, 14, amp=0.1); put(b, shimmer(0.8, 0.03), 0.42, -0.3)
    return reverb(b, 0.3)

def s_ding():
    b = stereo(1.3)
    for at, n in [(0.0, 'C6'), (0.13, 'G6')]:
        put(b, fmbell(N(n), 1.1, 3.5, 1.5, 0.42, 0.12), at, -0.15 if at == 0 else 0.15, 0.45)
        put(b, np.sin(2 * np.pi * N(n) * T(1.0)) * np.exp(-T(1.0) / 0.35) * att(T(1.0), 0.002), at, 0, 0.3)
    return reverb(b, 0.2, ROOM)

def s_bell():
    b = stereo(2.8)
    put(b, fmbell(N('E5'), 2.6, 1.4, 2.4, 1.15, 0.4), 0, 0, 0.55)
    put(b, fmbell(N('E6'), 1.6, 1.4, 1.2, 0.55, 0.2), 0.005, 0.1, 0.18)
    return reverb(b, 0.36)

def s_applause(d=3.0):
    b = stereo(d + 0.2)
    for _ in range(16):
        at = rng.uniform(0, 0.25); rate = rng.uniform(3.4, 5.6); pan = rng.uniform(-0.85, 0.85); amp = rng.uniform(0.35, 1)
        while at < d:
            fade = min(1, at / 0.3) * min(1, max(0, (d - at) / 1.0))
            put(b, CLAPS[rng.integers(len(CLAPS))], at, pan, 0.22 * amp * fade * rng.uniform(0.7, 1.1))
            at += 1 / rate * rng.uniform(0.85, 1.15)
    t = T(d); bed = lp(rng.standard_normal(len(t)), 3000) * 0.012 * np.clip(t / 0.3, 0, 1) * np.clip((d - t) / 1.0, 0, 1)
    put(b, bed, 0, -0.3); put(b, lp(rng.standard_normal(len(t)), 3000) * 0.012 * np.clip(t / 0.3, 0, 1) * np.clip((d - t) / 1.0, 0, 1), 0, 0.3)
    return reverb(b, 0.32)

def s_celebrate():
    a, c = s_tada(), s_applause(2.8)
    b = np.zeros((2, max(a.shape[1], c.shape[1] + int(0.25 * SR))))
    put2(b, a, 0, 1.0); put2(b, c, 0.25, 0.75); sparkles(b, 0.2, 1.4, 18, amp=0.08)
    return b

LIB = [
    ("sparkle", "بريق سحري", s_sparkle), ("chime", "جرس الكنز", s_chime), ("coins", "عملات ذهبية", s_coins),
    ("chest", "صندوق خشبي يُفتح", s_chest), ("lock", "قفل ومفتاح", s_lock), ("drumroll", "طبول ثم مفاجأة", s_drumroll),
    ("tada", "تا-دا!", s_tada), ("harp", "قيثارة", s_harp), ("musicbox", "صندوق موسيقى", s_musicbox),
    ("pop", "فقاعة", s_pop), ("powerup", "مكافأة الألعاب", s_powerup), ("whoosh", "نسمة سحرية", s_whoosh),
    ("ding", "إجابة صحيحة", s_ding), ("bell", "جرس هادئ", s_bell), ("applause", "تصفيق", s_applause),
    ("celebrate", "احتفال", s_celebrate),
]

if __name__ == "__main__":
    import scipy.io.wavfile as wf
    os.makedirs(OUT, exist_ok=True); lib = []
    for sid, name, fn in LIB:
        st = finalize(fn()); wav = f"{WAVDIR}/{sid}.wav"
        wf.write(wav, SR, (st.T * 32767).astype(np.int16))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame", "-q:a", "4", "-ar", "44100", f"{OUT}/{sid}.mp3"], check=True)
        dur = round(st.shape[1] / SR, 2); lib.append({"id": sid, "name": name, "src": f"sounds/{sid}.mp3", "dur": dur})
        print(f"{sid:10s} {name:18s} {dur:5.2f}s  LUFSmax {max_momentary_lufs(st):6.1f}  peak {np.abs(st).max():.2f}  {os.path.getsize(OUT + '/' + sid + '.mp3') // 1024} KB")
    json.dump(lib, open(f"{OUT}/library.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
