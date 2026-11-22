"""OneLinker CRM reel builder (runs in GitHub Actions).

For every content/<id>.json without out/<id>.mp4:
  1. voiceover per scene via ElevenLabs (secret ELEVENLABS_API_KEY)
  2. real CRM screens from www.onelinker.ai
  3. 1080x1920 tutorial-style motion graphics (same look as OneLinker_Tutorial_Final), timed to the voice
  4. writes out/<id>.mp4, out/<id>-preview.jpg, out/<id>.json
"""
import glob, io, json, math, os, subprocess, sys, traceback, wave, zipfile
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
NAVY = (13, 31, 92); BLUE = (29, 91, 240); LIGHT = (232, 240, 254); BG = (246, 248, 252); GREY = (90, 102, 130)
CACHE = '.cache'; os.makedirs(CACHE, exist_ok=True); os.makedirs('out', exist_ok=True)
SITE = 'https://www.onelinker.ai'
MILES = 'rNzVNTrvSffyxdrTbLKv'  # Miles - Technical British Narrator (old tutorial voice)
DEFAULT_VOICE = '3svOJAOhuPHXwQC2H5eq'  # Brady J - friendly, natural, conversational (more human for reels)
LOG = []

def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

# ---------- assets ----------
def fonts():
    d = os.path.join(CACHE, 'inter')
    if not os.path.exists(d):
        z = requests.get('https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip', timeout=120).content
        zipfile.ZipFile(io.BytesIO(z)).extractall(d)
    base = os.path.join(d, 'extras', 'otf')
    return {k: os.path.join(base, f'InterDisplay-{v}.otf') for k, v in
            {'xb': 'ExtraBold', 'b': 'Bold', 'sb': 'SemiBold', 'm': 'Medium'}.items()}
FP = fonts()
_fc = {}
def F(w, s):
    k = (w, s)
    if k not in _fc: _fc[k] = ImageFont.truetype(FP[w], s)
    return _fc[k]

def site_img(path):
    if os.path.exists(path):  # mockups saved in the repo (assets/mockups/...)
        return Image.open(path).convert('RGBA')
    local = os.path.join(CACHE, path.replace('/', '_'))
    if not os.path.exists(local):
        r = requests.get(f'{SITE}/{path.lstrip("/")}', timeout=60); r.raise_for_status()
        open(local, 'wb').write(r.content)
    return Image.open(local).convert('RGBA')

def logo_img():
    for p in ['assets/mockups/logo/onelinker-logo.png', 'icon.png', 'logo-128w.png']:
        try:
            im = site_img(p)
            a = np.array(im); white = a[..., :3].min(axis=2) > 240; a[white, 3] = 0
            return Image.fromarray(a)
        except Exception as e:
            log('logo fail', p, e)
    return Image.new('RGBA', (256, 256), BLUE + (255,))

# ---------- voice ----------
def tts(text, voice, out_wav, prev_text='', next_text=''):
    key = os.environ.get('ELEVENLABS_API_KEY')
    if not key: raise RuntimeError('ELEVENLABS_API_KEY secret is missing')
    r = requests.post(f'https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128',
                      headers={'xi-api-key': key, 'Content-Type': 'application/json'},
                      json={'text': text, 'model_id': 'eleven_multilingual_v2',
                            'previous_text': prev_text or None, 'next_text': next_text or None,
                            'voice_settings': {'stability': 0.4, 'similarity_boost': 0.75, 'style': 0.3, 'use_speaker_boost': True, 'speed': 1.0}},
                      timeout=120)
    if r.status_code != 200: raise RuntimeError(f'ElevenLabs {r.status_code}: {r.text[:300]}')
    mp3 = out_wav.replace('.wav', '.mp3'); open(mp3, 'wb').write(r.content)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', mp3, '-ac', '1', '-ar', '44100', out_wav], check=True)
    with wave.open(out_wav) as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768

# ---------- drawing helpers ----------
def ease(t): t = min(max(t, 0), 1); return 1 - (1 - t) ** 3
def back(t):
    t = min(max(t, 0), 1); c = 1.7
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2

def rounded(img, r):
    m = Image.new('L', img.size, 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], r, fill=255)
    out = img.convert('RGBA'); a = np.minimum(np.array(out.getchannel('A')), np.array(m)); out.putalpha(Image.fromarray(a)); return out

_sh = {}
def shadow(size, r, blur=40, alpha=55):
    k = (size, r, blur, alpha)
    if k not in _sh:
        s = Image.new('RGBA', (size[0] + blur * 4, size[1] + blur * 4), (0, 0, 0, 0))
        ImageDraw.Draw(s).rounded_rectangle([blur * 2, blur * 2 + 18, blur * 2 + size[0], blur * 2 + size[1] + 18], r, fill=NAVY + (alpha,))
        _sh[k] = (s.filter(ImageFilter.GaussianBlur(blur)), blur * 2)
    return _sh[k]

def fit(img, bw, bh):
    s = min(bw / img.width, bh / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)

def wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if font.getlength(t) <= maxw: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

class Reel:
    """Tutorial-style vertical reel (matches OneLinker_Tutorial_Final): white background, logo top bar,
    step pills, real CRM mockups with smooth camera zooms, step cards, subtitle box, progress bar, QR end card."""
    VX, VY, VW, VH = 40, 300, 1000, 1180   # viewport for screens

    def __init__(self, spec):
        self.spec = spec; self.logo = logo_img()
        self.steps = [sc for sc in spec['scenes'] if sc['type'] == 'feature']
        self.shots = {}
        for i, sc in enumerate(spec['scenes']):
            if sc.get('screen'):
                img = site_img(sc['screen'])
                if np.array(img.getchannel('A')).min() > 250:   # plain screenshot -> rounded window look
                    img = rounded(img, max(12, img.width // 60))
                self.shots[i] = img
        try: self.qr = site_img('assets/mockups/logo/signup-qr.png')
        except Exception:
            try: self.qr = site_img('signup-qr.png')
            except Exception: self.qr = None
        self.bg = self._bg()
        self.timeline = None; self.clipspan = None

    def _bg(self):
        im = Image.new('RGBA', (W, H), (255, 255, 255, 255))
        g = Image.new('RGBA', (W // 4, H // 4), (0, 0, 0, 0)); d = ImageDraw.Draw(g)
        d.ellipse([-80, 300, 200, 560], fill=(29, 91, 240, 22)); d.ellipse([150, -60, 360, 140], fill=(29, 91, 240, 14))
        im.alpha_composite(g.filter(ImageFilter.GaussianBlur(40)).resize((W, H), Image.BILINEAR))
        return im

    def label(self, sc):
        if sc.get('step'): return sc['step']
        h = sc.get('headline', [''])
        first = h[0] if isinstance(h[0], str) else h[0][0]
        return first.rstrip('.')

    def title(self, sc):
        return ' '.join((x if isinstance(x, str) else x[0]) for x in sc.get('headline', []))

    def topbar(self, im, d, cur):
        lg = self.logo.resize((60, 60), Image.LANCZOS); im.alpha_composite(lg, (44, 70))
        f = F('b', 42); d.text((118, 100), 'OneLinker', font=f, fill=NAVY, anchor='lm')
        d.text((118 + f.getlength('OneLinker '), 100), 'CRM', font=f, fill=BLUE, anchor='lm')
        n = len(self.steps)
        if n:
            gap = 12; pw = (W - 88 - gap * (n - 1)) // n; y0 = 170
            fs = F('sb', 25)
            for k, sc in enumerate(self.steps):
                x0 = 44 + k * (pw + gap)
                done, now = k < cur, k == cur
                fill = BLUE if now else ((219, 230, 253) if done else (240, 243, 249))
                tcol = (255, 255, 255) if now else (BLUE if done else (120, 130, 155))
                d.rounded_rectangle([x0, y0, x0 + pw, y0 + 54], 27, fill=fill)
                txt = f'{k + 1}  {self.label(sc)}'
                while fs.getlength(txt) > pw - 28 and len(txt) > 4: txt = txt[:-2] + '…' if not txt.endswith('…') else txt[:-2] + '…'
                d.text((x0 + pw // 2, y0 + 27), txt, font=fs, fill=tcol, anchor='mm')

    def subtitle(self, im, d, text, a=1.0):
        if not text: return
        f = F('sb', 42); lines = wrap(text, f, 860)[:3]
        lh = 56; h = 44 + lh * len(lines); y1 = 1760; y0 = y1 - h
        box = Image.new('RGBA', (W, H), (0, 0, 0, 0)); bd = ImageDraw.Draw(box)
        sh, off = shadow((W - 88, h), 18, 24, 40); box.alpha_composite(sh, (44 - off, y0 - off))
        bd.rounded_rectangle([44, y0, W - 44, y1], 18, fill=(255, 255, 255, 255), outline=(226, 232, 245))
        bd.rounded_rectangle([44, y0, 56, y1], 6, fill=BLUE + (255,))
        for k, ln in enumerate(lines): bd.text((86, y0 + 22 + k * lh), ln, font=f, fill=NAVY, anchor='lt')
        if a < 1: box.putalpha(box.getchannel('A').point(lambda v: int(v * a)))
        im.alpha_composite(box)

    def caption_at(self, i, t):
        sc = self.spec['scenes'][i]; text = sc.get('voice', '')
        if not text or self.clipspan is None: return ''
        st, en = self.clipspan[i]
        words = text.split(); chunks, cur = [], []
        for w in words:
            cur.append(w)
            if len(cur) >= 8 or w.endswith(('.', '?', '!')) and len(cur) >= 3: chunks.append(cur); cur = []
        if cur: chunks.append(cur)
        total = sum(len(c) for c in chunks); acc = 0
        if t < st: return ' '.join(chunks[0])
        for c in chunks:
            acc += len(c)
            if t <= st + (en - st) * acc / total: return ' '.join(c)
        return ' '.join(chunks[-1])

    def camera(self, im, i, s, dur):
        img = self.shots[i]; sc = self.spec['scenes'][i]
        asp = self.VW / self.VH
        def expand(x0, y0, x1, y1):
            w, h = x1 - x0, y1 - y0; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if w / h > asp: h = w / asp
            else: w = h * asp
            return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        full = expand(0, 0, img.width, img.height)
        full = [full[0] - (full[2] - full[0]) * 0.03, full[1] - (full[3] - full[1]) * 0.03,
                full[2] + (full[2] - full[0]) * 0.03, full[3] + (full[3] - full[1]) * 0.03]
        if sc.get('crop'):
            c = sc['crop']; tgt = expand(c[0] * img.width, c[1] * img.height, c[2] * img.width, c[3] * img.height)
        else:
            cw = (full[2] - full[0]) * 0.92; ch = (full[3] - full[1]) * 0.92
            cx, cy = (full[0] + full[2]) / 2, (full[1] + full[3]) / 2
            tgt = [cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2]
        p = ease((s - 1.9) / 1.4)                       # hold full view, then glide in
        drift = 0.02 * ease((s - 3.3) / max(1, dur - 3.3))
        r = [full[k] + (tgt[k] - full[k]) * p for k in range(4)]
        w, h = r[2] - r[0], r[3] - r[1]
        r = [r[0] + w * drift / 2, r[1] + h * drift / 2, r[2] - w * drift / 2, r[3] - h * drift / 2]
        view = img.crop(tuple(int(v) for v in r)).resize((self.VW, self.VH), Image.BILINEAR)
        a = ease((s - 1.0) / 0.5)
        y = self.VY + int((1 - a) * 60)
        if a < 1: view.putalpha(view.getchannel('A').point(lambda v: int(v * a)))
        im.alpha_composite(view, (self.VX, y))
        if sc.get('click') and s > 3.6:                  # optional cursor + click ring at [fx, fy] of the image
            fx, fy = sc['click']; px = (fx * img.width - r[0]) / (r[2] - r[0]) * self.VW + self.VX
            py = (fy * img.height - r[1]) / (r[3] - r[1]) * self.VH + y
            q = ease((s - 3.6) / 0.8); sx, sy = W * 0.8, H * 0.75
            cx, cy = sx + (px - sx) * q, sy + (py - sy) * q
            d = ImageDraw.Draw(im)
            if s > 4.5:
                rr = 20 + 50 * min(1, (s - 4.5) / 0.4); al = int(200 * max(0, 1 - (s - 4.5) / 0.5))
                d.ellipse([px - rr, py - rr, px + rr, py + rr], outline=BLUE + (al,), width=6)
            d.polygon([(cx, cy), (cx, cy + 46), (cx + 12, cy + 35), (cx + 22, cy + 56), (cx + 30, cy + 52), (cx + 20, cy + 32), (cx + 36, cy + 32)],
                      fill=(15, 20, 30), outline=(255, 255, 255))

    def stepcard(self, im, d, k, sc, s):
        a = 1.0 if s < 0.9 else max(0, 1 - (s - 0.9) / 0.35)
        if a <= 0: return
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0)); layer.alpha_composite(self.bg)
        ld = ImageDraw.Draw(layer)
        p = back(s / 0.45); r = int(70 * p)
        ld.ellipse([W // 2 - r, 760 - r, W // 2 + r, 760 + r], fill=BLUE)
        if p > 0.5: ld.text((W // 2, 760), str(k + 1), font=F('xb', 76), fill=(255, 255, 255), anchor='mm')
        q = ease((s - 0.15) / 0.4)
        ld.text((W // 2, 880), f'STEP {k + 1} OF {len(self.steps)}', font=F('b', 30), fill=BLUE + (int(255 * q),), anchor='mm')
        y = 950
        for ln in wrap(self.title(sc), F('xb', 78), 940):
            ld.text((W // 2, y + int((1 - q) * 20)), ln, font=F('xb', 78), fill=NAVY + (int(255 * q),), anchor='mt'); y += 92
        bw = int(160 * ease((s - 0.35) / 0.4)); ld.rounded_rectangle([W // 2 - bw // 2, y + 26, W // 2 + bw // 2, y + 34], 4, fill=BLUE)
        if a < 1: layer.putalpha(layer.getchannel('A').point(lambda v: int(v * a)))
        im.alpha_composite(layer)

    def hook(self, im, d, sc, s):
        p = back(s / 0.5)
        lg = fit(self.logo, int(150 * p) + 1, int(150 * p) + 1); im.alpha_composite(lg, (W // 2 - lg.width // 2, 330 + (150 - lg.height) // 2))
        q = ease((s - 0.2) / 0.4); f = F('xb', 84)
        tw = f.getlength('OneLinker CRM'); x = W // 2 - tw / 2
        d.text((x, 560), 'OneLinker', font=f, fill=NAVY + (int(255 * q),), anchor='lt')
        d.text((x + f.getlength('OneLinker '), 560), 'CRM', font=f, fill=BLUE + (int(255 * q),), anchor='lt')
        y = 720; fh = F('xb', 70)
        for k, item in enumerate(sc.get('headline', [])):
            txt, col = (item, NAVY) if isinstance(item, str) else (item[0], BLUE if item[1] == 'blue' else NAVY)
            for ln in wrap(txt, fh, 960):
                a = ease((s - 0.5 - k * 0.15) / 0.45)
                d.text((W // 2, y + int((1 - a) * 24)), ln, font=fh, fill=col + (int(255 * a),), anchor='mt'); y += 84
        if sc.get('sub'):
            a = ease((s - 1.0) / 0.5)
            for ln in wrap(sc['sub'], F('m', 40), 900):
                d.text((W // 2, y + 24), ln, font=F('m', 40), fill=GREY + (int(255 * a),), anchor='mt'); y += 54
        # step preview pills, like the tutorial intro
        y = max(y + 80, 1180); fs = F('sb', 30)
        for k, st in enumerate(self.steps):
            a = back((s - 1.3 - k * 0.12) / 0.4)
            if s < 1.3 + k * 0.12: continue
            txt = self.label(st); w = int((fs.getlength(txt) + 110) * min(1, a)); x0 = W // 2 - w // 2
            d.rounded_rectangle([x0, y, x0 + w, y + 64], 32, fill=(240, 244, 252), outline=(222, 230, 247))
            if a > 0.8:
                d.ellipse([x0 + 12, y + 12, x0 + 52, y + 52], fill=BLUE); d.text((x0 + 32, y + 32), str(k + 1), font=F('b', 24), fill=(255, 255, 255), anchor='mm')
                d.text((x0 + 68, y + 32), txt, font=fs, fill=NAVY, anchor='lm')
            y += 80

    def cta(self, im, d, sc, s):
        p = back(s / 0.5)
        lg = fit(self.logo, int(110 * p) + 1, int(110 * p) + 1)
        f = F('b', 56); tw = 120 + f.getlength('OneLinker CRM'); x = W // 2 - tw / 2
        im.alpha_composite(lg, (int(x), 420 + (110 - lg.height) // 2))
        d.text((x + 130, 475), 'OneLinker', font=f, fill=NAVY, anchor='lm'); d.text((x + 130 + f.getlength('OneLinker '), 475), 'CRM', font=f, fill=BLUE, anchor='lm')
        lines = sc.get('lines') or ['Every listing.', 'Every lead.']
        y = 600
        for k, ln in enumerate(lines[:2]):
            a = ease((s - 0.3 - k * 0.2) / 0.45)
            d.text((W // 2, y + int((1 - a) * 24)), ln, font=F('xb', 92), fill=(NAVY if k == 0 else BLUE) + (int(255 * a),), anchor='mt'); y += 108
        q = back((s - 0.9) / 0.45)
        if s > 0.9:
            bw, bh = int(560 * q), int(112 * q)
            d.rounded_rectangle([W // 2 - bw // 2, 900 - bh // 2, W // 2 + bw // 2, 900 + bh // 2], 56, fill=BLUE)
            if q > 0.7: d.text((W // 2, 900), sc.get('button', 'Start free trial').replace('  →', '').replace(' →', ''), font=F('b', 46), fill=(255, 255, 255), anchor='mm')
        a = ease((s - 1.2) / 0.4)
        d.text((W // 2, 1010), 'onelinker.ai', font=F('b', 48), fill=NAVY + (int(255 * a),), anchor='mm')
        d.text((W // 2, 1070), 'No credit card  ·  Set up in minutes', font=F('m', 34), fill=GREY + (int(255 * a),), anchor='mm')
        if self.qr is not None and s > 1.4:
            q2 = back((s - 1.4) / 0.45); sz = int(300 * q2) + 1
            qr = self.qr.resize((sz, sz), Image.LANCZOS); bx = Image.new('RGBA', (sz + 40, sz + 40), (255, 255, 255, 255))
            bx = rounded(bx, 24); bx.alpha_composite(qr, (20, 20))
            sh, off = shadow(bx.size, 24, 24, 40); im.alpha_composite(sh, (W // 2 - bx.width // 2 - off, 1150 - off)); im.alpha_composite(bx, (W // 2 - bx.width // 2, 1150))
            d.text((W // 2, 1150 + bx.height + 40), 'Scan to sign up', font=F('sb', 32), fill=GREY, anchor='mm')

    def frame(self, t, timeline):
        for i, (st, du) in enumerate(timeline):
            if st <= t < st + du or i == len(timeline) - 1: break
        sc = self.spec['scenes'][i]; s = t - st
        im = self.bg.copy(); d = ImageDraw.Draw(im)
        if sc['type'] == 'hook':
            self.hook(im, d, sc, s)
        elif sc['type'] == 'cta':
            self.cta(im, d, sc, s)
        else:
            k = self.steps.index(sc)
            self.topbar(im, d, k)
            if i in self.shots: self.camera(im, i, s, timeline[i][1])
            self.stepcard(im, d, k, sc, s)
        if sc['type'] != 'cta':
            a = 1.0 if sc['type'] == 'hook' else ease((s - 1.0) / 0.3)
            self.subtitle(im, d, self.caption_at(i, t), a)
        tot = timeline[-1][0] + timeline[-1][1]
        d.rectangle([0, H - 8, int(W * min(1, t / tot)), H], fill=BLUE)
        return im.convert('RGB')

def music(dur, sr=44100):
    n = int(dur * sr); a = np.zeros(n); rng = np.random.default_rng(3)
    def add(sig, t0, g):
        i = int(t0 * sr); j = min(n, i + len(sig))
        if 0 <= i < n: a[i:j] += sig[:j - i] * g
    beat = 60 / 108; tt = np.arange(int(0.35 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
    ht = np.arange(int(0.05 * sr)) / sr; hat = np.diff(rng.standard_normal(len(ht)) * np.exp(-ht * 70), prepend=0)
    k = 0.0
    while k < dur - 0.4: add(kick, k, 0.5); add(hat, k + beat / 2, 0.1); k += beat
    chords = [[220, 277.2, 329.6], [196, 246.9, 293.7], [174.6, 220, 261.6], [196, 246.9, 293.7]]
    k, ci = 0.0, 0
    while k < dur:
        L = int(beat * 4 * sr); x = np.arange(L) / sr
        env = np.minimum(1, x / 0.4) * np.minimum(1, (beat * 4 - x) / 0.4)
        add(sum(np.sin(2 * np.pi * f * x) for f in chords[ci % 4]) * env, k, 0.05); k += beat * 4; ci += 1
    return a / (np.abs(a).max() + 1e-9)

def build(path):
    spec = json.load(open(path)); rid = spec['id']
    voice = spec.get('voice_id') or DEFAULT_VOICE; sr = 44100
    vtexts = [sc.get('voice', '') for sc in spec['scenes']]
    clips, timeline, spans, t = [], [], [], 0.0
    for i, sc in enumerate(spec['scenes']):
        v = tts(sc['voice'], voice, f'{CACHE}/{rid}_{i}.wav', ' '.join(vtexts[:i])[-500:], ' '.join(vtexts[i+1:])[:500]) if sc.get('voice') else np.zeros(int(sr * 1.5))
        lead = 1.15 if sc['type'] == 'feature' else 0.35
        tail = 0.7 if i < len(spec['scenes']) - 1 else 2.6
        du = max(lead + len(v) / sr + tail, sc.get('min', 3.0) if sc['type'] != 'feature' else 5.2)
        clips.append((t + lead, v)); spans.append((t + lead, t + lead + len(v) / sr)); timeline.append((t, du)); t += du
    dur = min(t, 59.0)
    n = int(dur * sr); vo = np.zeros(n)
    for st, v in clips:
        i = int(st * sr); j = min(n, i + len(v)); vo[i:j] += v[:j - i]
    env = np.convolve(np.abs(vo), np.ones(4410) / 4410, 'same')
    duck = 1 - 0.75 * np.clip(env * 12, 0, 1)
    mix = 0.95 * vo / (np.abs(vo).max() + 1e-9) + 0.12 * music(dur)[:n] * duck
    mix *= np.minimum(1, (dur - np.arange(n) / sr) / 1.0)
    mix = np.clip(mix, -1, 1)
    wav = f'{CACHE}/{rid}_mix.wav'
    with wave.open(wav, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes((mix * 32767).astype(np.int16).tobytes())
    reel = Reel(spec); reel.clipspan = spans
    # preview sheet
    shots = [reel.frame(st + min(max(du * 0.75, 4.0), du - 0.1), timeline).resize((270, 480)) for st, du in timeline]
    sheet = Image.new('RGB', (270 * len(shots), 480), 'white')
    for k, s in enumerate(shots): sheet.paste(s, (k * 270, 0))
    sheet.save(f'out/{rid}-preview.jpg', quality=80)
    mp4 = f'out/{rid}.mp4'
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-i', wav, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21', '-pix_fmt', 'yuv420p',
                          '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', mp4], stdin=subprocess.PIPE)
    for f in range(int(dur * FPS)):
        p.stdin.write(reel.frame(f / FPS, timeline).tobytes())
    p.stdin.close(); p.wait()
    repo = os.environ.get('GITHUB_REPOSITORY', 'ONELINKERCRM/onelinker-shorts')
    meta = {'id': rid, 'duration': round(dur, 2), 'video_url': f'https://raw.githubusercontent.com/{repo}/main/{mp4}',
            'preview_url': f'https://raw.githubusercontent.com/{repo}/main/out/{rid}-preview.jpg',
            'scenes': [[round(a, 2), round(b, 2)] for a, b in timeline]}
    json.dump(meta, open(f'out/{rid}.json', 'w'), indent=2)
    log('built', rid, dur)


# ---------- image posts (Instagram grid / LinkedIn), 1080x1350 ----------
PW, PH = 1080, 1350

def post_slide(sl, logo, idx, total):
    lay = sl.get('layout', 'feature')
    if lay == 'statement':
        arr = np.zeros((PH, PW, 4), np.uint8)
        yy = np.linspace(0, 1, PH)[:, None]; xx = np.linspace(0, 1, PW)[None, :]
        k = np.clip(0.55 * yy + 0.45 * xx, 0, 1)
        for c, (a0, a1) in enumerate([(29, 13), (91, 31), (240, 92)]): arr[..., c] = (a0 + (a1 - a0) * k).astype(np.uint8)
        arr[..., 3] = 255; im = Image.fromarray(arr, 'RGBA'); d = ImageDraw.Draw(im)
        disc = Image.new('RGBA', (96, 96), (0, 0, 0, 0)); ImageDraw.Draw(disc).ellipse([0, 0, 95, 95], fill=(255, 255, 255))
        lg = fit(logo, 66, 66); disc.alpha_composite(lg, ((96 - lg.width) // 2, (96 - lg.height) // 2)); im.alpha_composite(disc, (72, 72))
        d.text((186, 120), 'OneLinker CRM', font=F('b', 40), fill=(255, 255, 255), anchor='lm')
        y = 360
        if sl.get('eyebrow'):
            d.text((80, y), sl['eyebrow'].upper(), font=F('b', 32), fill=(170, 196, 255), anchor='lt'); y += 70
        for item in sl.get('headline', []):
            txt, blue = (item, False) if isinstance(item, str) else (item[0], item[1] == 'blue')
            for ln in wrap(txt, F('xb', 96), 920):
                d.text((80, y), ln, font=F('xb', 96), fill=(170, 196, 255) if blue else (255, 255, 255), anchor='lt'); y += 112
        if sl.get('sub'):
            y += 30
            for ln in wrap(sl['sub'], F('m', 42), 900): d.text((80, y), ln, font=F('m', 42), fill=(214, 226, 255), anchor='lt'); y += 58
        d.text((80, PH - 90), 'onelinker.ai', font=F('b', 40), fill=(255, 255, 255), anchor='lm')
        d.text((PW - 80, PH - 90), 'Real estate CRM · Qatar', font=F('m', 32), fill=(214, 226, 255), anchor='rm')
        return im.convert('RGB')
    im = Image.new('RGBA', (PW, PH), (255, 255, 255, 255))
    g = Image.new('RGBA', (PW // 4, PH // 4), (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
    gd.ellipse([-60, 180, 200, 420], fill=(29, 91, 240, 24)); gd.ellipse([170, -50, 330, 110], fill=(29, 91, 240, 16))
    im.alpha_composite(g.filter(ImageFilter.GaussianBlur(36)).resize((PW, PH), Image.BILINEAR)); d = ImageDraw.Draw(im)
    lg = logo.resize((60, 60), Image.LANCZOS); im.alpha_composite(lg, (72, 66))
    f = F('b', 40); d.text((146, 96), 'OneLinker', font=f, fill=NAVY, anchor='lm'); d.text((146 + f.getlength('OneLinker '), 96), 'CRM', font=f, fill=BLUE, anchor='lm')
    if total > 1: d.text((PW - 72, 96), f'{idx + 1}/{total}', font=F('sb', 32), fill=GREY, anchor='rm')
    y = 190
    if sl.get('eyebrow'):
        fe = F('b', 28); tw = fe.getlength(sl['eyebrow'].upper())
        d.rounded_rectangle([72, y, 72 + tw + 44, y + 54], 27, fill=(232, 240, 254)); d.text((94, y + 27), sl['eyebrow'].upper(), font=fe, fill=BLUE, anchor='lm'); y += 84
    hs = 84 if lay != 'cover' else 96
    for item in sl.get('headline', []):
        txt, blue = (item, False) if isinstance(item, str) else (item[0], item[1] == 'blue')
        for ln in wrap(txt, F('xb', hs), 936):
            d.text((72, y), ln, font=F('xb', hs), fill=BLUE if blue else NAVY, anchor='lt'); y += int(hs * 1.14)
    if sl.get('sub'):
        y += 18
        for ln in wrap(sl['sub'], F('m', 38), 920): d.text((72, y), ln, font=F('m', 38), fill=GREY, anchor='lt'); y += 52
    if sl.get('bullets'):
        y += 30
        for b in sl['bullets']:
            d.ellipse([72, y + 2, 120, y + 50], fill=BLUE); d.line([(85, y + 27), (94, y + 36), (109, y + 17)], fill=(255, 255, 255), width=6)
            ly = y
            for ln in wrap(b, F('sb', 40), 860): d.text((144, ly + 2), ln, font=F('sb', 40), fill=NAVY, anchor='lt'); ly += 52
            y = max(y + 74, ly + 22)
    if sl.get('screen'):
        img = site_img(sl['screen'])
        if sl.get('crop'):
            x0, y0, x1, y1 = sl['crop']; img = img.crop((int(x0 * img.width), int(y0 * img.height), int(x1 * img.width), int(y1 * img.height)))
        if np.array(img.getchannel('A')).min() > 250: img = rounded(img, max(12, img.width // 60))
        top = y + 40; avail = PH - 130 - top
        if avail > 200:
            sh = fit(img, 1000, avail); im.alpha_composite(sh, (PW // 2 - sh.width // 2, top + (avail - sh.height) // 2))
    d.rectangle([0, PH - 96, PW, PH], fill=(244, 247, 253))
    d.text((72, PH - 48), 'onelinker.ai', font=F('b', 34), fill=BLUE, anchor='lm')
    d.text((PW - 72, PH - 48), 'The real estate CRM built for Qatar', font=F('m', 30), fill=GREY, anchor='rm')
    return im.convert('RGB')

def build_post(path):
    spec = json.load(open(path)); rid = spec['id']; logo = logo_img()
    slides = spec['slides']; urls = []
    repo = os.environ.get('GITHUB_REPOSITORY', 'ONELINKERCRM/onelinker-shorts')
    for k, sl in enumerate(slides):
        fn = f'out/{rid}-{k + 1}.jpg'
        post_slide(sl, logo, k, len(slides)).save(fn, quality=92)
        urls.append(f'https://raw.githubusercontent.com/{repo}/main/{fn}')
    json.dump({'id': rid, 'kind': 'post', 'image_urls': urls}, open(f'out/{rid}.json', 'w'), indent=2)
    log('built post', rid, len(urls))

if __name__ == '__main__':
    ok = True
    for path in sorted(glob.glob('content/*.json')):
        rid = os.path.splitext(os.path.basename(path))[0]
        is_post = json.load(open(path)).get('kind') == 'post'
        if os.path.exists(f'out/{rid}.mp4') or (is_post and os.path.exists(f'out/{rid}-1.jpg')): continue
        try: build_post(path) if is_post else build(path)
        except Exception:
            ok = False; log('FAILED', rid); log(traceback.format_exc())
            json.dump({'id': rid, 'error': traceback.format_exc()[-1500:]}, open(f'out/{rid}.json', 'w'), indent=2)
    open('out/last-run.log', 'w').write('\n'.join(LOG))
    sys.exit(0 if ok else 1)
