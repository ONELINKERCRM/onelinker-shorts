"""OneLinker CRM reel builder (runs in GitHub Actions).

For every content/<id>.json without out/<id>.mp4:
  1. voiceover per scene via ElevenLabs (secret ELEVENLABS_API_KEY)
  2. real CRM screens from www.onelinker.ai
  3. 1080x1920 motion graphics in the OneLinker brand theme, timed to the voice
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
    local = os.path.join(CACHE, path.replace('/', '_'))
    if not os.path.exists(local):
        r = requests.get(f'{SITE}/{path.lstrip("/")}', timeout=60); r.raise_for_status()
        open(local, 'wb').write(r.content)
    return Image.open(local).convert('RGBA')

def logo_img():
    for p in ['icon.png', 'logo-128w.png']:
        try:
            im = site_img(p)
            a = np.array(im); white = a[..., :3].min(axis=2) > 240; a[white, 3] = 0
            return Image.fromarray(a)
        except Exception as e:
            log('logo fail', p, e)
    return Image.new('RGBA', (256, 256), BLUE + (255,))

# ---------- voice ----------
def tts(text, voice, out_wav):
    key = os.environ.get('ELEVENLABS_API_KEY')
    if not key: raise RuntimeError('ELEVENLABS_API_KEY secret is missing')
    r = requests.post(f'https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128',
                      headers={'xi-api-key': key, 'Content-Type': 'application/json'},
                      json={'text': text, 'model_id': 'eleven_multilingual_v2',
                            'voice_settings': {'stability': 0.45, 'similarity_boost': 0.8, 'style': 0.35, 'use_speaker_boost': True}},
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
    def __init__(self, spec):
        self.spec = spec; self.logo = logo_img()
        self.glow_cache = {}
        self.cards = {}
        for i, sc in enumerate(spec['scenes']):
            if sc.get('screen'):
                img = site_img(sc['screen'])
                # flatten transparent mockups onto white card if they have no alpha use
                a = np.array(img.getchannel('A'))
                if a.min() > 250:
                    card = rounded(fit(img, 900, 860), 28)
                    self.cards[i] = ('card', card)
                else:
                    self.cards[i] = ('free', fit(img, 960, 900))

    def base(self, t):
        im = Image.new('RGBA', (W, H), BG + (255,))
        step = int(t * 4)
        if step not in self.glow_cache:
            glow = Image.new('RGBA', (W // 4, H // 4), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
            cx = W // 8 + int(math.sin(step / 4 * 0.6) * 30)
            gd.ellipse([cx - 160, -125, cx + 160, 175], fill=BLUE + (36,))
            self.glow_cache = {step: glow.filter(ImageFilter.GaussianBlur(30)).resize((W, H), Image.BILINEAR)}
        im.alpha_composite(self.glow_cache[step])
        d = ImageDraw.Draw(im)
        lg = self.logo.resize((46, 46), Image.LANCZOS)
        d.rounded_rectangle([W // 2 - 200, 96, W // 2 + 200, 160], 32, fill=(255, 255, 255, 255), outline=(220, 228, 245))
        im.alpha_composite(lg, (W // 2 - 182, 105))
        d.text((W // 2 + 22, 128), 'OneLinker CRM  ·  Qatar', font=F('sb', 28), fill=NAVY, anchor='mm')
        return im

    def headline(self, d, lines, y, s, size):
        font = F('xb', size); yy0 = y
        for i, item in enumerate(lines):
            txt, col = (item, NAVY) if isinstance(item, str) else (item[0], BLUE if item[1] == 'blue' else NAVY)
            for j, ln in enumerate(wrap(txt, font, 980)):
                p = ease((s - (i + j) * 0.12) / 0.45)
                if p > 0:
                    d.text((W // 2, yy0 + int((1 - p) * 40)), ln, font=font, fill=col + (int(255 * p),), anchor='mt')
                yy0 += int(size * 1.12)
        return yy0

    def sub(self, d, text, y, s, size=40, t0=0.5):
        font = F('m', size); p = ease((s - t0) / 0.5)
        for ln in wrap(text, font, 940):
            if p > 0: d.text((W // 2, y + int((1 - p) * 20)), ln, font=font, fill=GREY + (int(255 * p),), anchor='mt')
            y += int(size * 1.3)
        return y

    def place(self, im, i, s, top, dur):
        kind, card = self.cards[i]
        if s <= 0.5: return
        p = back((s - 0.5) / 0.6); a = min(1, (s - 0.5) / 0.3)
        z = 1 + 0.04 * ease((s - 0.5) / max(dur, 1))
        c = card.resize((int(card.width * z), int(card.height * z)), Image.BILINEAR) if z > 1.001 else card
        y = top + int((1 - p) * 260)
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        if kind == 'card':
            pad = 34; pw, ph = c.width + pad * 2, c.height + pad * 2
            panel = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
            ImageDraw.Draw(panel).rounded_rectangle([0, 0, pw - 1, ph - 1], 44, fill=LIGHT + (255,))
            layer.alpha_composite(panel, (W // 2 - pw // 2, y - pad))
            sh, off = shadow(card.size, 28)
            layer.alpha_composite(sh, (W // 2 - card.width // 2 - off, y - off))
        layer.alpha_composite(c, (W // 2 - c.width // 2, y))
        if a < 1: layer.putalpha(layer.getchannel('A').point(lambda v: int(v * a)))
        im.alpha_composite(layer)

    def cta(self, sc, s):
        arr = np.zeros((H, W, 4), np.uint8)
        yy = np.linspace(0, 1, H)[:, None]; xx = np.linspace(0, 1, W)[None, :]
        k = np.clip(0.55 * yy + 0.45 * xx, 0, 1)
        for c, (a0, a1) in enumerate([(29, 13), (91, 31), (240, 92)]): arr[..., c] = (a0 + (a1 - a0) * k).astype(np.uint8)
        arr[..., 3] = 255
        im = Image.fromarray(arr, 'RGBA'); d = ImageDraw.Draw(im)
        p = back(s / 0.6); disc = 360
        dd = Image.new('RGBA', (disc, disc), (0, 0, 0, 0)); ImageDraw.Draw(dd).ellipse([0, 0, disc - 1, disc - 1], fill=(255, 255, 255, 255))
        lg = fit(self.logo, 250, 250); dd.alpha_composite(lg, ((disc - lg.width) // 2, (disc - lg.height) // 2))
        sz = max(1, int(disc * p)); dd = dd.resize((sz, sz), Image.LANCZOS)
        im.alpha_composite(dd, (W // 2 - sz // 2, 420 + (disc - sz) // 2))
        def txt(y, t_, f, col, t0):
            q = ease((s - t0) / 0.45)
            if q > 0: d.text((W // 2, y + int((1 - q) * 30)), t_, font=f, fill=col + (int(255 * q),), anchor='mm')
        txt(900, sc.get('title', 'OneLinker CRM'), F('xb', 110), (255, 255, 255), 0.4)
        y = 1010
        for ln in sc.get('lines', ['The real estate CRM built for Qatar']):
            txt(y, ln, F('m', 44), (214, 226, 255), 0.6); y += 66
        q = back((s - 1.1) / 0.5)
        if s > 1.1:
            bw, bh = int(640 * q), int(130 * q)
            d.rounded_rectangle([W // 2 - bw // 2, 1300 - bh // 2, W // 2 + bw // 2, 1300 + bh // 2], 65, fill=(255, 255, 255))
            if q > 0.7: d.text((W // 2, 1300), sc.get('button', 'Start free  →'), font=F('xb', 56), fill=BLUE, anchor='mm')
        txt(1460, 'www.onelinker.ai', F('b', 52), (255, 255, 255), 1.4)
        return im

    def frame(self, t, timeline):
        for i, (st, du) in enumerate(timeline):
            if st <= t < st + du or i == len(timeline) - 1: break
        sc = self.spec['scenes'][i]; s = t - st
        if sc['type'] == 'cta':
            im = self.cta(sc, s)
        else:
            im = self.base(t); d = ImageDraw.Draw(im)
            if sc['type'] == 'hook':
                y = self.headline(d, sc['headline'], 470, s, 104)
                y = self.sub(d, sc.get('sub', ''), y + 40, s, 42, 0.9)
                p = back((s - 1.2) / 0.6)
                if s > 1.2:
                    lg = fit(self.logo, int(300 * p) + 1, int(300 * p) + 1)
                    im.alpha_composite(lg, (W // 2 - lg.width // 2, 1200 + (300 - lg.height) // 2))
                if sc.get('footer'):
                    d.text((W // 2, 1600), sc['footer'], font=F('sb', 34), fill=NAVY + (int(255 * ease((s - 1.8) / 0.5)),), anchor='mm')
            else:
                y = self.headline(d, sc['headline'], 280, s, 96)
                y = self.sub(d, sc.get('sub', ''), y + 24, s, 38)
                if i in self.cards:
                    self.place(im, i, s, max(y + 70, 700), timeline[i][1])
            d.text((W // 2, H - 110), 'onelinker.ai', font=F('sb', 34), fill=BLUE, anchor='mm')
        for st, _ in timeline[1:]:
            if 0 <= t - st < 0.18:
                im.alpha_composite(Image.new('RGBA', (W, H), (255, 255, 255, int(200 * (1 - (t - st) / 0.18)))))
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
    voice = spec.get('voice_id', 'dXtC3XhB9GtPusIpNtQx'); sr = 44100
    clips, timeline, t = [], [], 0.0
    for i, sc in enumerate(spec['scenes']):
        v = tts(sc['voice'], voice, f'{CACHE}/{rid}_{i}.wav') if sc.get('voice') else np.zeros(int(sr * 1.5))
        lead = 0.25; tail = 0.45 if i < len(spec['scenes']) - 1 else 1.6
        du = max(lead + len(v) / sr + tail, sc.get('min', 2.5))
        clips.append((t + lead, v)); timeline.append((t, du)); t += du
    dur = min(t, 59.0)
    n = int(dur * sr); vo = np.zeros(n)
    for st, v in clips:
        i = int(st * sr); j = min(n, i + len(v)); vo[i:j] += v[:j - i]
    env = np.convolve(np.abs(vo), np.ones(4410) / 4410, 'same')
    duck = 1 - 0.75 * np.clip(env * 12, 0, 1)
    mix = 0.95 * vo / (np.abs(vo).max() + 1e-9) + 0.16 * music(dur)[:n] * duck
    mix *= np.minimum(1, (dur - np.arange(n) / sr) / 1.0)
    mix = np.clip(mix, -1, 1)
    wav = f'{CACHE}/{rid}_mix.wav'
    with wave.open(wav, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes((mix * 32767).astype(np.int16).tobytes())
    reel = Reel(spec)
    # preview sheet
    shots = [reel.frame(st + min(du * 0.8, du - 0.1), timeline).resize((270, 480)) for st, du in timeline]
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

if __name__ == '__main__':
    ok = True
    for path in sorted(glob.glob('content/*.json')):
        rid = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(f'out/{rid}.mp4'): continue
        try: build(path)
        except Exception:
            ok = False; log('FAILED', rid); log(traceback.format_exc())
            json.dump({'id': rid, 'error': traceback.format_exc()[-1500:]}, open(f'out/{rid}.json', 'w'), indent=2)
    open('out/last-run.log', 'w').write('\n'.join(LOG))
    sys.exit(0 if ok else 1)
