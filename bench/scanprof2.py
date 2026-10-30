"""A/B of detector paths and JPEG decode for BattleScan/server/cdn_app.py, interleaved per photo."""
import glob, io, os, sys, time, statistics as S
sys.path.insert(0, "/home/nez/Projects/Startup/Pokoin/BattleScan/server")
import numpy as np, onnxruntime as ort, torch
from PIL import Image
import cdn_app as A
A._load()
ONNX = "/home/nez/Projects/Startup/Pokoin/BattleScan/clones/TCG/backend/models/card_detector.onnx"
so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.inter_op_num_threads = 1
det = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])

def ort_detect(rgb):
    h, w = rgb.shape[:2]; r = 640 / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((640, 640, 3), 114, np.uint8)
    top, left = (640 - nh) // 2, (640 - nw) // 2
    canvas[top:top + nh, left:left + nw] = im
    x = (canvas.transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
    out = det.run(None, {"images": x})[0][0]
    out = out[out[:, 4] >= A.YOLO_CONF]
    boxes = []
    for x1, y1, x2, y2, c, _ in out:
        bx = [(x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r]
        bx = [max(0.0, min(v, lim)) for v, lim in zip(bx, (w, h, w, h))]
        if (bx[2] - bx[0]) * (bx[3] - bx[1]) >= A.MIN_AREA * w * h:
            boxes.append({"xyxy": bx, "conf": float(c)})
    boxes.sort(key=lambda b: -b["conf"])
    return boxes

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy
    return ix*iy/u if u else 0

def top1(rgb, box):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    hits, _ = A._embed_best(rgb[max(0,y1):y2, max(0,x1):x2], 1)
    return hits[0]["ct_id"] if hits else None

imgs = sorted(glob.glob("/home/nez/Projects/Startup/Pokoin/BattleScan/images/vinted/*.jpg"))[:24] + \
       sorted(glob.glob("/home/nez/Projects/Startup/Pokoin/BattleScan/images/*.png"))[:6]
blobs = [open(p, "rb").read() for p in imgs]
R = {k: [] for k in ["A_ultra_t6", "B_ultra_t4", "C_ort_t4", "dec_full", "dec_draft"]}
same_box = same_card = n = 0; draft_same = draft_n = 0
def ms(t0): return 1000 * (time.perf_counter() - t0)
for rep in range(3):
    for b in blobs:
        t0 = time.perf_counter(); im = Image.open(io.BytesIO(b)).convert("RGB")
        w, h = im.size; L = max(w, h)
        if L > 1600: im = im.resize((int(w*1600/L), int(h*1600/L)), Image.BILINEAR)
        R["dec_full"].append(ms(t0)); rgb = np.asarray(im)
        t0 = time.perf_counter(); im2 = Image.open(io.BytesIO(b))
        if im2.format == "JPEG": im2.draft("RGB", (1600, 1600))
        im2 = im2.convert("RGB"); w2, h2 = im2.size; L2 = max(w2, h2)
        if L2 > 1600: im2 = im2.resize((int(w2*1600/L2), int(h2*1600/L2)), Image.BILINEAR)
        R["dec_draft"].append(ms(t0)); rgb2 = np.asarray(im2)
        torch.set_num_threads(6)
        t0 = time.perf_counter(); a = A._detect(rgb); R["A_ultra_t6"].append(ms(t0))
        torch.set_num_threads(4)
        t0 = time.perf_counter(); A._detect(rgb); R["B_ultra_t4"].append(ms(t0))
        t0 = time.perf_counter(); c = ort_detect(rgb); R["C_ort_t4"].append(ms(t0))
        if rep == 0 and a:
            n += 1
            if c and iou(a[0]["xyxy"], c[0]["xyxy"]) > 0.9: same_box += 1
            ta = top1(rgb, a[0]["xyxy"]); tc = top1(rgb, c[0]["xyxy"]) if c else None
            same_card += ta == tc
            if rgb2.shape != rgb.shape or True:
                d2 = A._detect(rgb2)
                if d2:
                    draft_n += 1; draft_same += top1(rgb2, d2[0]["xyxy"]) == ta
def q(v, p): v = sorted(v); return v[min(len(v)-1, int(p*len(v)))]
for k, v in R.items(): print(f"{k:12s} n={len(v)} p50={q(v,.5):7.1f} p90={q(v,.9):7.1f} mean={S.mean(v):7.1f}")
print(f"ORT vs ultralytics: top box IoU>0.9 {same_box}/{n}, same top-1 card {same_card}/{n}; draft decode same card {draft_same}/{draft_n}")
