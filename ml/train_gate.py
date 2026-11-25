"""PCB gate: "is there a circuit board in this photo?"

Runs on the phone before the defect detector. Without it, a photo of a desk or
keyboard has no defects, so it would be reported as a passing board.

Design: reuse the trained detector's backbone (layers 0-9, it already knows
PCB texture) as a frozen feature extractor at 320 px, pool to a 512-d vector
(global average + max), and fit a logistic-regression head. The head is folded
into a torch Linear layer and the whole gate is exported to ExecuTorch.

Data
  positive: PKU-Market-PCB train boards, plus "in the wild" composites: a board
            pasted, scaled and rotated onto a non-PCB background (phone photos
            show the board on a desk, not a clean scan)
  negative: macOS wallpapers + frames recorded from the phone camera
            (ml/data/notpcb/phone, desks, hands, keyboards, cables)
  eval:     PKU validation boards (unseen) and held-out negatives

Usage:  ml/.venv-et/bin/python ml/train_gate.py
Output: ml/models/pcb_gate_v5.pte, ml/results/pcb_gate_v5.json (v4: real yellow / green / red boards; v5: + real room video negatives)
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

ROOT = Path(__file__).resolve().parent
PKU = ROOT / "data" / "pku"
NEG = ROOT / "data" / "notpcb"
SIZE = 320
random.seed(0); np.random.seed(0); torch.manual_seed(0)


class Gate(torch.nn.Module):
    """Backbone (YOLO layers 0-9) -> avg+max pool -> linear -> probability of PCB."""

    def __init__(self, layers, head=None):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.head = head if head is not None else torch.nn.Linear(512, 1)

    def fmap(self, x):
        for m in self.layers:
            x = m(x)
        return x                                                   # (N, 256, 10, 10) at 320 px

    def features(self, x):
        x = self.fmap(x)
        return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)

    def forward(self, x):
        return torch.sigmoid(self.head(self.features(x)))


class GateMap(torch.nn.Module):
    """Gate + board map: one backbone pass gives P(PCB) and a 10x10 'board here' map.
    The app tiles only the board area, so a small board in a big photo costs a few tiles."""

    def __init__(self, gate, cell):
        super().__init__()
        self.gate, self.cell = gate, cell                          # cell: 1x1 conv 256 -> 1

    def forward(self, x):
        f = self.gate.fmap(x)
        p = torch.sigmoid(self.gate.head(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        return p, torch.sigmoid(self.cell(f)).flatten(1)          # (1,1), (1,100)


def letterbox(img):
    """Fit into SIZE x SIZE on black, same as the Kotlin side."""
    h, w = img.shape[:2]
    s = SIZE / max(h, w)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((SIZE, SIZE, 3), np.uint8)
    out[: r.shape[0], : r.shape[1]] = r
    return out


def tensor(img_bgr):
    rgb = letterbox(img_bgr)[:, :, ::-1]
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div(255)


def jitter(img):
    """Phone-camera look: brightness, contrast, blur, noise, white balance."""
    img = img.astype(np.float32)
    img = img * random.uniform(0.6, 1.3) + random.uniform(-30, 30)
    img *= np.array([random.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    if random.random() < 0.5:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    img += np.random.normal(0, random.uniform(0, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def recolor(img):
    """Change the solder-mask colour (green -> blue / red / black / purple / white...) so the
    gate learns PCB structure (traces, pads, chips), not just the colour green."""
    r = random.random()
    if r < 0.35:
        return img
    if r < 0.5:  # black or dark board
        g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return np.clip(g.astype(np.float32) * random.uniform(0.35, 0.7), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + random.randint(10, 170)) % 180      # any hue
    if random.random() < 0.15:
        hsv[..., 1] = hsv[..., 1] // 4                                 # pale / white board
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def composite(board, bg, with_mask=False):
    """Paste a PCB onto a background: scaled 35-85 %, rotated, shifted."""
    H, W = 720, 960
    bg = cv2.resize(bg, (W, H))
    s = random.uniform(0.35, 0.85) * min(W / board.shape[1], H / board.shape[0])
    b = cv2.resize(board, (int(board.shape[1] * s), int(board.shape[0] * s)))
    M = cv2.getRotationMatrix2D((b.shape[1] / 2, b.shape[0] / 2), random.uniform(-25, 25), 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(b.shape[0] * sin + b.shape[1] * cos), int(b.shape[0] * cos + b.shape[1] * sin)
    M[0, 2] += nw / 2 - b.shape[1] / 2; M[1, 2] += nh / 2 - b.shape[0] / 2
    rot = cv2.warpAffine(b, M, (nw, nh)); mask = cv2.warpAffine(np.full(b.shape[:2], 255, np.uint8), M, (nw, nh))
    nw, nh = min(nw, W), min(nh, H); rot, mask = rot[:nh, :nw], mask[:nh, :nw]
    x, y = random.randint(0, W - nw), random.randint(0, H - nh)
    roi = bg[y:y + nh, x:x + nw]
    roi[mask > 0] = rot[mask > 0]
    if with_mask:
        full = np.zeros((H, W), np.uint8); full[y:y + nh, x:x + nw] = mask
        return bg, full
    return bg


def crops(img, n):
    """Random crops covering 40-100 % of the image."""
    out = []
    h, w = img.shape[:2]
    for _ in range(n):
        f = random.uniform(0.4, 1.0)
        cw, ch = int(w * f), int(h * f)
        x, y = random.randint(0, w - cw), random.randint(0, h - ch)
        c = img[y:y + ch, x:x + cw]
        out.append(cv2.flip(c, 1) if random.random() < 0.5 else c)
    return out


def real_board_photos():
    """v4: real photos of real boards (not PKU edits). Train = designs used for detector training;
    test = the held-out designs (ml/data/real_test) + the green board video crops (never trained on)."""
    import json as _j
    rep = _j.loads((ROOT / "results" / "real_boards_dataset.json").read_text())
    test_pcbd, test_ddet = set(rep["pcbdefect"]["test_designs"]), set(rep["defectdet"]["test_designs"])
    pcbd = sorted((ROOT / "data/ext/pcbdefect/PCB-Defect-Reannotated/train/images").glob("*.jpg"))
    dd = _j.loads((ROOT / "data/ext/defectdet/DefectDet/annotation/defect_annotations.json").read_text())
    ddet = [(ROOT / "data/ext/defectdet/DefectDet/images" / i["file_name"], i["extra"]["name"].split("_")[0]) for i in dd["images"]]
    train = [f for f in pcbd if "-".join(f.stem.split("-")[:2]) not in test_pcbd] + [f for f, g in ddet if g not in test_ddet]
    train += sorted((ROOT / "data/real_video").glob("*/v_red_*.jpg"))
    test = {"pcbdefect_yellow": sorted((ROOT / "data/real_test/pcbdefect/images").glob("*.jpg")),
            "defectdet_green": sorted((ROOT / "data/real_test/defectdet/images").glob("*.jpg")),
            "green_board_video": sorted((ROOT / "data/real_video").glob("*/v_green_*.jpg"))}
    return train, test


def small(path, side=1000):
    im = cv2.imread(str(path))
    s = side / max(im.shape[:2])
    return cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else im


def main():
    yolo = YOLO(str(ROOT / "runs/pcb_yolo11n/weights/best.pt")).model.float().eval().fuse()
    gate = Gate(list(yolo.model[:10])).eval()

    names = lambda f: [Path(l.strip()).stem for l in (PKU / "ImageSets" / f).read_text().splitlines() if l.strip()]
    load = lambda stem: cv2.imread(str(next(PKU.glob(f"images/**/{stem}.*"))))
    train_boards, val_boards = names("train_board.txt"), names("val_board.txt")

    # Consecutive phone frames are near-duplicates, so phone data is split by time
    # (first 80 % train, last 20 % test) to keep the test honest.
    def time_split(files):
        files = sorted(files); c = int(len(files) * 0.8); return files[:c], files[c:]
    walls = sorted((NEG / "wallpapers").glob("*.jpg")); random.shuffle(walls)
    phone_tr, phone_te = time_split((NEG / "phone").glob("*.jpg"))
    real_tr, real_te = time_split((ROOT / "data" / "pcb_real" / "phone").glob("*.jpg"))
    c = int(len(walls) * 0.8)
    # v5: real room video (desk, doors, shelves, cables, cloth), split by time like the other phone clips.
    room_tr, room_te = time_split((NEG / "room_video").glob("*.jpg"))
    neg_train, neg_test = walls[:c] + phone_tr + room_tr, walls[c:] + phone_te + room_te
    neg_files = walls + phone_tr + phone_te + room_tr + room_te
    bg_train = [cv2.imread(str(f)) for f in neg_train]
    bg_test = [cv2.imread(str(f)) for f in neg_test]
    n_phone = len(list((NEG / "phone").glob("*.jpg")))
    print(f"boards: {len(train_boards)} train / {len(val_boards)} val · real board frames {len(real_tr)}/{len(real_te)} · negatives: {len(neg_files)} ({n_phone} phone frames)")

    def build(boards, bgs, n_board_samples, n_neg_crops, real=()):
        X, y = [], []
        for f in real:                                                               # real phone photos of real boards
            im = cv2.imread(str(f))
            X.append(tensor(im)); y.append(1)
            for cimg in crops(im, 2):
                X.append(tensor(jitter(cimg))); y.append(1)
        for stem in random.sample(boards, min(n_board_samples, len(boards))):
            b = recolor(load(stem))
            X.append(tensor(jitter(b))); y.append(1)                                   # clean scan
            X.append(tensor(jitter(composite(b, random.choice(bgs))))); y.append(1)   # board on a desk
            X.append(tensor(jitter(random.choice(crops(b, 1))))); y.append(1)         # close-up
        for bg in bgs:
            for c in crops(bg, n_neg_crops):
                X.append(tensor(jitter(c))); y.append(0)
        return X, np.array(y)

    def embed(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                out.append(gate.features(torch.stack(X[i:i + 32])).numpy())
        return np.concatenate(out)

    Xtr, ytr = build(train_boards, bg_train, 541, 8, real_tr)
    real_train, real_test = real_board_photos()
    n_add = 0
    for f in random.sample(real_train, min(260, len(real_train))):     # real boards: whole photo, on a desk, close-up
        b = small(f)
        for im in (b, composite(b.copy(), random.choice(bg_train)), random.choice(crops(b, 1))):
            Xtr.append(tensor(jitter(im))); n_add += 1
    ytr = np.concatenate([ytr, np.ones(n_add, int)])
    print(f"v4: + {n_add} real-board training images from {len(real_train)} photos (yellow bare copper, green mask, red board)")
    Xte, yte = build(val_boards, bg_test, 152, 8, real_te)
    # Separate check on real phone frames only (held-out end of each clip).
    room_photo = sorted((NEG / "room_photo_test").glob("*.jpg"))
    Xr = [tensor(cv2.imread(str(f))) for f in real_te] + [tensor(cv2.imread(str(f))) for f in phone_te + room_te + room_photo]
    yr = np.array([1] * len(real_te) + [0] * (len(phone_te) + len(room_te) + len(room_photo)))
    print(f"train {len(ytr)} ({ytr.sum()} PCB) · test {len(yte)} ({yte.sum()} PCB)")
    Ftr, Fte = embed(Xtr), embed(Xte)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    clf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit((Ftr - mu) / sd, ytr)

    # Fold standardisation into the Linear layer: w' = w / sd, b' = b - sum(w * mu / sd)
    w = clf.coef_[0] / sd
    b = clf.intercept_[0] - float((clf.coef_[0] * mu / sd).sum())
    head = torch.nn.Linear(512, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(w, dtype=torch.float32).unsqueeze(0)); head.bias.fill_(b)
    gate.head = head

    with torch.no_grad():
        p_te = torch.cat([gate(torch.stack(Xte[i:i + 32])) for i in range(0, len(Xte), 32)]).squeeze(1).numpy()
    pred = (p_te >= 0.5).astype(int)
    with torch.no_grad():
        p_r = gate(torch.stack(Xr)).squeeze(1).numpy()
    real_check = dict(real_board_frames=int(yr.sum()), real_board_accepted=int(((p_r >= 0.5) & (yr == 1)).sum()),
                      real_background_frames=int((yr == 0).sum()), real_background_rejected=int(((p_r < 0.5) & (yr == 0)).sum()))
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    res = dict(test_images=int(len(yte)), pcb=int(yte.sum()), not_pcb=int(len(yte) - yte.sum()),
               accuracy=round(float((pred == yte).mean()), 4),
               pcb_recall=round(tp / (tp + fn), 4), not_pcb_recall=round(tn / (tn + fp), 4),
               pcb_missed_as_not_pcb=int(fn), non_pcb_accepted_as_pcb=int(fp), threshold=0.5,
               negatives_used=len(neg_files), phone_frames=n_phone,
               real_phone_check=real_check, real_board_train_frames=len(real_tr),
               note="Test PCBs are unseen boards 10-12 (scans, desk composites, close-ups) and the last 20 % of the real-board clip; negatives held out.")
    with torch.no_grad():
        rp = [float(gate(tensor(small(f)).unsqueeze(0))) for f in room_te + room_photo]
        res["v5_room_test"] = dict(room_video_last_25pct=len(room_te), room_photo=len(room_photo),
                                   rejected=int(sum(p < 0.5 for p in rp)), below_pass_min_0_7=int(sum(p < 0.7 for p in rp)),
                                   max_prob=round(max(rp), 3), room_photo_prob=round(rp[-1], 3) if room_photo else None)
        for name, files in real_test.items():
            ps = [float(gate(tensor(small(f)).unsqueeze(0))) for f in files]
            res.setdefault("v4_held_out_real_boards", {})[name] = dict(images=len(ps), accepted=int(sum(p >= 0.5 for p in ps)),
                                                                       blocked_below_0_05=int(sum(p < 0.05 for p in ps)),
                                                                       min_prob=round(min(ps), 3))
    print(json.dumps(res, indent=2))

    # ---- Board map: where is the board? Labels come free from the composites (we pasted the board).
    def cells(mask):
        """Fraction of each 32x32 cell (10x10 grid at 320 px, letterboxed like the image) covered by board."""
        return cv2.resize(letterbox(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))[:, :, 0].astype(np.float32) / 255,
                          (10, 10), interpolation=cv2.INTER_AREA).ravel()

    MASKS = ROOT / "data" / "pcb_real" / "masks"                      # from ml/real_board_masks.py

    def real_board(f):
        """A real phone frame of a populated board + its board mask (None if no mask)."""
        mf = MASKS / (f.stem + ".png")
        return (cv2.imread(str(f)), cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)) if mf.exists() else None

    def map_set(boards, bgs, n, real=()):
        X, Y = [], []
        for f in real:                                                   # real populated board on a cloth
            rb = real_board(f)
            if rb is None: continue
            img, m = rb
            X.append(tensor(jitter(img))); Y.append(cells(m))
            ys, xs = np.nonzero(m)                                       # same board cut out, pasted elsewhere
            cut = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            for _ in range(2):
                ci, cm_ = composite(cut, random.choice(bgs), with_mask=True)
                X.append(tensor(jitter(ci))); Y.append(cells(cm_))
        for stem in random.sample(boards, min(n, len(boards))):
            b = recolor(load(stem))
            img, m = composite(b, random.choice(bgs), with_mask=True)
            X.append(tensor(jitter(img))); Y.append(cells(m))
            X.append(tensor(jitter(b))); Y.append(cells(np.full(b.shape[:2], 255, np.uint8)))   # board fills photo
        for bg in random.sample(bgs, min(n // 2, len(bgs))):
            c = crops(bg, 1)[0]
            X.append(tensor(jitter(c))); Y.append(np.zeros(100, np.float32))
        return X, np.array(Y)

    def cell_feats(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                f = gate.fmap(torch.stack(X[i:i + 32]))
                out.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).numpy())
        return np.concatenate(out)

    Mtr, Ytr = map_set(train_boards, bg_train, 400, real_tr)
    Mte, Yte = map_set(val_boards, bg_test, 152)
    Rte, RYte = map_set([], bg_test, 0, real_te)                           # held-out real frames (+ their pastes)
    Ctr, ytr_c = cell_feats(Mtr), Ytr.ravel()
    keep = (ytr_c > 0.6) | (ytr_c < 0.15)                           # skip half-covered edge cells
    cm, cs = Ctr[keep].mean(0), Ctr[keep].std(0) + 1e-6
    cclf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit((Ctr[keep] - cm) / cs, (ytr_c[keep] > 0.5).astype(int))
    cell = torch.nn.Conv2d(256, 1, 1)
    with torch.no_grad():
        cell.weight.copy_(torch.tensor(cclf.coef_[0] / cs, dtype=torch.float32).view(1, 256, 1, 1))
        cell.bias.fill_(float(cclf.intercept_[0] - (cclf.coef_[0] * cm / cs).sum()))
    gm = GateMap(gate, cell).eval()

    def box(cellmap):
        on = np.argwhere(cellmap.reshape(10, 10) >= 0.5)
        if len(on) == 0: return None
        (r0, c0), (r1, c1) = on.min(0), on.max(0)
        return c0 * 32, r0 * 32, (c1 + 1) * 32, (r1 + 1) * 32

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u else 0
    with torch.no_grad():
        maps = torch.cat([gm(torch.stack(Mte[i:i + 32]))[1] for i in range(0, len(Mte), 32)]).numpy()
    pred = maps.ravel() >= 0.5; truth = Yte.ravel() > 0.5; sure = (Yte.ravel() > 0.6) | (Yte.ravel() < 0.15)
    ious, covered = [], []
    for mp, yt in zip(maps, Yte):
        if yt.max() > 0.5:
            tb, pb = box(yt), box(mp)
            ious.append(iou(tb, pb) if pb else 0.0)
            covered.append(bool(pb) and pb[0] <= tb[0] and pb[1] <= tb[1] and pb[2] >= tb[2] and pb[3] >= tb[3])
    res["board_map"] = dict(test_images=len(Mte), cell_accuracy=round(float((pred == truth)[sure].mean()), 4),
                            median_box_iou=round(float(np.median(ious)), 3),
                            board_fully_inside_box=round(float(np.mean(covered)), 3),
                            note="10x10 map at 320 px; the app pads the box by one cell before tiling.")
    def padded_ok(mp, yt):
        """Board fully inside the box after the app's one-cell padding (or whole-photo fallback)."""
        on = np.argwhere(mp.reshape(10, 10) >= 0.5)
        if len(on) == 0: return True
        (r0, c0), (r1, c1) = on.min(0) - 1, on.max(0) + 1
        if (r1 - r0 + 1) * (c1 - c0 + 1) > 80: return True
        t = np.argwhere(yt.reshape(10, 10) > 0.3)
        return bool(len(t) == 0 or (t.min(0) >= [r0, c0]).all() and (t.max(0) <= [r1, c1]).all())
    with torch.no_grad():
        rmaps = gm(torch.stack(Rte))[1].numpy()
    res["board_map"]["synthetic_board_inside_padded_box"] = round(float(np.mean([padded_ok(a, b) for a, b in zip(maps, Yte)])), 3)
    res["board_map"]["real_board_inside_padded_box"] = f"{sum(padded_ok(a, b) for a, b in zip(rmaps, RYte))}/{len(RYte)}"
    print("board map:", res["board_map"])

    x = torch.zeros(1, 3, SIZE, SIZE)
    prog = to_edge_transform_and_lower(torch.export.export(gm, (x,)), partitioner=[XnnpackPartitioner()]).to_executorch()
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "pcb_gate_v5.pte").write_bytes(prog.buffer)
    res["pte_mb"] = round(len(prog.buffer) / 1e6, 2)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "pcb_gate_v5.json").write_text(json.dumps(res, indent=2))
    print("saved models/pcb_gate_v5.pte", res["pte_mb"], "MB")


if __name__ == "__main__":
    main()
