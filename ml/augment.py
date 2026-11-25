"""Phone-camera look for training tiles (OpenCV only, no extra packages).

PhoneLook works on the final 640x640 training image (after mosaic / affine / HSV) and changes only
pixels, never geometry, so boxes stay valid. Each effect fires with its own probability:
  defocus blur, motion blur, sensor noise, JPEG compression, low resolution (down + up), glare spot,
  shadow / uneven light, white-balance shift, grayscale.
Tilt and rotation come from Ultralytics' own affine settings (degrees / shear / perspective).

install() patches YOLODataset.build_transforms so PhoneLook is inserted before the final Format step
for training datasets only, including when Ultralytics rebuilds transforms at close_mosaic.
"""
import random

import cv2
import numpy as np


class PhoneLook:
    def __init__(self, p=0.8, p_hue=0.3):
        # p_hue is applied inside the p gate, so ~p * p_hue = 24 % of tiles get a new board colour.
        # Rotating the hue of EVERY tile (hsv_h=0.5) dropped PKU mAP50 to 0.29 after one epoch.
        self.p = p
        self.p_hue = p_hue

    def __call__(self, labels):
        if random.random() > self.p:
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        r = random.random
        if r() < self.p_hue:                              # other board colours (red / blue / black masks)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + random.randint(20, 160)) % 180).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if r() < 0.30:                                    # defocus
            k = random.choice((3, 5))
            img = cv2.GaussianBlur(img, (k, k), random.uniform(0.6, 1.6))
        if r() < 0.20:                                    # handheld motion blur
            k = random.randint(5, 11)
            kern = np.zeros((k, k), np.float32)
            kern[k // 2, :] = 1.0 / k
            m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1.0)
            img = cv2.filter2D(img, -1, cv2.warpAffine(kern, m, (k, k)))
        if r() < 0.30:                                    # low resolution phone crop
            s = random.uniform(0.45, 0.8)
            small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=random.choice((cv2.INTER_LINEAR, cv2.INTER_CUBIC)))
        f = img.astype(np.float32)
        if r() < 0.30:                                    # white balance
            f *= np.array([random.uniform(0.85, 1.15) for _ in range(3)], np.float32)
        if r() < 0.25:                                    # shadow / uneven light: linear gradient
            a = random.uniform(0, 2 * np.pi)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            g = (np.cos(a) * (xx / w - 0.5) + np.sin(a) * (yy / h - 0.5))
            f *= (1.0 + random.uniform(0.2, 0.5) * g)[..., None]
        if r() < 0.15:                                    # glare spot on copper / mask
            cx, cy = random.uniform(0, w), random.uniform(0, h)
            rad = random.uniform(0.08, 0.3) * w
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2))
            f += (random.uniform(60, 160) * spot)[..., None]
        if r() < 0.35:                                    # sensor noise
            f += np.random.normal(0, random.uniform(2, 9), f.shape).astype(np.float32)
        img = np.clip(f, 0, 255).astype(np.uint8)
        if r() < 0.08:                                    # grayscale
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        if r() < 0.40:                                    # JPEG
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 90)])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        labels["img"] = np.ascontiguousarray(img)
        return labels


def install(p=0.8):
    from ultralytics.data.dataset import YOLODataset

    if getattr(YOLODataset, "_phonelook", False):
        return
    original = YOLODataset.build_transforms

    def build_transforms(self, hyp):
        t = original(self, hyp)
        if self.augment:
            t.transforms.insert(len(t.transforms) - 1, PhoneLook(p))
        return t

    YOLODataset.build_transforms = build_transforms
    YOLODataset._phonelook = True
