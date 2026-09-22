#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
DATE8_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
OCR_PATTERNS = [
    re.compile(r"(?<!\d)(20\d{2})\s*[-./\\]\s*(0?[1-9]|1[0-2])\s*[-./\\]\s*(0?[1-9]|[12]\d|3[01])(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)"),
]
KEYWORDS = ("DATE", "EXAM", "STUDY", "TIME", "日期", "检查", "时间")


def valid_date8(s: str) -> str:
    try:
        return datetime.strptime(s, "%Y%m%d").strftime("%Y%m%d")
    except Exception:
        return ""


def date_from_parts(y, m, d) -> str:
    try:
        return datetime(int(y), int(m), int(d)).strftime("%Y%m%d")
    except Exception:
        return ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--tesseract-cmd", default="")
    p.add_argument("--ocr-lang", default="eng")
    return p.parse_args()


def name_date(video: Path, root: Path):
    candidates = []
    cur = video.parent
    while True:
        for m in DATE8_RE.findall(cur.name):
            d = valid_date8(m)
            if d:
                candidates.append(d)
        if candidates or cur == root:
            break
        try:
            cur.relative_to(root)
        except Exception:
            break
        cur = cur.parent
    if not candidates:
        candidates = [valid_date8(x) for x in DATE8_RE.findall(video.stem)]
        candidates = [x for x in candidates if x]
    u = sorted(set(candidates))
    if len(u) == 1:
        return u[0], "OK"
    if len(u) > 1:
        return "|".join(u), "AMBIGUOUS"
    return "", "NO_DATE"


def first_frame(video: Path):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None, -1, "VIDEO_OPEN_FAILED"
        for i in range(10):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return frame, i, "OK"
        return None, -1, "NO_DECODABLE_FRAME"
    finally:
        cap.release()


def preprocess(frame: np.ndarray):
    h = frame.shape[0]
    regions = [frame, frame[:max(1, int(h * .45)), :], frame[int(h * .65):, :]]
    out = []
    for img in regions:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(big, (3, 3), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out.append(bw)
    return out


def ocr_text(frame: np.ndarray, lang: str):
    if pytesseract is None:
        return "", "OCR_UNAVAILABLE"
    try:
        parts = [
            pytesseract.image_to_string(img, lang=lang, config="--psm 6")
            for img in preprocess(frame)
        ]
        return "\n".join(parts), "OK"
    except Exception as e:
        return "", f"OCR_FAILED:{type(e).__name__}"


def extract_dates(text: str):
    found = []
    for pat in OCR_PATTERNS:
        for m in pat.finditer(text):
            g = m.groups()
            d = date_from_parts(g[0], g[1], g[2])
            if d:
                found.append(d)
    return sorted(set(found))


def choose_ocr_date(text: str, dates: list[str]):
    if not dates:
        return "", "OCR_NO_DATE"
    if len(dates) == 1:
        return dates[0], "UNIQUE_OCR_DATE"
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    near = []
    for i, line in enumerate(lines):
        if any(k in line.upper() or k in line for k in KEYWORDS):
            block = " ".join(lines[max(0, i - 1):i + 2])
            near.extend(extract_dates(block))
    near = sorted(set(near))
    if len(near) == 1:
        return near[0], "KEYWORD_RESOLVED_DATE"
    return "", "MULTIPLE_DATES_REVIEW"


def compare(nd, ns, od, os):
    if ns == "AMBIGUOUS":
        return "REVIEW_NAME_DATE"
    if not nd:
        return "NO_NAME_DATE"
    if not od:
        return "REVIEW_OCR"
    return "MATCH" if nd == od else "MISMATCH"


def main():
    args = parse_args()
    root = Path(args.video_root).resolve()
    out = Path(args.output_root).resolve()
    frames = out / "first_frames"
    out.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)

    if pytesseract is not None and args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    videos = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    rows = []
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video}")
        nd, ns = name_date(video, root)
        frame, fi, fs = first_frame(video)
        saved = ""
        text = ""
        od = ""
        os = "NO_FRAME"
        candidates = []
        if frame is not None:
            safe = re.sub(r'[<>:"/\\|?*]+', "_", "__".join(video.relative_to(root).parts))
            fp = frames / (Path(safe).stem + ".jpg")
            if cv2.imwrite(str(fp), frame):
                saved = str(fp)
            text, raw = ocr_text(frame, args.ocr_lang)
            if raw == "OK":
                candidates = extract_dates(text)
                od, os = choose_ocr_date(text, candidates)
            else:
                os = raw
        rows.append({
            "video_relpath": str(video.relative_to(root)),
            "video_filename": video.name,
            "patient_folder": video.parent.name,
            "name_date": nd,
            "name_date_status": ns,
            "first_decodable_frame_index": fi,
            "frame_status": fs,
            "saved_first_frame": saved,
            "ocr_date": od,
            "all_ocr_date_candidates": "|".join(candidates),
            "ocr_status": os,
            "date_compare_status": compare(nd, ns, od, os),
            "ocr_text": text[:1500],
        })

    fields = list(rows[0].keys()) if rows else []
    audit = out / "video_exam_date_audit.csv"
    review = out / "date_mismatch_review.csv"
    for path, subset in [
        (audit, rows),
        (review, [r for r in rows if r["date_compare_status"] != "MATCH"]),
    ]:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    counts = Counter(r["date_compare_status"] for r in rows)
    (out / "SUMMARY.txt").write_text(
        "\n".join([f"total_videos: {len(rows)}"] + [f"{k}: {v}" for k, v in sorted(counts.items())]),
        encoding="utf-8",
    )
    print(counts)


if __name__ == "__main__":
    main()
