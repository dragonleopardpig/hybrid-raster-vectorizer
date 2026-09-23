"""Offline OCR: Tesseract for prose, PP-FormulaNet for mathematics."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .components import Component


@dataclass
class Reading:
    text: str
    engine: str
    confidence: float


def _require(command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise SystemExit(f"Required command not found: {command}")
    return executable


def isolate(
    gray: np.ndarray, components: list[Component], *, pad: int = 12, scale: int = 3
) -> np.ndarray:
    """A white-background crop holding only this block's ink.

    Cropping the rectangle alone would drag in whatever else happens to cross
    it, which for a plot is usually an axis.
    """
    x = max(0, min(c.x for c in components) - pad)
    y = max(0, min(c.y for c in components) - pad)
    right = min(gray.shape[1], max(c.right for c in components) + pad)
    bottom = min(gray.shape[0], max(c.bottom for c in components) + pad)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    for component in components:
        region = mask[component.y : component.bottom, component.x : component.right]
        np.maximum(region, component.mask, out=region)

    window = np.where(mask[y:bottom, x:right] > 0, gray[y:bottom, x:right], np.uint8(255))
    if scale > 1:
        window = cv2.resize(window, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(window, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)


def read_tesseract(image: np.ndarray, *, psm: int = 7, language: str = "eng") -> Reading:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "crop.png"
        cv2.imwrite(str(path), image)
        completed = subprocess.run(
            [_require("tesseract"), str(path), "stdout", "--psm", str(psm), "-l", language, "tsv"],
            check=False,
            capture_output=True,
            text=True,
        )

    words: list[str] = []
    confidences: list[float] = []
    for line in completed.stdout.splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) < 12:
            continue
        text = fields[11].strip()
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        if text and confidence >= 0:
            words.append(text)
            confidences.append(confidence)

    if not words:
        return Reading(text="", engine="tesseract", confidence=0.0)
    return Reading(
        text=" ".join(words),
        engine="tesseract",
        confidence=float(np.mean(confidences)) / 100.0,
    )


class FormulaReader:
    """Persistent PP-FormulaNet worker; the model costs far too much to reload."""

    def __init__(self, command: str | None = None) -> None:
        self._command = command or _require("formulaocr-offline")
        self._process: subprocess.Popen[str] | None = None
        self._counter = 0
        self._directory: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> FormulaReader:
        self._directory = tempfile.TemporaryDirectory()
        self._process = subprocess.Popen(
            [self._command, "--worker", "--no-classify"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        handshake = self._process.stdout.readline()  # type: ignore[union-attr]
        if not handshake or not json.loads(handshake).get("ready"):
            raise SystemExit("formulaocr-offline worker failed to start")
        return self

    def __exit__(self, *_exception: object) -> None:
        if self._process is not None:
            self._process.stdin.close()  # type: ignore[union-attr]
            self._process.wait(timeout=30)
            self._process = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    def read(self, image: np.ndarray) -> Reading:
        if self._process is None or self._directory is None:
            raise RuntimeError("FormulaReader used outside its context manager")
        self._counter += 1
        path = Path(self._directory.name) / f"crop-{self._counter:03d}.png"
        cv2.imwrite(str(path), image)

        request = json.dumps({"id": self._counter, "image": str(path)})
        self._process.stdin.write(request + "\n")  # type: ignore[union-attr]
        self._process.stdin.flush()  # type: ignore[union-attr]
        response = json.loads(self._process.stdout.readline())  # type: ignore[union-attr]
        if not response.get("ok"):
            return Reading(text="", engine="formulaocr", confidence=0.0)
        return Reading(text=response["formula"].strip(), engine="formulaocr", confidence=0.75)


_WORDLIKE = re.compile(r"^[A-Za-z][A-Za-z.'-]*$")


def looks_like_prose(reading: Reading) -> bool:
    """Prose is confidently recognised words and digits, not stray symbols."""
    if reading.confidence < 0.60 or not reading.text:
        return False
    tokens = reading.text.split()
    if not tokens:
        return False
    recognised = sum(
        1 for token in tokens if _WORDLIKE.match(token) or re.fullmatch(r"[0-9][0-9.,:-]*", token)
    )
    return recognised / len(tokens) >= 0.6
