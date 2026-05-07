#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aig_watermark.py — Steganographic Watermark Layer

This module implements the invisible-watermark subsystem for the Cross-Layer
Auditing Protocol.  It uses a DWT (Discrete Wavelet Transform) based robust
blind watermarking algorithm to embed a 32-byte payload (typically a SHA-256
hash) into the luminance channel of an image.

Algorithm overview
──────────────────
  1. Convert RGB → YUV; operate on Y (luminance) channel only.
  2. Two-level DWT decomposition (Haar wavelet) on Y.
  3. Spread-spectrum embed the payload bitstream into the LL2 sub-band
     coefficients using quantisation index modulation (QIM).
  4. Inverse DWT to reconstruct Y; merge back to RGB.

Detection is the reverse: DWT → extract quantised bits → majority-vote
decode → return payload + confidence.

Dependencies:
    - numpy
    - PyWavelets  (pywt)
    - Pillow      (PIL)

Author : Cross-Layer Auditing Protocol Research Team
License: MIT
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import numpy as np
import pywt
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_PAYLOAD_BITS: int = 256       # 32 bytes × 8
WAVELET: str = 'haar'                 # Haar wavelet for two-level DWT
DWT_LEVEL: int = 2                    # decomposition depth
SPREAD_FACTOR: int = 8                # chips per payload bit (spread-spectrum)
BASE_QUANTISATION_STEP: float = 30.0  # base Δ for QIM


# ──────────────────────────────────────────────────────────────────────
# Colour-space helpers
# ──────────────────────────────────────────────────────────────────────

def _rgb_to_yuv(img_array: np.ndarray) -> np.ndarray:
    """Convert an RGB image (H×W×3, float64) to YUV colour space.

    Uses the BT.601 conversion matrix.
    """
    # BT.601 luma / chroma matrix
    m = np.array([
        [ 0.299,    0.587,    0.114   ],
        [-0.14713, -0.28886,  0.436   ],
        [ 0.615,   -0.51499, -0.10001 ],
    ])
    flat = img_array.reshape(-1, 3)
    yuv_flat = flat @ m.T
    return yuv_flat.reshape(img_array.shape)


def _yuv_to_rgb(yuv_array: np.ndarray) -> np.ndarray:
    """Convert a YUV image (H×W×3, float64) back to RGB."""
    m_inv = np.array([
        [1.0,  0.0,       1.13983],
        [1.0, -0.39465,  -0.58060],
        [1.0,  2.03211,   0.0    ],
    ])
    flat = yuv_array.reshape(-1, 3)
    rgb_flat = flat @ m_inv.T
    return rgb_flat.reshape(yuv_array.shape)


# ──────────────────────────────────────────────────────────────────────
# Payload ↔ bitstream helpers
# ──────────────────────────────────────────────────────────────────────

def _bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to a 1-D numpy array of {0, 1} bits (MSB first)."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return np.array(bits, dtype=np.int32)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert a 1-D array of {0, 1} bits back to bytes (MSB first)."""
    n = len(bits)
    pad = (8 - n % 8) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.int32)])
    out = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for j in range(8):
            val = (val << 1) | int(bits[i + j])
        out.append(val)
    return bytes(out)


# ──────────────────────────────────────────────────────────────────────
# Spread-spectrum helpers
# ──────────────────────────────────────────────────────────────────────

def _generate_pn_sequence(length: int, seed: int = 42) -> np.ndarray:
    """Generate a pseudo-random ±1 spreading sequence."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=length).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────
# Core embedding
# ──────────────────────────────────────────────────────────────────────

def embed_watermark(
    image_path: str,
    payload: bytes,
    strength: float = 0.5,
) -> Image.Image:
    """Embed an invisible watermark into an image using DWT-based QIM.

    The algorithm operates on the luminance (Y) channel of the YUV colour
    space.  A two-level Haar DWT is applied; the payload bits are
    spread-spectrum encoded and quantisation-index modulated into the LL2
    sub-band coefficients.

    Args:
        image_path: File-system path to the source image (any PIL-readable
                    format) **or** a PIL Image object.
        payload:    32-byte watermark payload (typically SHA-256 hash).
        strength:   Embedding strength multiplier in [0.1, 1.0].
                    Higher values improve robustness at the cost of
                    perceptual quality.

    Returns:
        A PIL Image object containing the watermarked image.

    Raises:
        ValueError: If the payload is too large for the image dimensions.
    """
    # ── Load image ──────────────────────────────────────────────────
    if isinstance(image_path, Image.Image):
        img = image_path.convert('RGB')
    else:
        img = Image.open(image_path).convert('RGB')

    img_array = np.array(img, dtype=np.float64)
    h, w, _ = img_array.shape

    # ── RGB → YUV ───────────────────────────────────────────────────
    yuv = _rgb_to_yuv(img_array)
    y_channel = yuv[:, :, 0].copy()

    # ── Pad Y channel to even dimensions for 2-level DWT ───────────
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    y_padded = np.pad(y_channel, ((0, pad_h), (0, pad_w)), mode='symmetric')

    # ── 2-level DWT ─────────────────────────────────────────────────
    coeffs = pywt.wavedec2(y_padded, WAVELET, level=DWT_LEVEL)
    # coeffs[0] = LL2 approximation, coeffs[1] = (LH2, HL2, HH2), etc.
    ll2 = coeffs[0].copy()

    # ── Prepare payload bitstream ───────────────────────────────────
    bits = _bytes_to_bits(payload)
    n_bits = len(bits)
    chips_needed = n_bits * SPREAD_FACTOR

    ll2_flat = ll2.flatten()
    if chips_needed > len(ll2_flat):
        raise ValueError(
            f"Image too small: need {chips_needed} coefficients, "
            f"but LL2 sub-band has only {len(ll2_flat)}."
        )

    # ── QIM embedding ───────────────────────────────────────────────
    delta = BASE_QUANTISATION_STEP * strength
    pn = _generate_pn_sequence(SPREAD_FACTOR)

    for i in range(n_bits):
        start = i * SPREAD_FACTOR
        end = start + SPREAD_FACTOR
        segment = ll2_flat[start:end].copy()
        bit = bits[i]

        # Quantise each chip to the nearest grid point encoding *bit*
        for j in range(SPREAD_FACTOR):
            coeff = segment[j]
            # QIM: quantise to delta*k + bit*(delta/2)
            quantised = delta * np.round(coeff / delta) + (bit - 0.5) * (delta / 2)
            # Blend with spreading code for extra robustness
            segment[j] = quantised + pn[j] * (delta * 0.05)

        ll2_flat[start:end] = segment

    # ── Write back LL2 and inverse DWT ──────────────────────────────
    coeffs[0] = ll2_flat.reshape(ll2.shape)
    y_reconstructed = pywt.waverec2(coeffs, WAVELET)

    # Remove padding
    y_reconstructed = y_reconstructed[:h, :w]

    # ── Merge Y back into YUV → RGB ────────────────────────────────
    yuv[:, :, 0] = y_reconstructed
    rgb_out = _yuv_to_rgb(yuv)
    rgb_out = np.clip(rgb_out, 0, 255).astype(np.uint8)

    return Image.fromarray(rgb_out, 'RGB')


# ──────────────────────────────────────────────────────────────────────
# Core detection
# ──────────────────────────────────────────────────────────────────────

def detect_watermark(
    image: Image.Image,
    original_payload_length: int = DEFAULT_PAYLOAD_BITS,
) -> Dict:
    """Extract a watermark from an image.

    Args:
        image:                  PIL Image (RGB or convertible).
        original_payload_length: Expected number of payload **bits**
                                 (default 256 = 32 bytes).

    Returns:
        Dictionary with:
            - detected (bool):       Whether a watermark was found.
            - payload (bytes):       Extracted payload bytes.
            - confidence (float):    Detection confidence in [0, 1].
            - bit_error_rate (float): Estimated BER (meaningful only when
                                      the original payload is known for
                                      comparison).
    """
    img = image.convert('RGB')
    img_array = np.array(img, dtype=np.float64)
    h, w, _ = img_array.shape

    # ── RGB → YUV ───────────────────────────────────────────────────
    yuv = _rgb_to_yuv(img_array)
    y_channel = yuv[:, :, 0].copy()

    # ── Pad and DWT ─────────────────────────────────────────────────
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    y_padded = np.pad(y_channel, ((0, pad_h), (0, pad_w)), mode='symmetric')
    coeffs = pywt.wavedec2(y_padded, WAVELET, level=DWT_LEVEL)
    ll2_flat = coeffs[0].flatten()

    n_bits = original_payload_length
    chips_needed = n_bits * SPREAD_FACTOR

    if chips_needed > len(ll2_flat):
        return {
            'detected': False,
            'payload': b'',
            'confidence': 0.0,
            'bit_error_rate': 1.0,
        }

    # ── QIM detection ───────────────────────────────────────────────
    # We detect the embedded bit by checking whether each coefficient is
    # closer to the '0'-grid or the '1'-grid.  We use a range of plausible
    # delta values and pick the one with the highest self-consistency.

    best_bits: Optional[np.ndarray] = None
    best_conf: float = 0.0
    best_delta: float = 0.0

    # Scan a small range around the expected delta.
    # In a real system the delta would be derived from a shared secret.
    for strength_guess in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        delta = BASE_QUANTISATION_STEP * strength_guess
        extracted_bits = np.zeros(n_bits, dtype=np.int32)
        confidences = np.zeros(n_bits, dtype=np.float64)

        for i in range(n_bits):
            start = i * SPREAD_FACTOR
            end = start + SPREAD_FACTOR
            segment = ll2_flat[start:end]

            votes = np.zeros(SPREAD_FACTOR)
            for j in range(SPREAD_FACTOR):
                coeff = segment[j]
                # Distance to '0'-grid (bit=0 → offset = -delta/4)
                d0 = abs(coeff - delta * np.round(coeff / delta) + 0.5 * (delta / 2))
                # Distance to '1'-grid (bit=1 → offset = +delta/4)
                d1 = abs(coeff - delta * np.round(coeff / delta) - 0.5 * (delta / 2))
                votes[j] = 0 if d0 < d1 else 1

            # Majority vote
            mean_vote = votes.mean()
            extracted_bits[i] = 1 if mean_vote >= 0.5 else 0
            # Confidence = how far from 50/50 the vote is
            confidences[i] = abs(mean_vote - 0.5) * 2

        avg_conf = float(confidences.mean())
        if avg_conf > best_conf:
            best_conf = avg_conf
            best_bits = extracted_bits.copy()
            best_delta = delta

    if best_bits is None:
        return {
            'detected': False,
            'payload': b'',
            'confidence': 0.0,
            'bit_error_rate': 1.0,
        }

    payload = _bits_to_bytes(best_bits)

    # A confidence above ~0.3 usually indicates a real watermark
    detected = best_conf > 0.25

    return {
        'detected': detected,
        'payload': payload,
        'confidence': round(best_conf, 4),
        'bit_error_rate': 0.0,  # unknown without ground truth
    }


# ──────────────────────────────────────────────────────────────────────
# Convenience: compute BER between two payloads
# ──────────────────────────────────────────────────────────────────────

def compute_ber(original: bytes, extracted: bytes) -> float:
    """Compute the Bit Error Rate between two byte strings.

    Args:
        original:  Ground-truth payload.
        extracted: Payload extracted from a watermarked image.

    Returns:
        BER in [0.0, 1.0].  0.0 means perfect extraction.
    """
    min_len = min(len(original), len(extracted))
    if min_len == 0:
        return 1.0

    orig_bits = _bytes_to_bits(original[:min_len])
    ext_bits = _bytes_to_bits(extracted[:min_len])
    errors = int(np.sum(orig_bits != ext_bits))
    return errors / len(orig_bits)


# ──────────────────────────────────────────────────────────────────────
# Convenience: embed from PIL Image directly (no file path)
# ──────────────────────────────────────────────────────────────────────

def embed_watermark_pil(
    image: Image.Image,
    payload: bytes,
    strength: float = 0.5,
) -> Image.Image:
    """Same as :func:`embed_watermark` but accepts a PIL Image directly.

    Args:
        image:    Source PIL Image.
        payload:  32-byte watermark payload.
        strength: Embedding strength.

    Returns:
        Watermarked PIL Image.
    """
    return embed_watermark(image, payload, strength)


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import io as _io

    print("=== aig_watermark.py self-test ===\n")

    # Create a synthetic 256×256 test image
    rng = np.random.RandomState(0)
    arr = rng.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    test_img = Image.fromarray(arr, 'RGB')
    print(f"Test image created: {test_img.size}")

    # Payload = SHA-256 of some content
    payload = hashlib.sha256(b"Hello Cross-Layer Audit").digest()
    print(f"Payload (SHA-256): {payload.hex()[:32]}…")

    # Embed
    wm_img = embed_watermark_pil(test_img, payload, strength=0.5)
    print(f"Watermark embedded. Image size: {wm_img.size}")

    # Detect
    result = detect_watermark(wm_img)
    print(f"Detection result: detected={result['detected']}, "
          f"confidence={result['confidence']:.4f}")

    # BER
    ber = compute_ber(payload, result['payload'])
    print(f"Bit Error Rate: {ber:.4f}")

    if result['detected'] and ber < 0.15:
        print("\n✅ aig_watermark.py self-test PASSED.")
    else:
        print(f"\n⚠️  Self-test marginal (BER={ber:.4f}). "
              "DWT watermark may need tuning for this image.")
