#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aig_format.py — .aig Format Definition & Metadata Layer

This module defines the binary layout of the .aig (AI-Generated) file format
and implements the metadata-layer operations for the Cross-Layer Auditing
Protocol (CLAP).  The .aig format bundles:

  1. Generation provenance (model fingerprint, prompts, parameters)
  2. A C2PA-style manifest with an Ed25519 digital signature
  3. The image payload (WebP-encoded)

All multi-byte integers are stored in **big-endian** order.

Reference implementation for:
    "Cross-Layer Auditing Protocol: Eliminating Integrity Conflicts
     in AI-Generated Content Provenance"

Dependencies:
    - cryptography  (Ed25519 signing / verification)
    - hashlib, struct, json, time  (stdlib)

Author : Cross-Layer Auditing Protocol Research Team
License: MIT
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import time
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

AIG_MAGIC: bytes = b'\x89AIG\x0D\x0A\x1A\x0A'        # 8 bytes
AIG_VERSION: int = 0x0100                               # uint16 – v1.0
AIG_EOF_MARKER: bytes = b'\xA1\x60\x00\xD0'            # 4 bytes

# Content-origin type codes
ORIGIN_STABLE_DIFFUSION: int = 0x01
ORIGIN_DALLE: int = 0x02
ORIGIN_MIDJOURNEY: int = 0x03
ORIGIN_OTHER: int = 0x04

# Signature algorithm identifiers
SIG_ALG_ED25519: int = 0x01

# Encoder identifiers
ENCODER_WEBP: int = 0x01

# Fixed-size field lengths
MODEL_FINGERPRINT_LEN: int = 32   # SHA-256 digest
RESERVED_LEN: int = 8
SEED_LEN: int = 8
TIMESTAMP_LEN: int = 8


# ──────────────────────────────────────────────────────────────────────
# Key helpers
# ──────────────────────────────────────────────────────────────────────

def generate_ed25519_keypair() -> Tuple[bytes, bytes]:
    """Generate a fresh Ed25519 key pair.

    Returns:
        (private_key_bytes, public_key_bytes) – raw 32-byte representations.
    """
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()
    sk_bytes = sk.private_bytes_raw()
    pk_bytes = pk.public_bytes_raw()
    return sk_bytes, pk_bytes


def _load_private_key(raw: bytes) -> Ed25519PrivateKey:
    """Reconstruct an Ed25519PrivateKey from 32 raw bytes."""
    return Ed25519PrivateKey.from_private_bytes(raw)


def _load_public_key(raw: bytes) -> Ed25519PublicKey:
    """Reconstruct an Ed25519PublicKey from 32 raw bytes."""
    return Ed25519PublicKey.from_public_bytes(raw)


# ──────────────────────────────────────────────────────────────────────
# Model fingerprint utility
# ──────────────────────────────────────────────────────────────────────

def compute_model_fingerprint(model_name: str, model_version: str) -> bytes:
    """Return SHA-256(model_name + model_version) as 32 bytes.

    Args:
        model_name:    Human-readable model identifier, e.g. "stable-diffusion".
        model_version: Version string, e.g. "xl-1.0".

    Returns:
        32-byte digest used as the model fingerprint field.
    """
    data = (model_name + model_version).encode('utf-8')
    return hashlib.sha256(data).digest()


# ──────────────────────────────────────────────────────────────────────
# C2PA manifest helpers
# ──────────────────────────────────────────────────────────────────────

def build_c2pa_manifest(
    *,
    generator: str,
    model_name: str,
    model_version: str,
    signer: str = "CLAP Prototype Signer",
    action: str = "c2pa.created",
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    """Construct a minimal C2PA-compatible manifest dictionary.

    The manifest follows the C2PA 1.x claim structure with custom
    AI-generation assertions.

    Args:
        generator:     Tool that produced the image (e.g. "StableDiffusion").
        model_name:    Name of the generative model.
        model_version: Version of the generative model.
        signer:        Identity of the signing entity.
        action:        C2PA action string.
        extra:         Arbitrary additional fields.

    Returns:
        A JSON-serialisable dictionary.
    """
    manifest: Dict[str, Any] = {
        "claim_generator": f"CLAP/{AIG_VERSION:#06x}",
        "title": "AI-Generated Image",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": action,
                            "softwareAgent": generator,
                            "parameters": {
                                "model_name": model_name,
                                "model_version": model_version,
                            },
                        }
                    ]
                },
            },
            {
                "label": "c2pa.hash.data",
                "data": {
                    "name": "image_payload",
                    "algorithm": "SHA-256",
                    # Hash will be filled during packing
                    "hash": "",
                },
            },
        ],
        "signature_info": {
            "signer": signer,
            "algorithm": "Ed25519",
            "time": "",  # filled during packing
        },
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def sign_c2pa_manifest(manifest: dict, private_key: bytes) -> bytes:
    """Sign the canonical JSON encoding of *manifest* with Ed25519.

    The manifest is serialised with sorted keys and no extra whitespace
    to guarantee a deterministic byte representation.

    Args:
        manifest:    C2PA manifest dictionary.
        private_key: 32-byte Ed25519 private key.

    Returns:
        64-byte Ed25519 signature.
    """
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')
    sk = _load_private_key(private_key)
    return sk.sign(canonical)


def verify_c2pa_signature(file_bytes: bytes, public_key: bytes) -> dict:
    """Verify the C2PA signature embedded in an .aig file.

    This function unpacks the file to locate the manifest and its
    signature, then verifies integrity.

    Args:
        file_bytes: Complete .aig file content.
        public_key: 32-byte Ed25519 public key.

    Returns:
        A dictionary with keys:
            - valid (bool):   Whether the signature is cryptographically valid.
            - manifest (dict): The extracted C2PA manifest.
            - signer (str):   The signer identity from the manifest.
            - algorithm (str): Signature algorithm name.
            - error (str):    Human-readable error message (empty on success).
    """
    result: Dict[str, Any] = {
        "valid": False,
        "manifest": {},
        "signer": "",
        "algorithm": "",
        "error": "",
    }

    try:
        parsed = unpack_aig(file_bytes)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"Unpack failed: {exc}"
        return result

    manifest = parsed.get("c2pa_manifest", {})
    sig_bytes = parsed.get("signature", b"")
    result["manifest"] = manifest
    result["signer"] = manifest.get("signature_info", {}).get("signer", "")
    result["algorithm"] = manifest.get("signature_info", {}).get("algorithm", "")

    # Recompute canonical encoding
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')

    try:
        pk = _load_public_key(public_key)
        pk.verify(sig_bytes, canonical)
        result["valid"] = True
    except InvalidSignature:
        result["error"] = "Signature verification failed – data may have been tampered."
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"Verification error: {exc}"

    return result


# ──────────────────────────────────────────────────────────────────────
# Pack / Unpack
# ──────────────────────────────────────────────────────────────────────

def pack_aig(
    metadata: dict,
    image_bytes: bytes,
    private_key: bytes,
    public_key: bytes,
) -> bytes:
    """Serialise all components into the .aig binary format.

    Args:
        metadata: Dictionary with the following expected keys:
            - origin_type (int):       One of ORIGIN_* constants.
            - model_name (str):        Generative model name.
            - model_version (str):     Generative model version.
            - prompt (str):            Generation prompt.
            - negative_prompt (str):   Negative prompt (may be empty).
            - seed (int):              Random seed used for generation.
            - generation_params (dict): Additional params (steps, cfg, …).
            - c2pa_manifest (dict):    Pre-built C2PA manifest
                                       (signature_info.time & hash will be
                                       filled automatically).
        image_bytes: WebP-encoded image payload.
        private_key: 32-byte Ed25519 private key.
        public_key:  32-byte Ed25519 public key.

    Returns:
        Complete .aig file as a bytes object.
    """
    buf = io.BytesIO()

    # ── 1. Magic (8 B) ──
    buf.write(AIG_MAGIC)

    # ── 2. Version (2 B) ──
    buf.write(struct.pack('>H', AIG_VERSION))

    # ── 3. Content origin type (1 B) ──
    origin = metadata.get('origin_type', ORIGIN_OTHER)
    buf.write(struct.pack('>B', origin))

    # ── 4. Model fingerprint (32 B) ──
    fp = compute_model_fingerprint(
        metadata.get('model_name', ''),
        metadata.get('model_version', ''),
    )
    buf.write(fp)

    # ── 5. Reserved (8 B) ──
    buf.write(b'\x00' * RESERVED_LEN)

    # ── 6. Prompt ──
    prompt_bytes = metadata.get('prompt', '').encode('utf-8')
    buf.write(struct.pack('>I', len(prompt_bytes)))
    buf.write(prompt_bytes)

    # ── 7. Negative prompt ──
    neg_prompt_bytes = metadata.get('negative_prompt', '').encode('utf-8')
    buf.write(struct.pack('>I', len(neg_prompt_bytes)))
    buf.write(neg_prompt_bytes)

    # ── 8. Seed (8 B) ──
    buf.write(struct.pack('>Q', metadata.get('seed', 0)))

    # ── 9. Generation parameters (variable-length JSON) ──
    gen_params = metadata.get('generation_params', {})
    gen_json = json.dumps(gen_params, sort_keys=True).encode('utf-8')
    buf.write(struct.pack('>I', len(gen_json)))
    buf.write(gen_json)

    # ── 10. Timestamp (8 B – Unix ms) ──
    ts_ms = int(time.time() * 1000)
    buf.write(struct.pack('>Q', ts_ms))

    # ── 11. C2PA manifest ──
    manifest: dict = metadata.get('c2pa_manifest', {})
    # Fill the image-payload hash
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    for assertion in manifest.get('assertions', []):
        if assertion.get('label') == 'c2pa.hash.data':
            assertion['data']['hash'] = image_hash
    # Fill signing timestamp
    sig_info = manifest.setdefault('signature_info', {})
    sig_info['time'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    manifest_json = json.dumps(manifest, sort_keys=True).encode('utf-8')
    buf.write(struct.pack('>I', len(manifest_json)))
    buf.write(manifest_json)

    # ── 12. Signature block ──
    signature = sign_c2pa_manifest(manifest, private_key)

    buf.write(struct.pack('>B', SIG_ALG_ED25519))  # algorithm id
    buf.write(struct.pack('>H', len(public_key)))   # public key length
    buf.write(public_key)                            # public key
    buf.write(struct.pack('>H', len(signature)))     # signature length
    buf.write(signature)                             # signature

    # ── 13. Image payload ──
    buf.write(struct.pack('>B', ENCODER_WEBP))       # encoder id
    buf.write(struct.pack('>I', len(image_bytes)))   # payload length
    buf.write(image_bytes)                           # payload

    # ── 14. EOF marker (4 B) ──
    buf.write(AIG_EOF_MARKER)

    return buf.getvalue()


def unpack_aig(file_bytes: bytes) -> dict:
    """Deserialise an .aig binary blob into its constituent fields.

    Args:
        file_bytes: Complete .aig file content.

    Returns:
        Dictionary with all parsed fields.  Keys mirror the binary layout:
            magic, version, origin_type, model_fingerprint, prompt,
            negative_prompt, seed, generation_params, timestamp_ms,
            c2pa_manifest, sig_algorithm, public_key, signature,
            encoder_id, image_bytes, eof_marker.

    Raises:
        ValueError: If the magic number or EOF marker is invalid.
    """
    buf = io.BytesIO(file_bytes)
    result: Dict[str, Any] = {}

    # 1. Magic
    magic = buf.read(8)
    if magic != AIG_MAGIC:
        raise ValueError(f"Invalid magic number: {magic!r}")
    result['magic'] = magic

    # 2. Version
    result['version'] = struct.unpack('>H', buf.read(2))[0]

    # 3. Origin type
    result['origin_type'] = struct.unpack('>B', buf.read(1))[0]

    # 4. Model fingerprint
    result['model_fingerprint'] = buf.read(MODEL_FINGERPRINT_LEN)

    # 5. Reserved
    result['reserved'] = buf.read(RESERVED_LEN)

    # 6. Prompt
    prompt_len = struct.unpack('>I', buf.read(4))[0]
    result['prompt'] = buf.read(prompt_len).decode('utf-8')

    # 7. Negative prompt
    neg_len = struct.unpack('>I', buf.read(4))[0]
    result['negative_prompt'] = buf.read(neg_len).decode('utf-8')

    # 8. Seed
    result['seed'] = struct.unpack('>Q', buf.read(8))[0]

    # 9. Generation params
    gen_len = struct.unpack('>I', buf.read(4))[0]
    result['generation_params'] = json.loads(buf.read(gen_len).decode('utf-8'))

    # 10. Timestamp
    result['timestamp_ms'] = struct.unpack('>Q', buf.read(8))[0]

    # 11. C2PA manifest
    manifest_len = struct.unpack('>I', buf.read(4))[0]
    result['c2pa_manifest'] = json.loads(buf.read(manifest_len).decode('utf-8'))

    # 12. Signature block
    result['sig_algorithm'] = struct.unpack('>B', buf.read(1))[0]
    pk_len = struct.unpack('>H', buf.read(2))[0]
    result['public_key'] = buf.read(pk_len)
    sig_len = struct.unpack('>H', buf.read(2))[0]
    result['signature'] = buf.read(sig_len)

    # 13. Image payload
    result['encoder_id'] = struct.unpack('>B', buf.read(1))[0]
    img_len = struct.unpack('>I', buf.read(4))[0]
    result['image_bytes'] = buf.read(img_len)

    # 14. EOF marker
    eof = buf.read(4)
    if eof != AIG_EOF_MARKER:
        raise ValueError(f"Invalid EOF marker: {eof!r}")
    result['eof_marker'] = eof

    return result


# ──────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────

def compute_image_hash(image_bytes: bytes) -> str:
    """Return the hex-encoded SHA-256 of the raw image payload."""
    return hashlib.sha256(image_bytes).hexdigest()


def get_header_size(file_bytes: bytes) -> int:
    """Return the byte offset where the image payload begins.

    Useful for measuring the metadata overhead of the .aig format.
    """
    parsed = unpack_aig(file_bytes)
    total = len(file_bytes)
    img_len = len(parsed['image_bytes'])
    eof_len = 4  # EOF marker
    return total - img_len - eof_len


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== aig_format.py self-test ===")

    # Generate key pair
    sk, pk = generate_ed25519_keypair()
    print(f"Ed25519 key pair generated. PK={pk.hex()[:16]}…")

    # Build a tiny test image (1×1 white WebP placeholder)
    from PIL import Image as PILImage
    img = PILImage.new('RGB', (64, 64), color=(200, 100, 50))
    img_buf = io.BytesIO()
    img.save(img_buf, format='WEBP')
    test_image_bytes = img_buf.getvalue()

    # Build metadata
    manifest = build_c2pa_manifest(
        generator="StableDiffusion",
        model_name="stable-diffusion",
        model_version="xl-1.0",
    )
    meta = {
        'origin_type': ORIGIN_STABLE_DIFFUSION,
        'model_name': 'stable-diffusion',
        'model_version': 'xl-1.0',
        'prompt': 'A futuristic cityscape at sunset',
        'negative_prompt': 'blurry, low quality',
        'seed': 42,
        'generation_params': {'steps': 30, 'cfg_scale': 7.5, 'sampler': 'euler_a'},
        'c2pa_manifest': manifest,
    }

    # Pack
    aig_bytes = pack_aig(meta, test_image_bytes, sk, pk)
    print(f"Packed .aig file: {len(aig_bytes)} bytes")

    # Unpack
    parsed = unpack_aig(aig_bytes)
    assert parsed['magic'] == AIG_MAGIC
    assert parsed['version'] == AIG_VERSION
    assert parsed['prompt'] == 'A futuristic cityscape at sunset'
    assert parsed['image_bytes'] == test_image_bytes
    print("Unpack OK – all fields match.")

    # Verify signature
    vr = verify_c2pa_signature(aig_bytes, pk)
    assert vr['valid'], f"Signature verification failed: {vr['error']}"
    print(f"C2PA signature VALID. Signer: {vr['signer']}")

    # Header size
    hdr = get_header_size(aig_bytes)
    print(f"Header overhead: {hdr} bytes ({hdr/len(aig_bytes)*100:.1f}%)")

    print("\n✅ aig_format.py self-test PASSED.")
