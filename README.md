# Cross-Layer Auditing Protocol (.aig Prototype)

A complete prototype verification system for the **Cross-Layer Auditing Protocol (CLAP)**, designed to eliminate "integrity conflict" vulnerabilities in AI-generated content provenance. Jointly evaluates C2PA metadata signatures and DWT-domain invisible watermarks through a decision matrix to produce authoritative provenance verdicts.


## Core Idea

Existing AI content provenance schemes (pure C2PA or pure watermarking) suffer from single-point-of-failure vulnerabilities. An attacker can strip C2PA metadata, forge signatures, or overwrite watermarks, creating an exploitable gap between the two layers.

The `.aig` (AI-Generated) format is the first reference implementation of CLAP for images:

1. **Parse** the `.aig` file to extract the C2PA metadata and verify its Ed25519 digital signature.
2. **Read** the image payload and extract the DWT-domain invisible watermark.
3. **Cross-reference** both results through the **Audit Decision Matrix** — if both layers agree, the file is **Trustable**; any single-layer failure or cross-layer mismatch triggers a **Suspicious** or **Untrusted** verdict.

## Environment & Installation

Python 3.10+ recommended. Install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install cryptography Pillow numpy PyWavelets matplotlib
```

## File Structure

```
project_root/
├── aig_format.py          # .aig binary format definition & C2PA metadata layer
├── aig_watermark.py       # DWT-based blind watermark embedding & detection
├── aig_cross_layer.py     # Cross-layer audit engine, attack simulations,
│                          #   benchmarks, paper figures & LaTeX generation
├── README.md              # This file
├── experiment.log         # Auto-generated run log
└── figures/               # Auto-generated 300 dpi paper figures
    ├── fig1_performance.png
    ├── fig2_overhead.png
    ├── fig3_attack_matrix.png
    └── fig4_radar_comparison.png
```

## Usage

Run the complete experiment suite from the project root:

```bash
python aig_cross_layer.py
```

The script executes six phases in sequence:

| Phase | Description |
|:---|:---|
| 1 | Generate Ed25519 keypair and synthetic test images (256, 512, 1024) |
| 2 | Standard provenance workflow → expected verdict: **Trustable** |
| 3 | Four security attack simulations → all should be detected |
| 4 | Performance benchmarks (30 iterations per resolution) |
| 5 | File-overhead analysis at three resolutions |
| 6 | Generate 4 paper figures (300 dpi), LaTeX tables, and paper text |

Individual modules can also be tested standalone:

```bash
python aig_format.py      # Self-test: pack → unpack → verify round-trip
python aig_watermark.py   # Self-test: embed → detect → BER measurement
```

## Expected Output

### Standard Workflow
```
C2PA verification: PASS
Watermark detection: PASS
BER: <0.01
>>> FINAL VERDICT: Trustable <<<
```

### Attack Simulations
| Attack | C2PA Alone | WM Alone | CLAP (Ours) |
|:---|:---|:---|:---|
| Signature Stripping | Vulnerable (missed) | Detected | **Detected** |
| C2PA Spoofing | Detected | Vulnerable (missed) | **Detected** |
| Watermark Overwrite | Vulnerable (missed) | Detected | **Detected** |
| Integrity Conflict | Vulnerable (missed) | Detected | **Detected** |

CLAP achieves **100% detection rate** — it catches attacks that single-layer schemes miss.

### Performance (512x512, Apple Silicon M-series)
| Operation | Latency |
|:---|:---|
| Pack .aig | ~0.4 ms |
| Sign C2PA | ~0.4 ms |
| Verify C2PA | ~0.4 ms |
| Embed Watermark | ~23 ms |
| Detect Watermark | ~56 ms |
| Full Audit | ~60 ms |

### File Overhead
| Resolution | Overhead |
|:---|:---|
| 256×256 | ~47% |
| 512×512 | ~18% |
| 1024×1024 | ~10% |

The fixed metadata overhead is ~930 bytes regardless of resolution; the percentage shrinks as image size grows.

## Paper Figure Guide

- **fig1_performance.png** — Bar chart comparing operation latency across three resolutions (log scale).
- **fig2_overhead.png** — Pie chart showing metadata overhead vs. image payload (512×512).
- **fig3_attack_matrix.png** — Heatmap demonstrating that only CLAP detects all four attack types.
- **fig4_radar_comparison.png** — Multi-dimensional comparison: CLAP vs. Pure C2PA vs. Pure Watermark.

## Key Design Decisions

1. **Watermark payload** = SHA-256 of the canonical C2PA manifest (before hash/time are filled). Stored as a signed `clap.watermark` assertion inside the manifest itself, creating cryptographic binding between layers.
2. **DWT (Haar, 2-level) + QIM** chosen over spatial-domain LSB for robustness against lossy compression.
3. **Lossless WebP** encoding in the prototype ensures watermark survival; production deployment would add error-correction coding for lossy tolerance.
4. **Ed25519** for C2PA signatures — fast, compact (64-byte signatures), and widely adopted.


---
*Author: Cross-Layer Auditing Protocol Research Team*
*License: MIT*
