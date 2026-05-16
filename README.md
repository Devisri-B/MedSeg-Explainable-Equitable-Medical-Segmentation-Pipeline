# MedSeg-RAI — Responsible Semantic Segmentation for Histopathology

**Automated, multi-class semantic segmentation of biological structures at the microscopic level — wrapped in the explainability, fairness, and monitoring layer that regulated healthcare AI demands.**

This project segments and *quantifies* five nucleus classes (plus background) across **19 tissue types** in H&E histopathology images, distinguishes **healthy vs. degraded** biological states (viable vs. necrotic/dead and neoplastic cells), and pairs the model with a full **Responsible-AI toolkit**: Seg-Grad-CAM explanations, per-pixel uncertainty, a fairness/bias audit across tissue subgroups, live data-drift + performance monitoring, and governance documentation mapped to the **IEEE ethical-AI criteria** and **FDA Good Machine Learning Practice**.

> ⚠️ **Research/portfolio project — not a medical device.** Not validated for clinical or diagnostic use.

---

## Why this exists

It was built to demonstrate the exact skill stack screened for in medical-imaging / medical-device AI roles. Each requirement below is satisfied by a concrete, runnable module:

| Role requirement | Where it lives in this repo |
|---|---|
| Automated **semantic segmentation** to **quantify & classify multiple complex classes** at microscopic level | [`medseg/train.py`](medseg/train.py), [`medseg/models/`](medseg/models), [`medseg/quantify.py`](medseg/quantify.py) — 6-class U-Net + per-class counts/areas |
| Evaluate **healthy vs. degraded** biological states → **therapeutic efficacy** | [`medseg/quantify.py`](medseg/quantify.py) — *tissue-degradation index* from Dead/Neoplastic vs. healthy classes |
| **IEEE ethical-AI** frameworks (Accountability, Transparency, Algorithmic Bias, Privacy) | [`docs/ETHICS_IEEE.md`](docs/ETHICS_IEEE.md) |
| **AI fairness analysis** & algorithmic bias | [`medseg/fairness/`](medseg/fairness) — per-tissue disparity audit across 19 subgroups |
| **Decision explainability** | [`medseg/explain/`](medseg/explain) — Seg-Grad-CAM + uncertainty maps |
| **Rigorous model performance monitoring** | [`medseg/monitoring/`](medseg/monitoring) — drift detection + alerting monitor |
| **Healthcare / regulated-industry** experience | [`docs/REGULATORY.md`](docs/REGULATORY.md), [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), [`docs/DATASHEET.md`](docs/DATASHEET.md) |
| Passion for innovation & **ethics of healthcare AI** | this README + the docs/ governance set |

---

## What it does, at a glance

- **Segmentation** — U-Net (ImageNet-pretrained encoder via `segmentation-models-pytorch`, with a dependency-free fallback) producing a 6-class pixel map.
- **Quantification** — connected-component counts, per-class area fractions, and a **tissue-degradation index** = degraded area / total cellular area, a proxy readout for therapeutic response.
- **Explainability** — *Seg-Grad-CAM* (which input regions drove a class) and *uncertainty maps* (softmax entropy / MC-dropout) so a reviewer can see where the model is unsure.
- **Fairness** — Dice/IoU computed **per tissue type**; reports worst-group performance, max–min gap, and disparity ratio to surface algorithmic bias.
- **Monitoring** — population-stability/KL **drift detection** on incoming image statistics plus a `PerformanceMonitor` that logs metrics over time and raises alerts when they breach thresholds.
- **Governance** — Model Card, Datasheet, IEEE ethics assessment, and FDA/HIPAA regulatory notes.

---

## Repository layout

```
medseg/
  config.py              Typed config + YAML/CLI merge
  data/
    pannuke.py           Download + load PanNuke, mask -> semantic label conversion
    synthetic.py         Dependency-free synthetic tissue images (for smoke tests/demo)
    dataset.py           torch Dataset, augmentation, train/val/test splits
  models/unet.py         U-Net builder (smp) + compact fallback + device selection
  losses.py              Dice + Cross-Entropy combined loss
  metrics.py             Confusion-matrix-based per-class Dice / IoU
  train.py               Training loop (AdamW, cosine, early stop, checkpointing)
  evaluate.py            Test-set evaluation + reports
  quantify.py            Counts, areas, tissue-degradation index
  explain/               Seg-Grad-CAM + uncertainty
  fairness/              Per-tissue bias audit
  monitoring/            Drift detection + performance monitor
  app/gradio_app.py      Interactive demo
docs/                    MODEL_CARD, DATASHEET, ETHICS_IEEE, REGULATORY
tests/                   Smoke test (runs without downloading data)
configs/default.yaml     All hyper-parameters
```

---

## Quickstart

### 1. Environment (Apple Silicon / macOS)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[full]"        # or: pip install -r requirements.txt
```

PyTorch will automatically use the **MPS (Metal)** backend on Apple Silicon.

### 2. Smoke test — no dataset required

Generates synthetic tissue images and runs the full pipeline (data → model → train step → metrics) to confirm everything is wired correctly:

```bash
pytest -q            # or: python -m tests.test_smoke
```

### 3. Get the data (PanNuke)

```bash
python scripts/download_data.py --root data/pannuke
```

PanNuke is a free academic dataset: ~7,900 256×256 H&E patches, 5 nucleus classes, 19 tissue types, distributed in 3 folds. See [`docs/DATASHEET.md`](docs/DATASHEET.md).

### 4. Train

```bash
python -m medseg.train --config configs/default.yaml
# quick run:  python -m medseg.train --config configs/default.yaml --epochs 5
```

### 5. Evaluate, explain, audit, monitor

```bash
python -m medseg.evaluate    --run outputs/pannuke_unet           # per-class metrics
python -m medseg.fairness.audit --run outputs/pannuke_unet        # per-tissue disparity
python -m medseg.monitoring.monitor --run outputs/pannuke_unet    # drift + alerts demo
python -m medseg.app.gradio_app --run outputs/pannuke_unet        # interactive demo
```

---

## How this transfers to other healthcare roles

The *organ* is incidental; the *competencies* are universal. The same pipeline maps onto:

| Target domain | Swap in | What stays identical |
|---|---|---|
| Radiology (CT/MRI/X-ray) | 3D U-Net + DICOM loader | losses, metrics, fairness, monitoring, governance |
| Ophthalmology (fundus/OCT) | vessel/lesion masks | the entire RAI layer |
| Dermatology | clinical photos | explainability + bias audit |
| Cell biology / drug discovery | live-cell microscopy | quantification + degradation index |

The data layer is deliberately the *only* part that's dataset-specific (`medseg/data/`). Everything else is modality-agnostic.

---

## Responsible AI & governance

This repo treats trustworthiness as a first-class deliverable, not an afterthought:

- **Accountability** — versioned configs, deterministic seeds, full run artifacts, and a Model Card.
- **Transparency** — Seg-Grad-CAM + uncertainty make each prediction inspectable; the Datasheet documents data provenance and known biases.
- **Algorithmic Bias** — the fairness audit quantifies performance disparities across tissue subgroups before deployment.
- **Privacy** — guidance on de-identification (HIPAA Safe Harbor) and why public, consented research data is used.

See [`docs/ETHICS_IEEE.md`](docs/ETHICS_IEEE.md) and [`docs/REGULATORY.md`](docs/REGULATORY.md).

---

## Interview talking points

- *"I built a 6-class histopathology segmentation model and, crucially, the governance layer around it: I audit Dice across all 19 tissue types and flag subgroups where the model underperforms — that's how you catch algorithmic bias before it reaches patients."*
- *"I quantify a tissue-degradation index from the necrotic/dead-cell class, which is the kind of readout used to evaluate therapeutic response."*
- *"For explainability I use Seg-Grad-CAM plus uncertainty maps, so a pathologist can see both *what* the model predicted and *how confident* it is."*
- *"I map the whole system to the IEEE ethical-AI pillars and FDA Good Machine Learning Practice, because in a regulated industry the model is only half the product."*

---

## License & citation

Code: MIT. PanNuke data is released for non-commercial research — cite Gamper et al. (2019/2020). Full references in [`docs/DATASHEET.md`](docs/DATASHEET.md).
