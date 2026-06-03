# Regulatory & Compliance Notes (healthcare / regulated industry)

Context for adapting this work toward a regulated setting. This is **educational**,
not legal or regulatory advice, and the project itself is **not** a medical device.
The point is to show fluency with the frameworks an employer in diagnostics/medical
devices (e.g. Abbott) operates under.

## 1. FDA — Software as a Medical Device (SaMD)
A model that drives a diagnostic or treatment decision is typically **SaMD** and
regulated by the FDA. Likely pathways: **510(k)** (substantial equivalence),
**De Novo** (novel, low–moderate risk), or **PMA** (high risk).

### Good Machine Learning Practice (GMLP)
The FDA/Health-Canada/MHRA **10 guiding principles** for ML in medical devices, and
how this repo already gestures at them:

| GMLP principle | Where this project engages it |
|---|---|
| 1. Multidisciplinary expertise across the lifecycle | Model Card "intended use"; docs invite clinical review |
| 2. Good software & security engineering | typed config, tests, reproducible runs, pinned deps |
| 3. Clinical study participants represent the intended population | Datasheet documents tissue coverage & gaps |
| 4. Training data independent of test data | train Folds 1–2, test Fold 3 (no leakage) |
| 5. Reference datasets are well characterised | Datasheet (PanNuke provenance, labels, biases) |
| 6. Model design tailored to data & intended use | U-Net + Dice/CE for imbalanced multi-class pixels |
| 7. Human–AI team performance considered | uncertainty + Grad-CAM support human-in-the-loop |
| 8. Testing demonstrates performance in clinically relevant conditions | fairness audit + drift stress-test |
| 9. Clear, essential information to users | Model Card, per-class metrics, confusion matrix |
| 10. Deployed models are monitored & re-managed | `medseg/monitoring` drift + performance alerting |

### Predetermined Change Control Plan (PCCP)
For models that update over time, the FDA expects a **PCCP** declaring *what* may
change, *how* it will be validated, and *what* triggers action. The monitoring module
(PSI drift thresholds + Dice degradation alerts) is the technical backbone of such a
plan: defined thresholds → defined response (recalibrate/retrain/roll back).

## 2. HIPAA (US privacy)
- **PHI** must be protected. For research/portfolio work, prefer **de-identified**
  data. The **Safe-Harbor** method removes 18 identifier types; histopathology image
  pixels generally carry none, but file metadata can — strip it.
- PanNuke is already de-identified public research data, so no PHI is processed here.
- Production systems handling PHI need BAAs, access controls, encryption at rest/in
  transit, and audit logging — outside this repo's scope but noted for completeness.

## 3. European Union
- **EU MDR 2017/745:** software with a medical purpose is a medical device and needs
  CE marking via a Notified Body.
- **EU AI Act:** medical-device AI is generally **high-risk**, triggering requirements
  for risk management, data governance, transparency, human oversight, and
  post-market monitoring — all of which mirror the modules in this project.
- **GDPR:** lawful basis, data minimisation, and (for automated decisions) a right to
  explanation — supported here by the explainability module.

## 4. Quality & traceability
- **IEC 62304** (medical device software lifecycle) and **ISO 14971** (risk management)
  would govern a real build; **ISO 13485** covers the quality management system.
- This repo's reproducible configs, versioned checkpoints, tests, and documentation are
  the kinds of artifacts those standards expect — scaled to a portfolio.

## TL;DR for an interview
> "I treated the model as one component of a regulated *system*: data provenance
> (Datasheet), intended-use limits (Model Card), bias evaluation (fairness audit),
> human-in-the-loop explainability, and post-market drift monitoring tied to defined
> thresholds — which is essentially the skeleton of an FDA GMLP + PCCP story."
