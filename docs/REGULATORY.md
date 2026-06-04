# Regulatory and Compliance Notes (healthcare and regulated industry)

Context for adapting this work toward a regulated setting. This is educational, not legal or
regulatory advice, and the project itself is not a medical device. The point is to show fluency
with the frameworks an employer in diagnostics or medical devices (for example Abbott) operates
under.

## 1. FDA, Software as a Medical Device (SaMD)
A model that drives a diagnostic or treatment decision is typically SaMD and regulated by the
FDA. Likely pathways are 510(k) (substantial equivalence), De Novo (novel, low to moderate
risk), or PMA (high risk).

### Good Machine Learning Practice (GMLP)
The FDA, Health Canada, and MHRA published 10 guiding principles for machine learning in medical
devices. How this repo engages each one:

| GMLP principle | Where this project engages it |
|---|---|
| 1. Multidisciplinary expertise across the lifecycle | Model Card intended use; docs invite clinical review |
| 2. Good software and security engineering | typed config, tests, reproducible runs, pinned dependencies |
| 3. Data represents the intended population | Datasheet documents tissue coverage and gaps |
| 4. Training data independent of test data | train folds 1 and 2, test fold 3, no leakage |
| 5. Reference datasets are well characterised | Datasheet covers PanNuke provenance, labels, biases |
| 6. Model design tailored to data and intended use | U-Net++ with Focal-Tversky for imbalanced multi-class pixels |
| 7. Human and AI team performance considered | uncertainty and Grad-CAM support human-in-the-loop review |
| 8. Testing shows performance in clinically relevant conditions | fairness audit across tissue and stain, plus drift stress test |
| 9. Clear, essential information for users | Model Card, per-class metrics, confusion matrix |
| 10. Deployed models are monitored and re-managed | `medseg/monitoring` drift and performance alerting |

On principle 8, the fairness audit does not just claim coverage. It measures performance per
tissue type and per stain brightness and flags the groups that fall below the four-fifths rule
(see [RESULTS.md](RESULTS.md)), which is the kind of subgroup evidence a submission needs.

### Predetermined Change Control Plan (PCCP)
For models that update over time, the FDA expects a PCCP that declares what may change, how it
will be validated, and what triggers action. The monitoring module (drift thresholds plus Dice
degradation alerts) is the technical backbone of such a plan: defined thresholds lead to a
defined response, whether that is recalibrate, retrain, or roll back.

## 2. HIPAA (US privacy)
- Protected health information must be protected. For research or portfolio work, prefer
  de-identified data. The Safe-Harbor method removes 18 identifier types. Histopathology pixels
  generally carry none, but file metadata can, so strip it.
- PanNuke is already de-identified public research data, so no protected health information is
  processed here.
- Production systems handling such data need business associate agreements, access controls,
  encryption at rest and in transit, and audit logging. These are outside this repo's scope but
  noted for completeness.

## 3. European Union
- EU MDR 2017/745: software with a medical purpose is a medical device and needs CE marking
  through a Notified Body.
- EU AI Act: medical-device AI is generally high-risk, which triggers requirements for risk
  management, data governance, transparency, human oversight, and post-market monitoring. These
  mirror the modules in this project.
- GDPR: lawful basis, data minimisation, and for automated decisions a right to explanation,
  which the explainability module supports.

## 4. Quality and traceability
- IEC 62304 (medical device software lifecycle) and ISO 14971 (risk management) would govern a
  real build, and ISO 13485 covers the quality management system.
- This repo's reproducible configs, versioned checkpoints, tests, and documentation are the kinds
  of artifacts those standards expect, scaled to a portfolio.

## Short version for an interview
I treated the model as one component of a regulated system: data provenance (Datasheet),
intended-use limits (Model Card), bias evaluation (fairness audit) that flagged real subgroup
gaps, human-in-the-loop explainability, and post-market drift monitoring tied to defined
thresholds. That is essentially the skeleton of an FDA GMLP and PCCP story.
