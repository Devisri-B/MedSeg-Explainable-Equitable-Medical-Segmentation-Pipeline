# Ethical-AI Assessment (IEEE-aligned)

This document maps the project to the IEEE approach to trustworthy AI: the Ethically Aligned
Design vision, the IEEE 7000 family of standards, and the IEEE CertifAIEd assessment pillars
of Accountability, Transparency, Algorithmic Bias, and Privacy. For each pillar it states the
concern, what the project does, where in the code, and the residual risk a careful reviewer
should still flag.

## 1. Accountability
Who is answerable for the system's behaviour, and can a result be reproduced and traced?

What the project does:
- Every run is fully specified by a versioned config (`configs/*.yaml`) and a fixed seed
  (`medseg/utils.py`, set_seed). The exact config is saved into each run directory
  (`outputs/<run>/config.yaml`).
- Each checkpoint stores its config, class names, and validation metrics, so a prediction can
  be traced back to the model and data that produced it (`medseg/train.py`,
  `medseg/evaluate.py`, load_run).
- A Model Card records intended use and limits.

Residual risk: there is no human sign-off workflow or audit log of who deployed which version.
That belongs in the surrounding clinical quality system.

## 2. Transparency
Can a stakeholder understand how an output was reached and how the system was built?

What the project does:
- Decision explainability through Seg-Grad-CAM (`medseg/explain/seg_gradcam.py`), which shows
  which input regions drove a class, and uncertainty maps (softmax entropy and MC-dropout,
  `medseg/explain/uncertainty.py`), which show where the model is unsure.
- Process transparency through the Datasheet, which documents data provenance, licensing, and
  known biases, and through code that is small and readable.
- Quantification (`medseg/quantify.py`) turns pixels into named, inspectable readouts (counts,
  area fractions, degradation index) rather than a single opaque score.

Residual risk: saliency methods are themselves approximate and can mislead. They support
expert review, they do not replace it.

## 3. Algorithmic Bias
Does the system perform fairly across clinically relevant subgroups?

What the project does:
- A fairness audit (`medseg/fairness/audit.py`) computes Dice per tissue type and per
  stain-brightness bin, then reports worst-group performance, the best-minus-worst gap, the
  worst-over-best ratio (the four-fifths rule), and the coefficient of variation, and flags any
  group with more than 20 percent relative degradation.
- Imbalance is handled at training time with square-root class weighting clipped at 10 and a
  Focal-Tversky loss that focuses on rare, hard classes (`medseg/losses.py`).

What the audit found on the final model (PanNuke fold 3):
- By tissue: the best group is Testis at 0.714 and the worst is Uterus at 0.396. The gap is
  0.318 and the worst-over-best ratio is 0.56, which fails the four-fifths rule, so the audit
  flags it. Some groups have few test images, so their numbers are noisy.
- By stain brightness: bright 0.651, medium 0.599, dark 0.454. The model is clearly weaker on
  dark slides, which is a stain and scanner effect rather than a property of the tissue.

These disparities are reported in the Model Card and Results, not hidden behind an average.
The next steps they point to are more labelled data for the weak tissues, stronger stain
normalisation and augmentation, and group-aware thresholds where appropriate.

Residual risk: the audit can only cover subgroups that have labels. Unmeasured factors, such as
institution or patient demographics that PanNuke does not record, may still carry bias.

## 4. Privacy
Are individuals' data protected by design?

What the project does:
- It uses a public, de-identified, consented research dataset (PanNuke) and a fully synthetic
  generator with no patient data, so no protected health information is handled in the
  portfolio context.
- Guidance on HIPAA Safe-Harbor de-identification and EU GDPR is in [REGULATORY.md](REGULATORY.md)
  for anyone adapting this to real data.

Residual risk: histopathology images can in principle carry re-identification risk through
genomic correlates. Production use needs a formal privacy review, data minimisation, and access
controls.

## How this satisfies the role
The posting asks for familiarity with ethical AI frameworks such as IEEE (Accountability,
Transparency, Algorithmic Bias, and Privacy), and for AI fairness analysis, decision
explainability, and rigorous performance monitoring. This project implements a working instance
of each pillar and ties it to runnable code and measured results, not just prose.

## References
- IEEE, Ethically Aligned Design, first edition.
- IEEE 7000-2021, Model Process for Addressing Ethical Concerns During System Design.
- IEEE 7001 (Transparency), 7002 (Data Privacy), 7003 (Algorithmic Bias Considerations).
- IEEE CertifAIEd ontological specifications (Accountability, Transparency, Bias, Privacy).
