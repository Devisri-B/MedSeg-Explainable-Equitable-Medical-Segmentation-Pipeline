# Ethical-AI Assessment (IEEE-aligned)

This document maps the project to the IEEE approach to trustworthy AI — the
*Ethically Aligned Design* vision, the **IEEE 7000™** family of standards, and the
**IEEE CertifAIEd™** assessment pillars: **Accountability, Transparency, Algorithmic
Bias, and Privacy**. For each pillar: the concern, what this project does about it,
where in the code, and the residual risk an honest reviewer should still flag.

---

## 1. Accountability
*Who is answerable for the system's behaviour, and can a result be reproduced and traced?*

**What we do**
- Every run is fully specified by a versioned config (`configs/*.yaml`) and a fixed
  seed (`medseg/utils.py::set_seed`); the exact config is snapshotted into each run
  directory (`outputs/<run>/config.yaml`).
- Each model checkpoint stores its config, class names, and validation metrics, so a
  prediction can always be traced back to the model and data that produced it
  (`medseg/train.py`, `medseg/evaluate.py::load_run`).
- A **Model Card** ([MODEL_CARD.md](MODEL_CARD.md)) records intended use and limits.

**Residual risk:** no human-in-the-loop sign-off workflow or audit log of *who*
deployed *which* version; that belongs in the surrounding clinical quality system.

---

## 2. Transparency
*Can a stakeholder understand how an output was reached and how the system was built?*

**What we do**
- **Decision explainability:** Seg-Grad-CAM (`medseg/explain/seg_gradcam.py`) shows
  which input regions drove a class; **uncertainty maps** (softmax entropy and
  MC-dropout, `medseg/explain/uncertainty.py`) show *where the model is unsure*.
- **Process transparency:** the **Datasheet** ([DATASHEET.md](DATASHEET.md)) documents
  data provenance, licensing, and known biases; the code is small and readable by design.
- **Quantification** (`medseg/quantify.py`) turns pixels into named, inspectable
  readouts (counts, area fractions, degradation index) rather than an opaque score.

**Residual risk:** saliency methods are themselves approximate and can mislead;
they support, not replace, expert review.

---

## 3. Algorithmic Bias
*Does the system perform equitably across clinically relevant subgroups?*

**What we do**
- A dedicated **fairness audit** (`medseg/fairness/audit.py`) computes Dice **per
  tissue type** and **per stain-brightness bin**, then reports worst-group performance,
  best–worst gap, worst/best ratio (the "4/5ths rule"), and coefficient of variation —
  and **flags** any subgroup with >20% relative degradation.
- Imbalance is handled at training time via median-frequency class weighting and a
  Dice term that excludes the dominant background class (`medseg/losses.py`).

**Mitigations when a disparity is found:** targeted data collection for the worst
group, group/class reweighting, stain normalisation/augmentation, group-aware
thresholds, and — crucially — **disclosing** the disparity in the Model Card rather
than hiding behind an average.

**Residual risk:** we can only audit subgroups we have labels for; unmeasured
confounders (institution, demographic factors absent from PanNuke) may carry bias.

---

## 4. Privacy
*Are individuals' data protected by design?*

**What we do**
- The project uses a **public, de-identified, consented** research dataset (PanNuke)
  and a fully **synthetic** generator with no patient data — so no PHI is handled in
  the portfolio context.
- Guidance on HIPAA Safe-Harbor de-identification and EU GDPR is documented in
  [REGULATORY.md](REGULATORY.md) for anyone adapting this to real data.

**Residual risk:** histopathology images *can* in principle carry re-identification
risk (e.g., via genomic correlates); production use needs a formal privacy review and,
ideally, data minimisation and access controls.

---

## How this satisfies the role
The posting asks for *"familiarity with ethical AI frameworks and assessment criteria,
such as those from IEEE (Accountability, Transparency, Algorithmic Bias, and Privacy)"*
and *"AI fairness analysis, decision explainability, and rigorous model performance
monitoring."* This project implements a working instance of **each** of those pillars
and ties them to runnable code, not just prose.

## References
- IEEE, *Ethically Aligned Design* (1st ed.).
- IEEE 7000-2021, *Model Process for Addressing Ethical Concerns During System Design*.
- IEEE 7001 (Transparency), 7002 (Data Privacy), 7003 (Algorithmic Bias Considerations).
- IEEE CertifAIEd™ ontological specifications (Accountability, Transparency, Bias, Privacy).
