# Deploying the demo to Hugging Face

This publishes the Gradio demo as a Hugging Face Space. Because the trained checkpoint is
187 MB, it lives in a separate model repo and the Space downloads it at startup. That keeps
the Space repo small and is the pattern Hugging Face recommends.

## One command (recommended)

You are already logged in (`hf auth login`). From the repo root:

```
python deploy/deploy_hf.py \
    --model-repo Devisri515/medseg-rai-pannuke \
    --space      Devisri515/medseg-rai \
    --weights    outputs/pannuke_resnet50/best_model.pth \
    --data-root  data/pannuke
```

This creates two public repos under your account:
- model: https://huggingface.co/Devisri515/medseg-rai-pannuke  (holds best_model.pth)
- space: https://huggingface.co/spaces/Devisri515/medseg-rai   (the live demo)

The Space builds for a few minutes, then goes live. Add `--private` to keep both private.

## What the script does
1. Creates the model repo and uploads best_model.pth plus a model card.
2. Assembles the Space in a temp folder: the medseg package, app.py (with your model repo
   baked in), requirements.txt, the Space README, and three real example patches taken from
   your local data.
3. Creates the gradio Space and uploads everything.

## Manual alternative (web UI)
1. huggingface.co, New, Model. Name it medseg-rai-pannuke. Upload best_model.pth and
   deploy/MODEL_README.md renamed to README.md.
2. huggingface.co, New, Space. SDK Gradio. Name it medseg-rai.
3. In the Space repo add: deploy/app.py as app.py (replace __MODEL_REPO__ with your model
   repo id), deploy/requirements.txt, deploy/README.md as README.md (replace __MODEL_REPO__),
   and the whole medseg/ folder. Optionally add a few PNGs under examples/.
4. The Space rebuilds and launches.

## Notes
- If the build says gradio sdk_version 6.16.0 is unavailable, edit the Space README and set
  sdk_version to the latest it offers. Any gradio 5 or 6 works with this app.
- Inference runs on free CPU hardware; one 256 by 256 patch takes a second or two.
- The weights are non-commercial (PanNuke is CC BY-NC-SA 4.0); the model card states this.
