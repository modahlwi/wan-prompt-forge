# wan-prompt-forge

Write prompts for **long Wan 2.2 I2V videos** in ComfyUI: give it a logline, it plans a chain of shots, writes a Wan-safe prompt for each one, and emits a ready-to-paste (or directly submitted) ComfyUI graph.

The core idea: Wan 2.2 does not produce a 60-second story in one go. It produces **49-frame shots** (2 s @ 24 fps, length must be `4n+1`). Long videos are built by chaining shots where the **end keyframe of shot N becomes the start keyframe of shot N+1**. This app is the planner for that chain.

## What you get

1. `shot_plan.json` — title, style anchor, global negative, and per-shot:
   - `type`: `first_last` (transition, uses `WanFirstLastFrameToVideo`) or `image_to_video` (open-ended)
   - `frames`, `start_image_prompt`, `end_image_prompt`, `positive`, `seed`, `notes`
2. `workflow.json` — the full ComfyUI graph: model chain (high/low noise + Lightning LoRAs), one `CLIPTextEncode` per shot, two `KSamplerAdvanced` nodes per shot (4-step Lightning split at step 2), `VAEDecode` → `CreateVideo` → `SaveVideo`.
3. A copy-paste block of keyframe prompts to feed an image model (Flux/SDXL) to generate the start/end frames.

## Quickstart

```bash
pip install -r requirements.txt
ollama serve && ollama pull qwen2.5:14b   # or any OpenAI-compatible endpoint
python app.py                             # Gradio UI at http://localhost:7860
```

CLI:

```bash
python prompt_forge.py "a courier races through an airport to catch a flight" \
  --shots 4 --duration 8 --out workflow.json
python prompt_forge.py "..." --shots 3 --submit --server http://127.0.0.1:8188
```

## Model choice

Any OpenAI-compatible chat endpoint works (Ollama, vLLM, LM Studio, OpenRouter, OpenAI). Use something that follows structured-output instructions well — Qwen2.5-14B, Mistral-Small, DeepSeek are good cheap picks. Set `BASE_URL` and `MODEL` in `prompt_forge.py` or via `PROMPT_FORGE_BASE_URL` / `PROMPT_FORGE_MODEL`.

## Wiring the LLM inside ComfyUI instead

Two options:

- **Recommended (no custom nodes):** the app POSTs the graph straight to ComfyUI's `/prompt` API (`--submit`). The LLM runs in the app, ComfyUI only renders.
- **In-graph:** install an LLM node pack such as [`comfyui_LLM_party`](https://github.com/heshengtao/comfyui_LLM_party) or [`ComfyUI-IF_AI_tools`](https://github.com/if-ai/ComfyUI-IF_AI_tools), set its system prompt to `planner_system.txt`, and wire its text output into the `text` input of each `CLIPTextEncode` node. Check the exact class name in ComfyUI's node search, since it differs between packs.

## Notes on the generated graph

- `length` is always `4n+1`; the app clamps and repairs whatever the LLM proposes.
- With the 4-step Lightning LoRAs: `cfg = 1.0`, `euler`/`simple`, and the sampler split at step 2 (high-noise model 0→2, low-noise model 2→4). Consider `shift` 3.0–4.0 on `ModelSamplingSD3` for the distilled pair; 5.0 is for the full non-distilled run.
- The negative prompt stays a flat tag list — that is the right shape for Wan. Do not put narrative in it.
- Shot stitching: extract the last frame with `ffmpeg -i shotN.mp4 -vf "select=eq(n\,48)" -vvs 1 shotN_end.png` and use it as the next shot's start image.

## Files

```
app.py                 Gradio UI
prompt_forge.py        planner + graph builder + ComfyUI submitter
planner_system.txt     the shot-planner system prompt (edit this to change the voice)
workflows/            reference graphs, including the 2-shot passport example
```
