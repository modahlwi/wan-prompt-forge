#!/usr/bin/env python3
"""prompt_forge - turn a logline into a Wan 2.2 shot plan and a ComfyUI graph."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from openai import OpenAI

BASE_URL = os.getenv("PROMPT_FORGE_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("PROMPT_FORGE_MODEL", "qwen2.5:14b")
SYSTEM_PATH = Path(__file__).with_name("planner_system.txt")

STYLE_PRESETS = {
    "cinematic_golden_hour": (
        "Cinematic, 35mm film, shallow depth of field, golden-hour backlight, warm amber "
        "terminal lighting, handheld motion blur, film grain"
    ),
    "documentary_handheld": (
        "Documentary, handheld camera, natural light, candid framing, subtle grain, realism"
    ),
    "neo_noir": (
        "Neo-noir, low-key lighting, wet reflective streets, deep shadows, teal and amber "
        "palette, slow deliberate camera moves"
    ),
    "product_ad_clean": (
        "Commercial product film, glossy studio lighting, clean negative space, crisp focus, "
        "slow smooth camera moves"
    ),
    "anime_cell_shaded": (
        "Anime cel shading, bold linework, saturated colour, dynamic action lines, expressive "
        "camera motion"
    ),
}

DEFAULT_NEGATIVE = (
    "text, watermark, subtitles, logo, low quality, blurry, jpeg artifacts, distorted face, "
    "extra fingers, deformed limbs, static frame, frozen image, cartoon, oversaturated"
)

MODEL_CHAIN = {
    "1": {"class_type": "UNETLoader", "inputs": {
        "unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["1", 0],
        "lora_name": "Wan2.2-I2V-A14B-4steps-Lightning/high_noise_model.safetensors",
        "strength_model": 1.0}},
    "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": 3.0}},
    "4": {"class_type": "UNETLoader", "inputs": {
        "unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
    "5": {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["4", 0],
        "lora_name": "Wan2.2-I2V-A14B-4steps-Lightning/low_noise_model.safetensors",
        "strength_model": 1.0}},
    "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["5", 0], "shift": 3.0}},
    "7": {"class_type": "CLIPLoader", "inputs": {
        "clip_name": "umt5_xxl_fp8.safetensors", "type": "wan", "device": "default"}},
    "8": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
}


def frames_4n1(n: int) -> int:
    """Wan requires length = 4n + 1."""
    n = int(n)
    v = (n - 1) // 4 * 4 + 1
    return max(13, min(125, v))


def frames_for(duration_s: float, shots: int, fps: int) -> list[int]:
    per = max(1, int(round(duration_s / max(1, shots) * fps)))
    return [frames_4n1(per)] * max(1, shots)


def plan_shot_story(logline: str, shots: int, duration_s: float, fps: int,
                    style_key: str = "cinematic_golden_hour",
                    model: str = MODEL, base_url: str = BASE_URL) -> dict:
    frame_counts = frames_for(duration_s, shots, fps)
    anchor = STYLE_PRESETS.get(style_key, STYLE_PRESETS["cinematic_golden_hour"])
    client = OpenAI(base_url=base_url, api_key=os.getenv("PROMPT_FORGE_API_KEY", "not-needed"))
    user = (
        f"Logline: {logline}\n"
        f"Plan {shots} shots, each {', '.join(str(f) for f in frame_counts)} frames. "
        f"Style anchor to reuse verbatim: {anchor}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    plan = json.loads(resp.content if hasattr(resp, "content") else resp.choices[0].message.content)
    return validate_plan(plan, frame_counts, anchor)


def validate_plan(plan: dict, frame_counts: list[int], anchor: str) -> dict:
    plan.setdefault("title", "untitled")
    plan.setdefault("style_anchor", anchor)
    plan.setdefault("global_negative", DEFAULT_NEGATIVE)
    shots = plan.get("shots") or []
    if not shots:
        shots = [{"id": 1, "type": "image_to_video", "positive": plan["title"]}]
    clean = []
    for i, shot in enumerate(shots):
        shot.setdefault("id", i + 1)
        if shot.get("type") not in ("first_last", "image_to_video"):
            shot["type"] = "first_last" if shot.get("end_image_prompt") else "image_to_video"
        shot["frames"] = frame_counts[i] if i < len(frame_counts) else frames_4n1(shot.get("frames", 49))
        shot.setdefault("start_image_prompt", shot.get("positive", "")[:120])
        shot.setdefault("end_image_prompt", None)
        pos = shot.get("positive", "").strip()
        if plan["style_anchor"] not in pos:
            pos = f"{pos} {plan['style_anchor']}".strip()
        shot["positive"] = pos
        shot.setdefault("seed", 1000 + i * 17)
        shot.setdefault("notes", "")
        clean.append(shot)
    plan["shots"] = clean
    return plan


def build_graph(plan: dict, width: int = 1024, height: int = 576) -> dict:
    nodes = dict(MODEL_CHAIN)
    nodes["13"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"clip": ["7", 0], "text": plan["global_negative"]}}
    nid = 14
    for shot in plan["shots"]:
        pos = nid
        nodes[str(pos)] = {"class_type": "CLIPTextEncode",
                          "inputs": {"clip": ["7", 0], "text": shot["positive"]}}
        nid += 1
        start = nid
        nodes[str(start)] = {"class_type": "LoadImage",
                            "inputs": {"image": shot["start_image_file"] if shot.get("start_image_file")
                                      else f"keyframe_{plan['title']}_s{shot['id']}_start.png"}}
        nid += 1
        if shot["type"] == "first_last":
            end = nid
            nodes[str(end)] = {"class_type": "LoadImage",
                              "inputs": {"image": shot.get("end_image_file")
                                        or f"keyframe_{plan['title']}_s{shot['id']}_end.png"}}
            nid += 1
            cond = nid
            nodes[str(cond)] = {"class_type": "WanFirstLastFrameToVideo", "inputs": {
                "positive": [str(pos), 0], "negative": ["13", 0], "vae": ["8", 0],
                "width": width, "height": height, "length": shot["frames"], "batch_size": 1,
                "start_image": [str(start), 0], "end_image": [str(end), 0]}}
            nid += 1
        else:
            cond = nid
            nodes[str(cond)] = {"class_type": "WanImageToVideo", "inputs": {
                "positive": [str(pos), 0], "negative": ["13", 0], "vae": ["8", 0],
                "width": width, "height": height, "length": shot["frames"], "batch_size": 1,
                "start_image": [str(start), 0]}}
            nid += 1
        s1 = nid
        nodes[str(s1)] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["3", 0], "positive": [str(cond), 0], "negative": [str(cond), 1],
            "latent_image": [str(cond), 2], "add_noise": "enable", "noise_seed": shot["seed"],
            "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable"}}
        nid += 1
        s2 = nid
        nodes[str(s2)] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["6", 0], "positive": [str(cond), 0], "negative": [str(cond), 1],
            "latent_image": [str(s1), 0], "add_noise": "disable", "noise_seed": shot["seed"],
            "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 2, "end_at_step": 4, "return_with_leftover_noise": "disable"}}
        nid += 1
        dec = nid
        nodes[str(dec)] = {"class_type": "VAEDecode",
                          "inputs": {"samples": [str(s2), 0], "vae": ["8", 0]}}
        nid += 1
        vid = nid
        nodes[str(vid)] = {"class_type": "CreateVideo",
                           "inputs": {"images": [str(dec), 0], "fps": 24}}
        nid += 1
        save = nid
        nodes[str(save)] = {"class_type": "SaveVideo", "inputs": {
            "video": [str(vid), 0],
            "filename_prefix": f"{str(plan['title']).lower().replace(' ', '_')}_shot{shot['id']}",
            "format": "mp4", "codec": "h264"}}
        nid += 1
    return nodes


def keyframe_prompts(plan: dict) -> str:
    out = [f"# Keyframes for {plan['title']} - feed these to your image model (Flux/SDXL)"]
    for shot in plan["shots"]:
        out.append(f"## shot {shot['id']} - start frame")
        out.append(f"{shot['start_image_prompt']} {plan['style_anchor']}")
        if shot["type"] == "first_last" and shot.get("end_image_prompt"):
            out.append(f"## shot {shot['id']} - end frame")
            out.append(f"{shot['end_image_prompt']} {plan['style_anchor']}")
        out.append("")
    return "\n".join(out)


def submit_to_comfy(workflow: dict, server: str = "http://127.0.0.1:8188") -> dict:
    req = urllib.request.Request(
        f"{server}/prompt",
        data=json.dumps({"prompt": workflow}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Wan 2.2 long-video prompt planner")
    ap.add_argument("logline")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--duration", type=float, default=8.0, help="total seconds")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--style", default="cinematic_golden_hour", choices=list(STYLE_PRESETS))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--out", default="workflow.json")
    ap.add_argument("--plan-out", default="shot_plan.json")
    ap.add_argument("--submit", action="store_true", help="send the graph to ComfyUI /prompt")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    plan = plan_shot_story(args.logline, args.shots, args.duration, args.fps, args.style,
                          args.model, args.base_url)
    workflow = build_graph(plan, args.width, args.height)
    Path(args.plan_out).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    Path(args.out).write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    print(keyframe_prompts(plan))
    print(f"wrote {args.plan_out} and {args.out} ({len(plan['shots'])} shots)")
    if args.submit:
        print(json.dumps(submit_to_comfy(workflow, args.server), indent=2)[:600])


if __name__ == "__main__":
    main()
