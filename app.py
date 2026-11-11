#!/usr/bin/env python3
"""Gradio UI for wan-prompt-forge."""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from prompt_forge import (BASE_URL, DEFAULT_NEGATIVE, MODEL, STYLE_PRESETS,
                         build_graph, keyframe_prompts, plan_shot_story, submit_to_comfy)

OUT = Path("output")
OUT.mkdir(exist_ok=True)

EXAMPLE = (
    "PASSPORT RUSH: an open passport on a glossy airport counter, a traveler sprinting "
    "through a crowded golden-hour terminal, he slams the passport shut, grabs a boarding "
    "pass and runs for the gate. He makes it as the gate closes. Final shot: he sits in his "
    "seat, exhausted, smiling, the window reflecting the runway."
)


def markdown_plan(plan: dict) -> str:
    lines = [f"### {plan['title']}", f"Style anchor: `{plan['style_anchor']}`", ""]
    lines.append("| # | type | frames | positive prompt | start keyframe | end keyframe | seed |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in plan["shots"]:
        end = s.get("end_image_prompt") or "—"
        lines.append(f"| {s['id']} | {s['type']} | {s['frames']} | {s['positive']} | "
                     f"{s['start_image_prompt']} | {end} | {s['seed']} |")
    return "\n".join(lines)


def generate(logline, shots, duration, fps, width, height, style, model, base_url, server, submit):
    if not logline.strip():
        return "Enter a logline.", "", "", ""
    plan = plan_shot_story(logline, int(shots), float(duration), int(fps), style, model, base_url)
    workflow = build_graph(plan, int(width), int(height))
    plan_file = OUT / "shot_plan.json"
    wf_file = OUT / "workflow.json"
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    wf_file.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    status = ""
    if submit:
        try:
            r = submit_to_comfy(workflow, server)
            status = f"Submitted to ComfyUI at {server}, prompt id: {r.get('prompt_id')}"
        except Exception as exc:  # noqa: BLE001 - surface the error to the UI
            status = f"Could not submit to ComfyUI ({server}): {exc}"
    return markdown_plan(plan) + ("\n\n" + status if status else ""), str(plan_file), str(wf_file), keyframe_prompts(plan)


with gr.Blocks(title="Wan Prompt Forge") as demo:
    gr.Markdown("# Wan Prompt Forge\nLogline → chained shot plan → Wan 2.2 I2V ComfyUI workflow. "
                "Each shot is 49 frames (2 s); long videos are chains of shots stitched by keyframes.")
    with gr.Row():
        with gr.Column(scale=2):
            logline = gr.Textbox(label="Logline", value=EXAMPLE, lines=6)
            with gr.Row():
                shots = gr.Slider(1, 8, value=4, step=1, label="Shots")
                duration = gr.Slider(2, 60, value=8, step=1, label="Total duration (s)")
                fps = gr.Number(value=24, label="FPS")
            with gr.Row():
                width = gr.Number(value=1024, label="Width")
                height = gr.Number(value=576, label="Height")
            style = gr.Dropdown(list(STYLE_PRESETS), value="cinematic_golden_hour", label="Style preset")
            with gr.Row():
                base_url = gr.Textbox(label="LLM base URL", value=BASE_URL)
                model = gr.Textbox(label="Model", value=MODEL)
            with gr.Row():
                server = gr.Textbox(label="ComfyUI server", value="http://127.0.0.1:8188")
                submit = gr.Checkbox(value=False, label="Submit graph to ComfyUI automatically")
            btn = gr.Button("Generate shot plan")
        with gr.Column(scale=3):
            plan_md = gr.Markdown()
            keyframes = gr.Textbox(label="Keyframe prompts (paste into Flux/SDXL)", lines=12)
            plan_file = gr.File(label="shot_plan.json")
            wf_file = gr.File(label="workflow.json")
    btn.click(generate, [logline, shots, duration, fps, width, height, style, model, base_url, server, submit],
              [plan_md, plan_file, wf_file, keyframes])
    gr.Markdown(f"Global negative is the fixed tag list: {DEFAULT_NEGATIVE}")

if __name__ == "__main__":
    demo.launch()
