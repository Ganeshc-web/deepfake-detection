"""
Gradio web application for deepfake detection.

Uses a free 3-model Hugging Face ensemble with TTA — no training or API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from PIL import Image

from src.predict import generate_explanation, load_detector, predict_video

# ---------------------------------------------------------------------------
# Global detector (downloads model on first run)
# ---------------------------------------------------------------------------
DETECTOR = None
MODEL_ERROR: str | None = None

try:
    DETECTOR = load_detector()
except Exception as exc:
    MODEL_ERROR = str(exc)


DARK_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#0f0f0f",
    body_background_fill_dark="#0f0f0f",
    block_background_fill="#1a1a1a",
    block_background_fill_dark="#1a1a1a",
    block_border_width="1px",
    block_title_text_color="#e0e0e0",
    body_text_color="#e0e0e0",
    body_text_color_dark="#e0e0e0",
)


def _check_model_loaded() -> str | None:
    if DETECTOR is None:
        return MODEL_ERROR or "Model failed to load."
    return None


def analyze_image(image: Image.Image | None) -> tuple[str, Image.Image | None]:
    """Analyze an uploaded image and return prediction + probability chart."""
    error = _check_model_loaded()
    if error:
        return f"Error: {error}", None

    if image is None:
        return "Please upload an image first.", None

    try:
        details = DETECTOR.predict_with_details(image)
        breakdown = " | ".join(
            f"{score.name}: {score.fake_prob * 100:.0f}% fake"
            for score in details.model_scores
        )
        result_text = (
            f"**{details.label}**\n\n"
            f"Confidence: **{details.confidence:.1f}%**\n\n"
            f"Model scores: {breakdown}"
        )
        chart = generate_explanation(DETECTOR, image, result=details)
        return result_text, chart
    except Exception as exc:
        return f"Error analyzing image: {exc}", None


def analyze_video(video_path: str | None) -> str:
    """Analyze an uploaded video with frame sampling."""
    error = _check_model_loaded()
    if error:
        return f"Error: {error}"

    if video_path is None:
        return "Please upload a video first."

    try:
        label, confidence = predict_video(DETECTOR, video_path)
        return f"**{label}**\n\nAverage Confidence: **{confidence:.1f}%**"
    except FileNotFoundError as exc:
        return f"Error: {exc}"
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error analyzing video: {exc}"


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(theme=DARK_THEME, title="Deepfake Detector") as demo:
        gr.Markdown(
            """
            # Deepfake Detector
            ### Upload an image or video to check if it's real or AI-generated
            """
        )

        if MODEL_ERROR:
            gr.Markdown(f"> **Warning:** {MODEL_ERROR}")
        elif DETECTOR is not None:
            gr.Markdown(
                f"> **{DETECTOR.display_name}** — Hugging Face, free, no API key"
            )

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Upload Image")
                        image_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column():
                        image_result = gr.Markdown(label="Result")
                        chart_output = gr.Image(
                            type="pil",
                            label="Ensemble + per-model scores",
                        )

                image_btn.click(
                    fn=analyze_image,
                    inputs=[image_input],
                    outputs=[image_result, chart_output],
                )

            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video")
                        video_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column():
                        video_result = gr.Markdown(label="Result")

                video_btn.click(
                    fn=analyze_video,
                    inputs=[video_input],
                    outputs=[video_result],
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        share=False,
    )
