"""
Interactive dashboard for attention visualization
"""

from typing import List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor

from ..core import AttentionVisualizer


class Dashboard:
    """Interactive dashboard for exploring attention patterns"""

    def __init__(self, model=None, processor=None):
        self.model = model
        self.processor = processor
        self.visualizer = None

        if model and processor:
            self.visualizer = AttentionVisualizer(model, processor)

    def load_model(self, model_name: str) -> str:
        """Load a model from HuggingFace"""
        try:
            if "clip" in model_name.lower():
                self.model = CLIPModel.from_pretrained(model_name)
                self.processor = CLIPProcessor.from_pretrained(model_name)
            elif "blip" in model_name.lower():
                self.model = BlipForConditionalGeneration.from_pretrained(model_name)
                self.processor = BlipProcessor.from_pretrained(model_name)
            else:
                return f"❌ Unsupported model: {model_name}"

            self.visualizer = AttentionVisualizer(self.model, self.processor)
            return f"✅ Successfully loaded {model_name}"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def visualize_attention(
        self,
        image,
        text: str,
        visualization_type: str,
        layer_index: int,
        aggregate_heads: bool,
        color_scheme: str,
    ) -> Tuple[Image.Image, str]:
        """Generate attention visualization"""
        if self.visualizer is None:
            return None, "Please load a model first!"

        try:
            # Create visualization
            viz = self.visualizer.visualize(
                image=image,
                text=text,
                layer_indices=[layer_index] if layer_index >= 0 else None,
                visualization_type=visualization_type.lower(),
                cmap=color_scheme,
                aggregate_heads=aggregate_heads,
            )

            # Get statistics
            stats = self.visualizer.get_attention_stats(image, text, layer_index)

            # Format statistics
            stats_text = f"""
            📊 Attention Statistics:
            - Average Entropy: {stats['entropy'].mean():.3f}
            - Attention Concentration: {stats['concentration']:.3f}
            - Top Attended Tokens: {', '.join([f"{token} ({score:.3f})" for token, score in stats['top_tokens'][:3]])}
            """

            # Convert to image
            viz_image = viz.to_image()

            return viz_image, stats_text

        except Exception as e:
            return None, f"Error: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="Transformers Attention Visualizer") as interface:
            gr.Markdown(
                """
            # 🔍 Transformers Attention Visualizer
            
            Explore attention patterns in multi-modal transformer models interactively!
            """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # Model selection
                    gr.Markdown("## Model Configuration")
                    model_dropdown = gr.Dropdown(
                        choices=[
                            "openai/clip-vit-base-patch32",
                            "openai/clip-vit-large-patch14",
                            "Salesforce/blip-image-captioning-base",
                            "Salesforce/blip-image-captioning-large",
                        ],
                        value="openai/clip-vit-base-patch32",
                        label="Select Model",
                    )
                    load_button = gr.Button("Load Model", variant="primary")
                    model_status = gr.Textbox(label="Status", interactive=False)

                    # Input configuration
                    gr.Markdown("## Input Configuration")
                    image_input = gr.Image(label="Upload Image", type="pil")
                    text_input = gr.Textbox(
                        label="Enter Text", placeholder="A photo of a cat", value="A photo of a cat"
                    )

                    # Visualization settings
                    gr.Markdown("## Visualization Settings")
                    viz_type = gr.Radio(
                        choices=["Heatmap", "Flow", "Evolution"],
                        value="Heatmap",
                        label="Visualization Type",
                    )

                    layer_slider = gr.Slider(
                        minimum=-1,
                        maximum=11,
                        value=-1,
                        step=1,
                        label="Layer Index (-1 for last layer)",
                    )

                    aggregate_heads = gr.Checkbox(value=True, label="Average Attention Heads")

                    color_scheme = gr.Dropdown(
                        choices=["Blues", "Viridis", "Plasma", "Inferno", "Magma"],
                        value="Blues",
                        label="Color Scheme",
                    )

                    visualize_button = gr.Button("Generate Visualization", variant="primary")

                with gr.Column(scale=2):
                    # Output
                    gr.Markdown("## Visualization Output")
                    output_image = gr.Image(label="Attention Visualization")
                    output_stats = gr.Textbox(label="Statistics", lines=6)

            # Examples
            gr.Markdown("## Examples")
            gr.Examples(
                examples=[
                    ["examples/cat.jpg", "A fluffy orange cat"],
                    ["examples/dog.jpg", "A happy golden retriever"],
                    ["examples/landscape.jpg", "A beautiful mountain landscape"],
                ],
                inputs=[image_input, text_input],
            )

            # Connect events
            load_button.click(fn=self.load_model, inputs=[model_dropdown], outputs=[model_status])

            visualize_button.click(
                fn=self.visualize_attention,
                inputs=[
                    image_input,
                    text_input,
                    viz_type,
                    layer_slider,
                    aggregate_heads,
                    color_scheme,
                ],
                outputs=[output_image, output_stats],
            )

        return interface


def launch_dashboard(model=None, processor=None, **kwargs):
    """Launch the interactive dashboard"""
    dashboard = Dashboard(model, processor)
    interface = dashboard.create_interface()
    interface.launch(**kwargs)


def main():
    """Entry point for command-line usage"""
    print("🚀 Launching Transformers Attention Visualizer Dashboard...")
    launch_dashboard(share=True)


if __name__ == "__main__":
    main()
