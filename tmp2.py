import gradio as gr
import tyro


def main(server_port: int = 7860):
    example_dir = "/weka/home-jensen/scena-image/nonsquare/"

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("""
# Scena gradio demo

## Workflow

1. Upload an image.
2. Choose camera trajecotry and #frames, click "Render".
3. Three videos will be generated: preprocessed input, intermediate output, and final output.

> For a 80-frame video, intermediate output takes 20s, final output takes 2~3m.
> Our model currently doesn't work well with human and animal images.
                        """)
        with gr.Row():
            with gr.Column():
                # input_imgs = gr.Image(type="numpy", label="Input")
                input_imgs = gr.Gallery(interactive=True, label="Input")
                gr.Examples(
                    examples=[
                        [
                            [
                                "/weka/home-jensen/scena-image/nonsquare/photo-1735252723552-138dc3fb6f14.png"
                            ]
                        ]
                    ],
                    inputs=[input_imgs],
                    label="Examples",
                )
                input_imgs.upload(fn=lambda v, n: (v, v, n + 1), outputs=[input_imgs])
                preprocess_btn = gr.Button("Preprocess")

    demo.launch(share=True, allowed_paths=[example_dir], server_port=server_port)


if __name__ == "__main__":
    tyro.cli(main)
