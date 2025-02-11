import gradio as gr


def process_gallery(images):
    return f"Gallery has {len(images)} items."


example_dir = "/weka/home-jensen/scena-image/scene/"
example_path = "/weka/home-jensen/scena-image/scene/zhuzhou.png"

with gr.Blocks() as demo:
    gallery = gr.Gallery(label="Gallery", interactive=True)
    # gallery = gr.Gallery(
    #     value=[example_path, example_path, example_path],
    #     label="Gallery",
    #     interactive=False,
    # )
    examples = gr.Examples(
        examples=[
            [
                [example_path, example_path, example_path],
                [example_path, example_path],
            ],
        ],
        inputs=[gallery],
    )

demo.launch(share=True, allowed_paths=[example_dir])
