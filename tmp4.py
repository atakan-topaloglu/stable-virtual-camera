import gradio as gr

example_dir = "/weka/home-jensen/scena-image/scene/"
img1 = "/weka/home-jensen/scena-image/scene/ComfyUI_temp_lvxrr_00039_.png"
img2 = "/weka/home-jensen/scena-image/scene/IMG_1619.jpg"
img3 = "/weka/home-jensen/scena-image/scene/image (16).png"

# with gr.Blocks() as demo:
#     gallery = gr.Gallery(label="Gallery", interactive=False)
#     dataset = gr.Dataset(components=[gallery], samples=[[[example_path]]])
#
# demo.launch(share=True, allowed_paths=[example_dir])
#
#
with gr.Blocks() as demo:
    gallery = gr.Gallery(label="Gallery", interactive=True, format)
    ex = gr.Examples(examples=[[[img1, img2]], [[img3]]], inputs=[gallery])
    demo.launch(share=True, allowed_paths=[example_dir])

# with gr.Blocks() as demo:
#     gallery = gr.Image(type="numpy", label="Input")
#     ex = gr.Examples(examples=[img1, img2, img3], inputs=[gallery])
#     demo.launch(share=True, allowed_paths=[example_dir])
