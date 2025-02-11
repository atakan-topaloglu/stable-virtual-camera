import time

import gradio as gr


def fake():
    time.sleep(1)
    return ["Done"] * 2


with gr.Blocks() as demo:
    with gr.Row():
        b = gr.Button()
        t1 = gr.Textbox("Both of these will update...")
        t2 = gr.Textbox("...but only this one will show a loading indicator")

    b.click(fake, None, [t1, t2], show_progress="full")

demo.launch(share=True)
