import threading
import time

import gradio as gr


class MyObj:
    def __init__(self):
        self.x = None


obj = MyObj()


def update_obj():
    time.sleep(5)  # simulate background work
    obj.x = "ready"


threading.Thread(target=update_obj, daemon=True).start()


with gr.Blocks() as demo:
    timer = gr.Timer(1)
    text = gr.Textbox("x", label="obj.x")
    btn = gr.Button("Click Me", interactive=False)
    timer.tick(
        lambda: (obj.x, gr.Button(interactive=obj.x is not None)), outputs=[text, btn]
    )

demo.launch(share=True)
