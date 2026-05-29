from fastapi import FastAPI
import gradio as gr
from finance_agent import create_ui

app = FastAPI()

# Create the Gradio UI
demo = create_ui()

# Mount Gradio app to FastAPI at the root path
app = gr.mount_gradio_app(app, demo, path="/")
