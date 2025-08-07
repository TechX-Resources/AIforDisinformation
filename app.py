import os
import gradio as gr
from pathlib import Path
from pipeline import *

file = Path("api_key.txt")


def get_api_key():
    if file.exists():
        try:
            key = file.read_text(encoding="utf-8").strip()
            return key or None
        except Exception:
            return None
    return None


def load_key():
    key = get_api_key()
    if key:
        return gr.update(value=key), "Loaded API key from file api_key.txt"
    else:
        return (
            gr.update(value=""),
            "No api_key.txt found. Paste your GROQ API key and click save",
        )


def save_key(key: str):
    key = (key or "").strip()
    if not key:
        return "Please enter your key here."
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(key, encoding="utf-8")
    try:
        os.chmod(file, 0o600)
    except Exception:
        pass
    return "API key saved to api_key.txt"


def use_key(input_key: str):
    key = get_api_key()
    if not key:
        msg = save_key(input_key)
        key = (input_key or "").strip()
        if not key:
            return "No api key available"
        status = msg
    else:
        status = "Using key from api_key.txt"
    return status


def chat_reply(message, history, key_from_ui):
    key = get_api_key() or (key_from_ui or "").strip()
    if not key:
        return "No key available."
    try:
        return fact_check_pipeline(message, key)
    except Exception as e:
        return f"Error: {e}"


def check_deepfake(image_path):
    if not image_path:
        return False
    result = deepfake_detection(image_path)
    return result


with gr.Blocks(title="AI for Disinformation") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Image Input (Deepfake Detection)")
            gr.Markdown("### Run time depends on your hardware")
            image = gr.Image(type="filepath", label="Upload image")
            run_btn = gr.Button("Run")
            result = gr.Checkbox(label="Is AI generated?", interactive=False)
            confidence_md = gr.Markdown()
            run_btn.click(
                fn=check_deepfake, inputs=[image], outputs=[result, confidence_md]
            )

        with gr.Column():
            gr.Markdown("## API Key")
            api_key_tb = gr.Textbox(
                label="API Key", type="password", placeholder="sk-..."
            )
            with gr.Row():
                save_btn = gr.Button("Save key")
                use_btn = gr.Button("Use key")
            status_md = gr.Markdown()

            gr.Markdown("## Text Input (Chatbot)")
            gr.ChatInterface(
                chat_reply,
                type="messages",
                autofocus=False,
                additional_inputs=[api_key_tb],
            )

    demo.load(load_key, inputs=None, outputs=[api_key_tb, status_md])
    save_btn.click(save_key, inputs=[api_key_tb], outputs=[status_md])
    use_btn.click(use_key, inputs=[api_key_tb], outputs=[status_md])

if __name__ == "__main__":
    demo.launch()
