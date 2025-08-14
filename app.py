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


def ocr_extraction(image_path):
    if not image_path:
        return False
    result = image_to_text(image_path)
    return result


def asr_extraction(audio_path):
    if not audio_path:
        return False
    result = audio_to_text(audio_path)
    return result


def check_deepfake(image_path):
    if not image_path:
        return False
    result = xceptionNet_inference(image_path)
    return result


with gr.Blocks(title="AI for Disinformation") as demo:
    with gr.Row():
        with gr.Tab("Claim Verification"):
            gr.Markdown("## API Key")
            api_key_tb = gr.Textbox(label="API Key", type="password")
            with gr.Row():
                save_btn = gr.Button("Save key", variant="primary")
                use_btn = gr.Button("Use key")
            status_md = gr.Markdown()

            gr.Markdown("## Text Input (Chatbot)")
            gr.ChatInterface(
                chat_reply,
                type="messages",
                autofocus=False,
                additional_inputs=[api_key_tb],
            )

            # OCR Section
            gr.Markdown("## OCR Detection (Image to Text)")
            with gr.Row():
                with gr.Column():
                    ocr_image = gr.Image(label="Upload Image", type="filepath")
                    ocr_run_btn = gr.Button("Run OCR", variant="primary")
                with gr.Column():
                    ocr_output_tb = gr.Textbox(label="Extracted Text", lines=2)
                    ocr_ask_btn = gr.Button("Ask AI", variant="primary")
                    ocr_response_tb = gr.Textbox(label="AI Response", lines=5)

            ocr_run_btn.click(
                fn=ocr_extraction, inputs=[ocr_image], outputs=[ocr_output_tb]
            )
            ocr_ask_btn.click(
                fn=chat_reply,
                inputs=[ocr_output_tb, gr.State([]), api_key_tb],
                outputs=[ocr_response_tb],
            )

            # ASR Section
            gr.Markdown("## ASR Detection (Audio to Text)")
            with gr.Row():
                with gr.Column():
                    asr_audio = gr.Audio(label="Upload Audio", type="filepath")
                    asr_run_btn = gr.Button("Run ASR", variant="primary")
                with gr.Column():
                    asr_output_tb = gr.Textbox(label="Extracted Text", lines=2)
                    asr_ask_btn = gr.Button("Ask AI", variant="primary")
                    asr_response_tb = gr.Textbox(label="AI Response", lines=5)

            asr_run_btn.click(
                fn=asr_extraction, inputs=[asr_audio], outputs=[asr_output_tb]
            )
            asr_ask_btn.click(
                fn=chat_reply,
                inputs=[asr_output_tb, gr.State([]), api_key_tb],
                outputs=[asr_response_tb],
            )

        with gr.Tab("Deepfake Detection"):
            gr.Markdown("## Image Input (Deepfake Detection)")
            gr.Markdown("### Run time depends on your hardware")
            image = gr.Image(type="filepath", label="Upload image")
            run_btn = gr.Button("Run", variant="primary")
            result = gr.Checkbox(label="Is AI generated?", interactive=False)
            confidence_md = gr.Markdown()
            run_btn.click(
                fn=check_deepfake, inputs=[image], outputs=[result, confidence_md]
            )

    demo.load(load_key, inputs=None, outputs=[api_key_tb, status_md])
    save_btn.click(save_key, inputs=[api_key_tb], outputs=[status_md])
    use_btn.click(use_key, inputs=[api_key_tb], outputs=[status_md])

if __name__ == "__main__":
    demo.launch(share=True)
