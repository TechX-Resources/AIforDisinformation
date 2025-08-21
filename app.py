import os
import gradio as gr
from pathlib import Path
import pipeline as pp

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
        return pp.fact_check_pipeline(message, key)
    except Exception as e:
        return f"Error: {e}"


def check_deepfake(image_path, model):
    if not image_path:
        return False
    if model == "XceptionNet":
        result = pp.xceptionNet_inference(image_path)
    elif model == "ResNet":
        result = pp.resnet_inference(image_path)
    return result


with gr.Blocks(
    title="🛡️ AI for Disinformation Detection", theme=gr.themes.Soft()
) as demo:
    # Header with description
    gr.Markdown(
        """
    # 🛡️ AI-Powered Disinformation Detection Suite
    
    **Detect and verify misinformation using advanced AI tools**
    
    This comprehensive platform helps you identify deepfakes, extract text from images/audio, and verify claims using AI assistance.
    """
    )

    # API Key Configuration Section (moved to top for better UX)
    with gr.Accordion("🔑 API Configuration", open=False):
        gr.Markdown("**Configure your API key to enable AI-powered features**")
        with gr.Row():
            api_key_tb = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="Enter your API key here...",
                scale=3,
            )
            with gr.Column(scale=1):
                save_btn = gr.Button("💾 Save", variant="primary", size="sm")
                use_btn = gr.Button("✅ Use", variant="secondary", size="sm")
        status_md = gr.Markdown()

    # Main Content Tabs
    with gr.Tabs():
        # Chat Interface Tab
        with gr.Tab("💬 AI Chat Assistant"):
            gr.Markdown(
                """
            ## 🤖 Interactive Claim Verification
            
            Chat with our AI assistant to verify claims, check facts, and get detailed analysis of potentially misleading information.
            
            **How to use:**
            1. Make sure your API key is configured above
            2. Type your question or paste suspicious content
            3. Get instant AI-powered analysis
            """
            )

            gr.ChatInterface(
                chat_reply,
                type="messages",
                autofocus=True,
                additional_inputs=[api_key_tb],
                title="Fact-Checking Assistant",
                description="Ask me to verify claims, check facts, or analyze suspicious content",
            )

        # Media Analysis Tab
        with gr.Tab("📱 Media Analysis"):
            gr.Markdown(
                """
            ## 🔍 Extract and Analyze Text from Media
            
            Upload images or audio files to extract text content and get AI analysis for potential misinformation.
            """
            )

            with gr.Row():
                # OCR Section
                with gr.Column():
                    gr.Markdown("### 📷 Image Text Extraction (OCR)")
                    gr.Markdown("*Extract text from screenshots, memes, or documents*")

                    ocr_image = gr.Image(label="Upload Image", type="filepath")
                    ocr_run_btn = gr.Button("🔤 Extract Text", variant="primary")
                    ocr_output_tb = gr.Textbox(
                        label="📝 Extracted Text",
                        lines=3,
                        placeholder="Extracted text will appear here...",
                    )
                    ocr_ask_btn = gr.Button("🤔 Analyze with AI", variant="secondary")
                    ocr_response_tb = gr.Textbox(
                        label="🧠 AI Analysis",
                        lines=4,
                        placeholder="AI analysis will appear here...",
                    )

                # ASR Section
                with gr.Column():
                    gr.Markdown("### 🎵 Audio Text Extraction (Speech-to-Text)")
                    gr.Markdown("*Convert speech from audio/video files to text*")

                    asr_audio = gr.Audio(label="Upload Audio", type="filepath")
                    asr_run_btn = gr.Button("🎙️ Transcribe Audio", variant="primary")
                    asr_output_tb = gr.Textbox(
                        label="📝 Transcribed Text",
                        lines=3,
                        placeholder="Transcribed text will appear here...",
                    )
                    asr_ask_btn = gr.Button("🤔 Analyze with AI", variant="secondary")
                    asr_response_tb = gr.Textbox(
                        label="🧠 AI Analysis",
                        lines=4,
                        placeholder="AI analysis will appear here...",
                    )

            # Event handlers for OCR
            ocr_run_btn.click(
                fn=pp.image_to_text, inputs=[ocr_image], outputs=[ocr_output_tb]
            )
            ocr_ask_btn.click(
                fn=chat_reply,
                inputs=[ocr_output_tb, gr.State([]), api_key_tb],
                outputs=[ocr_response_tb],
            )

            # Event handlers for ASR
            asr_run_btn.click(
                fn=pp.audio_to_text,
                inputs=[asr_audio, api_key_tb],
                outputs=[asr_output_tb],
            )
            asr_ask_btn.click(
                fn=chat_reply,
                inputs=[asr_output_tb, gr.State([]), api_key_tb],
                outputs=[asr_response_tb],
            )

        # Deepfake Detection Tab
        with gr.Tab("🎭 Deepfake Detection"):
            gr.Markdown(
                """
            ## 🕵️ AI-Generated Image Detection
            
            Upload an image to detect if it was generated by AI or is a deepfake. Our advanced models analyze visual patterns to identify artificial content.
            
            **Supported formats:** JPG, PNG, WebP
            **Processing time:** Depends on your hardware performance
            """
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🖼️ Image Upload")
                    model = gr.Dropdown(
                        choices=["ResNet", "XceptionNet"],
                        value="ResNet",
                        label="🧠 Detection Model",
                        info="Choose the AI model for detection",
                    )
                    image = gr.Image(
                        type="filepath", label="📸 Upload Image", height=300
                    )
                    run_btn = gr.Button(
                        "🔍 Analyze Image", variant="primary", size="lg"
                    )

                with gr.Column():
                    gr.Markdown("### 📊 Detection Results")
                    result = gr.Checkbox(
                        label="🚨 AI Generated Content Detected", interactive=False
                    )
                    confidence_md = gr.Markdown()

                    gr.Markdown(
                        """
                    **Understanding Results:**
                    - ✅ **Not AI Generated**: Likely authentic image
                    - ⚠️ **AI Generated**: Likely created by AI/deepfake
                    - Higher confidence scores indicate more certainty
                    """
                    )

            run_btn.click(
                fn=check_deepfake,
                inputs=[image, model],
                outputs=[result, confidence_md],
            )

        # Help & Information Tab
        with gr.Tab("❓ Help & Info"):
            gr.Markdown(
                """
            ## 🛟 How to Use This Platform
            
            ### 🔑 Getting Started
            1. **Configure API Key**: Click on "API Configuration" at the top and enter your API key
            2. **Choose Your Tool**: Select the appropriate tab based on what you want to analyze
            3. **Upload or Input**: Provide the content you want to check
            4. **Get Results**: Review the analysis and take appropriate action
            
            ### 🔧 Available Features
            
            #### 💬 **AI Chat Assistant**
            - Verify claims and statements
            - Get detailed fact-checking analysis
            - Ask questions about suspicious content
            - Real-time conversation with AI
            
            #### 📱 **Media Analysis**
            - **Image OCR**: Extract text from images, screenshots, memes
            - **Audio Transcription**: Convert speech to text from audio files
            - **AI Analysis**: Get intelligent analysis of extracted content
            
            #### 🎭 **Deepfake Detection**
            - Detect AI-generated images
            - Identify potential deepfakes
            - Multiple detection models available
            - Confidence scoring system
            
            ### 💡 **Tips for Best Results**
            - Use high-quality images for better OCR accuracy
            - Ensure clear audio for better transcription
            - Try different models for deepfake detection if unsure
            - Ask specific questions to the AI assistant for detailed analysis
            
            ### 🔒 **Privacy & Security**
            - Your API key is stored locally in your session
            - Uploaded files are processed securely
            - No data is permanently stored on our servers
            """
            )

    # Event handlers for API key management
    demo.load(load_key, inputs=None, outputs=[api_key_tb, status_md])
    save_btn.click(save_key, inputs=[api_key_tb], outputs=[status_md])
    use_btn.click(use_key, inputs=[api_key_tb], outputs=[status_md])

if __name__ == "__main__":
    demo.launch(share=True)
