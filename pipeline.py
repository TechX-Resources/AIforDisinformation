from llm import QwenChatbot
from web_search import verify_with_duckduckgo
from PIL import Image
import torch
import torch.nn as nn
from timm import create_model
from torchvision import transforms
from image_ocr import *
from audio_to_text import *
import torchvision.models as models


def fact_check_pipeline(user_input, api_key):
    chatbot = QwenChatbot(api_key=api_key)

    print("\n Summarizing input and generating search prompt...\n")
    search_prompt = chatbot.summarize_prompt(user_input)
    print("Search Prompt:\n", search_prompt, "\n")

    sources = {"DuckDuckGo": verify_with_duckduckgo(search_prompt)}

    combined_evidence = ""
    for source, entries in sources.items():
        combined_evidence += f"\n{source}:\n"
        for item in entries:
            combined_evidence += f"• {item}\n"

    print("Combined Evidence:\n", combined_evidence, "\n")
    print("Evidence collected. Evaluating truthfulness...\n")
    evaluation = chatbot.check_truthiness(combined_evidence, user_input)
    print("Evaluation Result:\n", evaluation)

    return evaluation


def audio_to_text(audio_path, apikey):
    if not audio_path:
        return False
    text = transcribe_audio(audio_path, apikey)
    return text


def image_to_text(image_path):
    if not image_path:
        return False
    result = image_to_text(image_path)
    return result


# Using xceptionNet
def xceptionNet_inference(image_path, model_path="model_weights/xception_deepfake.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_class = 2

    # Create model
    model = create_model("xception", pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_class)

    # Load state dict, strip 'module.' if needed
    state_dict = torch.load(model_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    is_fake = pred == 0
    confidence_text = (
        f"Confidence: {confidence * 100:.2f}% ({'Fake' if is_fake else 'Real'})"
    )
    return is_fake, confidence_text


def resnet_inference(image_path, model_path="model_weights/resnet18_full_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model with weights_only=False (if you trust the source)
    loaded_model = torch.load(model_path, map_location=device, weights_only=False)

    # Check if it's a full model or just state_dict
    if isinstance(loaded_model, dict):
        # It's a state dictionary
        model = models.resnet18(weights=None)  # Fixed deprecated parameter
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes

        # Handle DataParallel wrapper keys
        state_dict = loaded_model
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        model = loaded_model

    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    is_fake = pred == 0
    confidence_text = (
        f"Confidence: {confidence * 100:.2f}% ({'Fake' if is_fake else 'Real'})"
    )
    return is_fake, confidence_text
