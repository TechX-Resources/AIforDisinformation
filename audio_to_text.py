import whisper
import os

def transcribe_audio(file_path, output_path="transcription.txt"):
    model = whisper.load_model("base")  # or "small", "medium", "large"
    print(f"Transcribing: {file_path} ...")
    result = model.transcribe(file_path)

    # Save the transcription
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print("Transcription complete. Output saved to:", output_path)

# Example usage
# if __name__ == "__main__":
#     audio_file = "your_audio.mp3"  
#     transcribe_audio(audio_file)