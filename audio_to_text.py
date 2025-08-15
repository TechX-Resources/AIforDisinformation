import whisper
import os
from groq import Groq


def transcribe_audio(file_path, apikey):
    client = Groq(api_key=apikey)

    with open(file_path, "rb") as audio_file:
        print(f"Transcribing: {file_path} ...")
        result = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), audio_file.read()),
            model="whisper-large-v3",
            response_format="text",
        )
    return result


def process_audio_folder(folder_path, output_folder="transcriptions"):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".mp3", ".wav", ".m4a")):
            file_path = os.path.join(folder_path, filename)
            transcript = transcribe_audio(file_path)

            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_folder, base_name + ".txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcript)

    print("All audio files transcribed.")


# Example usage
if __name__ == "__main__":
    process_audio_folder("audios")
