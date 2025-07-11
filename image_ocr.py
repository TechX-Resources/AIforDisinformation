from PIL import Image
import pytesseract
import os

def ocr_image(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_folder(folder_path, output_folder="ocr_results"):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            extracted_text = ocr_image(image_path)

            if extracted_text:
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(output_folder, base_name + ".txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
    
    print("OCR completed for all images.")

# Example usage
# if __name__ == "__main__":
#     process_folder("images")
