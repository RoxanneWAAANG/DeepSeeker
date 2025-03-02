import os
import glob
import re

def clean_text(text):
    """
    Clean the text by removing extra whitespace.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_length=500, overlap=50):
    """
    Splits text into chunks with a specified maximum word count and overlap.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start+max_length])
        if chunk:
            chunks.append(chunk)
        start += max_length - overlap  # advance with overlap
    return chunks

if __name__ == "__main__":
    # Define input and output folders for preprocessing
    input_folder = "extracted_data"          # Folder with extracted text files
    output_folder = "preprocessed_data"        # Folder to save chunked output

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all text files in the extracted_data folder
    text_files = glob.glob(os.path.join(input_folder, "*.txt"))
    for text_file in text_files:
        with open(text_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)
        base_filename = os.path.splitext(os.path.basename(text_file))[0]
        output_file = os.path.join(output_folder, f"{base_filename}_chunks.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + "\n\n")
        print(f"Processed {text_file} into {len(chunks)} chunks, saved to {output_file}")
