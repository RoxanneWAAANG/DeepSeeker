import os
import glob
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyPDF2.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return text


if __name__ == "__main__":
    # Define input and output folders
    raw_data_folder = "raw_data"           # Folder containing your PDFs
    output_folder = "extracted_data"         # Folder to save extracted text files

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all PDF files in the raw_data folder
    pdf_files = glob.glob(os.path.join(raw_data_folder, "*.pdf"))
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        base_filename = os.path.splitext(os.path.basename(pdf_file))[0]
        output_file = os.path.join(output_folder, f"{base_filename}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved extracted text from {pdf_file} to {output_file}")
