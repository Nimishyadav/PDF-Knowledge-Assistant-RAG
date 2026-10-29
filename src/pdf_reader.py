from pypdf import PdfReader


def read_pdf(pdf_path):
    """
    Read a PDF file and return all extracted text.
    """

    # Open the PDF
    reader = PdfReader(pdf_path)

    # Store text from all pages
    text = ""

    # Go through each page
    for page in reader.pages:

        # Extract page text
        page_text = page.extract_text()

        # Add text if page is not empty
        if page_text:
            text += page_text + "\n"

    return text