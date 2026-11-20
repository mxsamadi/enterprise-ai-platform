from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("documents/sample.pdf")

reader = PdfReader(pdf_path)

print(f"Number of pages: {len(reader.pages)}")

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    print(f"\n--- Page {page_number} ---")
    print(text[:500])
