from loaders.pdf_loader import load_pdf_with_extras
import os

pdf_path = "d:\\Chat With Your PDFs\\dummy_test.pdf"
print(f"Testing loader on: {pdf_path}")

try:
    docs = load_pdf_with_extras(pdf_path)
    print(f"SUCCESS: Extracted {len(docs)} document chunks.")
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
except Exception as e:
    print(f"FAILURE: {e}")
