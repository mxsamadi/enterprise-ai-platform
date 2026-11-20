from reportlab.pdfgen import canvas

c = canvas.Canvas("documents/sample.pdf")

c.drawString(50, 750, "Enterprise AI Document Intelligence Platform")
c.drawString(50, 720, "Machine learning helps computers learn from data.")
c.drawString(50, 690, "RAG combines information retrieval with language models.")
c.drawString(50, 660, "Embeddings represent text as numerical vectors.")
c.drawString(50, 630, "Vector databases help find semantically similar documents.")

c.save()

print("Sample PDF created successfully!")
