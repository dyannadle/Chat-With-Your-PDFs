from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)
pdf.cell(40, 10, 'Hello World. This is a dummy PDF file for testing LLM connection errors.')
pdf.output('dummy_test.pdf')
