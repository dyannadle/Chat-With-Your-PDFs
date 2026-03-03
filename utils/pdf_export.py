from fpdf import FPDF  # Import the FPDF library for PDF generation
import datetime  # Import datetime for timestamping

class ChatPDF(FPDF):  # Define a custom PDF class inheriting from FPDF
    def header(self):  # Define the page header
        self.set_font('Arial', 'B', 15)  # Set font to Arial Bold, size 15
        self.cell(0, 10, 'Chat With Your PDFs - Conversation Export', 0, 1, 'C')  # Add a centered title
        self.ln(5)  # Add a small line break

    def footer(self):  # Define the page footer
        self.set_y(-15)  # Position at 1.5 cm from bottom
        self.set_font('Arial', 'I', 8)  # Set font to Arial Italic, size 8
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')  # Add a centered page number

def export_chat_to_pdf(messages, output_path="chat_export.pdf"):  # Define the main export function
    """
    Exports a list of chat messages to a PDF document.
    """
    pdf = ChatPDF()  # Initialize our custom PDF object
    pdf.add_page()  # Add the first page
    pdf.set_font("Arial", size=12)  # Set the default font for the body
    
    # Add a timestamp for the export
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Exported on: {timestamp}", 0, 1, 'L')
    pdf.ln(10)  # Add a line break
    
    for msg in messages:  # Iterate through each message in the session history
        role = msg["role"].capitalize()  # Capitalize the role name (User/Assistant)
        content = msg["content"]  # Get the text content
        
        # Set font for the role label (Bold)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{role}:", 0, 1, 'L')
        
        # Set font for the message content (Regular)
        pdf.set_font("Arial", size=11)
        # Use multi_cell to handle long messages and automatic line wraps
        # Replace non-latin characters if necessary (FPDF basic doesn't support full UTF-8 without extra setup)
        clean_content = content.encode('ascii', 'ignore').decode('ascii') 
        pdf.multi_cell(0, 10, clean_content)
        pdf.ln(5)  # Add a small space between messages
        
    pdf.output(output_path)  # Save the generated PDF to the specified path
    return output_path  # Return the path to the saved file
