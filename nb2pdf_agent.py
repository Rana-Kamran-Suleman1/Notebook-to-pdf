#!/usr/bin/env python3
"""
AI-Powered Jupyter Notebook to PDF Converter
Uses LangChain with Ollama to intelligently convert .ipynb files to professional PDFs.
"""

import argparse
import base64
import json
import os
import sys
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# LangChain imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# Notebook parsing
import nbformat

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, Preformatted, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# Syntax highlighting
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


@dataclass
class CellOutput:
    """Represents a notebook cell's output."""
    output_type: str
    text: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class NotebookCell:
    """Represents a parsed notebook cell."""
    cell_type: str  # 'markdown' or 'code'
    source: str
    outputs: list = field(default_factory=list)
    execution_count: Optional[int] = None


@dataclass
class ParsedNotebook:
    """Represents a fully parsed notebook."""
    metadata: dict
    cells: list
    name: str


class NotebookParser:
    """Parses .ipynb files into structured data."""

    @staticmethod
    def parse(notebook_path: str) -> ParsedNotebook:
        """Load and parse a .ipynb file."""
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        cells = []
        for cell in nb.cells:
            cell_type = cell.cell_type
            source = ''.join(cell.source) if isinstance(cell.source, list) else cell.source

            outputs = []
            if cell_type == 'code' and hasattr(cell, 'outputs'):
                for output in cell.outputs:
                    out = CellOutput(output_type=output.get('output_type', ''))

                    if output.get('output_type') == 'stream':
                        out.text = ''.join(output.get('text', []))
                    elif output.get('output_type') == 'execute_result':
                        out.text = ''.join(output.get('data', {}).get('text/plain', []))
                    elif output.get('output_type') == 'error':
                        out.error = ''.join(output.get('traceback', []))
                    elif output.get('output_type') == 'display_data':
                        out.data = output.get('data', {})

                    outputs.append(out)

            cells.append(NotebookCell(
                cell_type=cell_type,
                source=source,
                outputs=outputs,
                execution_count=cell.get('execution_count')
            ))

        return ParsedNotebook(
            metadata=nb.metadata,
            cells=cells,
            name=nb.metadata.get('kernelspec', {}).get('display_name', Path(notebook_path).stem)
        )


class MarkdownProcessor:
    """Processes markdown content for PDF-friendly formatting."""

    @staticmethod
    def process(markdown_text: str) -> list:
        """Convert markdown to ReportLab flowables."""
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        styles = getSampleStyleSheet()
        elements = []

        lines = markdown_text.split('\n')
        in_code_block = False
        code_content = []

        for line in lines:
            # Handle code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    # End code block - add as preformatted text with background
                    code_text = '\n'.join(code_content)
                    code_pre = Preformatted(code_text, styles['Code'])
                    code_table = Table([[code_pre]], colWidths=[6.5*inch])
                    code_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.Color(0.95, 0.95, 0.95)),
                        ('BOX', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('LEFTPADDING', (0, 0), (-1, -1), 5),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ]))
                    elements.append(code_table)
                    elements.append(Spacer(1, 0.1 * inch))
                    code_content = []
                in_code_block = not in_code_block
                continue

            if in_code_block:
                code_content.append(line)
                continue

            # Handle headings
            if line.startswith('### '):
                elements.append(Paragraph(line[4:], styles['Heading3']))
            elif line.startswith('## '):
                elements.append(Paragraph(line[3:], styles['Heading2']))
            elif line.startswith('# '):
                elements.append(Paragraph(line[2:], styles['Heading1']))
            # Handle bullet points
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                elements.append(Paragraph(f"• {line.strip()[2:]}", styles['BodyText']))
            # Handle horizontal rule
            elif line.strip() == '---':
                elements.append(Spacer(1, 0.2 * inch))
            # Regular text
            elif line.strip():
                # Convert bold and italic - handle them as whole words/phrases
                import re
                text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
                text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
                elements.append(Paragraph(text, styles['BodyText']))
            else:
                elements.append(Spacer(1, 0.1 * inch))

        return elements


class CodeHighlighter:
    """Applies syntax highlighting to code blocks."""

    @staticmethod
    def highlight(code: str) -> str:
        """Apply syntax highlighting to Python code."""
        formatter = HtmlFormatter()
        return highlight(code, PythonLexer(), formatter)

    @staticmethod
    def to_reportlab_formatted(code: str, styles) -> Preformatted:
        """Convert highlighted code to ReportLab preformatted text."""
        # Simple monospace formatting (without HTML colors for simplicity)
        return Preformatted(
            code,
            styles['Code']
        )


class PDFGenerator:
    """Generates professional PDFs from parsed notebooks."""

    def __init__(self, theme: str = 'light'):
        self.theme = theme
        self.styles = self._create_styles()

    def _create_styles(self):
        """Create custom paragraph styles for the PDF."""
        base = getSampleStyleSheet()

        # Title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=base['Title'],
            fontSize=28,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2C3E50') if self.theme == 'light' else colors.white
        )

        # Heading styles
        h1 = ParagraphStyle(
            'CustomH1',
            parent=base['Heading1'],
            fontSize=20,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2980B9')
        )

        h2 = ParagraphStyle(
            'CustomH2',
            parent=base['Heading2'],
            fontSize=16,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#27AE60')
        )

        h3 = ParagraphStyle(
            'CustomH3',
            parent=base['Heading3'],
            fontSize=14,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor('#8E44AD')
        )

        # Body text
        body = ParagraphStyle(
            'CustomBody',
            parent=base['BodyText'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )

        # Code style
        code = ParagraphStyle(
            'Code',
            parent=base['Code'],
            fontSize=9,
            fontName='Courier',
            leftIndent=20,
            rightIndent=20,
            spaceBefore=5,
            spaceAfter=5
        )

        # Output style
        output = ParagraphStyle(
            'Output',
            parent=base['Code'],
            fontSize=9,
            fontName='Courier',
            leftIndent=20,
            textColor=colors.HexColor('#2C3E50')
        )

        # Error style
        error_style = ParagraphStyle(
            'Error',
            parent=output,
            textColor=colors.HexColor('#E74C3C'),
        )

        return {
            'Title': title_style,
            'Heading1': h1,
            'Heading2': h2,
            'Heading3': h3,
            'BodyText': body,
            'Code': code,
            'Output': output,
            'Error': error_style
        }

    def generate(self, notebook: ParsedNotebook, output_path: str) -> str:
        """Generate PDF from parsed notebook."""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )

        elements = []

        # Title page
        elements.append(Spacer(1, 2 * inch))
        elements.append(Paragraph(notebook.name, self.styles['Title']))
        elements.append(Spacer(1, 0.5 * inch))

        # Table of contents placeholder
        elements.append(Paragraph("Table of Contents", self.styles['Heading2']))
        cell_count = sum(1 for c in notebook.cells if c.cell_type == 'markdown')
        elements.append(Paragraph(f"Total sections: {cell_count}", self.styles['BodyText']))
        elements.append(PageBreak())

        # Process cells
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'markdown':
                elements.extend(MarkdownProcessor.process(cell.source))
                elements.append(Spacer(1, 0.2 * inch))

            elif cell.cell_type == 'code':
                # Code cell header
                exec_num = cell.execution_count or ' '
                elements.append(Paragraph(
                    f"<b>Code [{exec_num}]</b>",
                    self.styles['Heading3']
                ))

                # Code block
                code_block = CodeHighlighter.to_reportlab_formatted(
                    cell.source.strip(), self.styles
                )
                elements.append(code_block)
                elements.append(Spacer(1, 0.1 * inch))

                # Outputs
                for output in cell.outputs:
                    if output.error:
                        elements.append(Preformatted(
                            output.error,
                            self.styles['Error']
                        ))
                    elif output.text:
                        out_pre = Preformatted(output.text.strip(), self.styles['Output'])
                        out_table = Table([[out_pre]], colWidths=[6.5*inch])
                        out_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, -1), colors.Color(0.97, 0.97, 0.97)),
                            ('BOX', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('LEFTPADDING', (0, 0), (-1, -1), 5),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                            ('TOPPADDING', (0, 0), (-1, -1), 5),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ]))
                        elements.append(out_table)
                    elif output.data:
                        # Handle image data
                        if 'image/png' in output.data:
                            img_data = output.data['image/png']
                            img_bytes = base64.b64decode(img_data)
                            img = Image(io.BytesIO(img_bytes), width=4*inch, height=3*inch)
                            elements.append(img)
                        elif 'image/jpeg' in output.data:
                            img_data = output.data['image/jpeg']
                            img_bytes = base64.b64decode(img_data)
                            img = Image(io.BytesIO(img_bytes), width=4*inch, height=3*inch)
                            elements.append(img)

                elements.append(Spacer(1, 0.3 * inch))

            elements.append(KeepTogether([]))

        # Build PDF
        doc.build(elements)
        return output_path


class NotebookToPDFAgent:
    """
    AI Agent that orchestrates notebook to PDF conversion using LangChain + Ollama.
    """

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """Initialize the agent with Ollama LLM."""
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.3,
            verbose=True
        )
        self.parser = NotebookParser()
        self.generator = PDFGenerator()

    def analyze_notebook_structure(self, notebook: ParsedNotebook) -> dict:
        """Use the LLM to analyze notebook structure and suggest processing strategy."""
        prompt = PromptTemplate.from_template("""
Analyze this Jupyter notebook and provide a JSON response with:
1. "title": Suggested title for the PDF report
2. "sections": List of main sections/headings found
3. "has_outputs": Boolean indicating if code cells have outputs
4. "complexity": "simple", "medium", or "complex" based on content
5. "recommendations": List of suggestions for formatting

Notebook cells preview:
{notebook_preview}

Return ONLY valid JSON.
""")

        # Create preview
        preview = []
        for i, cell in enumerate(notebook.cells[:10]):  # First 10 cells
            preview.append(f"Cell {i}: [{cell.cell_type}] {cell.source[:100]}...")

        chain = prompt | self.llm | JsonOutputParser()

        try:
            result = chain.invoke({"notebook_preview": "\n".join(preview)})
            return result
        except Exception as e:
            return {
                "title": notebook.name,
                "sections": [],
                "has_outputs": any(c.outputs for c in notebook.cells),
                "complexity": "medium",
                "recommendations": ["Standard processing recommended"]
            }

    def convert(self, input_path: str, output_path: Optional[str] = None,
                theme: str = "light", use_ai_analysis: bool = True) -> str:
        """
        Convert a Jupyter notebook to PDF.

        Args:
            input_path: Path to the .ipynb file
            output_path: Optional output PDF path (defaults to input name with .pdf)
            theme: "light" or "dark"
            use_ai_analysis: Whether to use AI to analyze structure first

        Returns:
            Path to the generated PDF
        """
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Notebook not found: {input_path}")

        if not input_path.endswith('.ipynb'):
            raise ValueError("Input file must be a .ipynb file")

        # Determine output path
        if output_path is None:
            output_path = str(Path(input_path).with_suffix('.pdf'))

        # Parse notebook
        print(f"Parsing notebook: {input_path}")
        notebook = self.parser.parse(input_path)

        # AI analysis (optional)
        if use_ai_analysis:
            print("Analyzing notebook structure with AI...")
            try:
                analysis = self.analyze_notebook_structure(notebook)
                print(f"Analysis complete: {analysis.get('complexity', 'unknown')} complexity")
            except Exception as e:
                print(f"AI analysis skipped (Ollama may not be running): {e}")

        # Generate PDF
        print(f"Generating PDF: {output_path}")
        self.generator = PDFGenerator(theme=theme)
        self.generator.generate(notebook, output_path)

        return output_path


def main():
    """CLI entry point with interactive input support."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Jupyter Notebook to PDF Converter"
    )
    parser.add_argument("input", nargs="?", help="Path to the .ipynb file")
    parser.add_argument("-o", "--output", help="Output PDF path")
    parser.add_argument("-m", "--model", default="llama3.2",
                        help="Ollama model to use (default: llama3.2)")
    parser.add_argument("-u", "--url", default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--theme", choices=["light", "dark"], default="light",
                        help="PDF theme (default: light)")
    parser.add_argument("--no-ai", action="store_true",
                        help="Disable AI analysis")

    args = parser.parse_args()

    # Interactive input if no path provided
    input_path = args.input
    if input_path is None:
        input_path = input("Enter path to notebook file (.ipynb): ").strip()

    # Create agent
    agent = NotebookToPDFAgent(model=args.model, base_url=args.url)

    # Convert
    try:
        output_path = agent.convert(
            input_path,
            output_path=args.output,
            theme=args.theme,
            use_ai_analysis=not args.no_ai
        )
        print(f"Successfully generated: {output_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
