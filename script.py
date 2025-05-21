#!/usr/bin/env python3
import os
import sys
import argparse
from fpdf import FPDF
import mimetypes

class SimpleFileExporter:
    """
    Simple PDF exporter that lists file routes and their content sequentially
    Optimized for SvelteKit and React JSX files
    """
    
    def __init__(self, start_dir='.', output_file='output.pdf', max_file_size=1048576):
        """
        Initialize the exporter
        
        Args:
            start_dir (str): Directory to start scanning from
            output_file (str): Output PDF filename
            max_file_size (int): Maximum file size in bytes to include
        """
        self.start_dir = os.path.abspath(start_dir)
        self.output_file = output_file
        self.max_file_size = max_file_size
        self.processed_files = 0
        
        # Initialize PDF
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(True, margin=15)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.add_page()
        
        # Add title
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Project Files Content', 0, 1, 'C')
        self.pdf.ln(5)
    
    def is_text_file(self, filepath):
        """Check if file is a text file that should be included"""
        # List of extensions to include
        code_extensions = {
            # Svelte/React
            '.svelte', '.jsx', '.tsx', '.js', '.ts',
            # Web
            '.html', '.css', '.json', '.md',
            # Config
            '.config.js', '.config.ts', '.json', '.yaml', '.yml',
            # Other common code files
            '.py', '.php', '.rb', '.go', '.java', '.c', '.cpp', '.cs'
        }
        
        ext = os.path.splitext(filepath)[1].lower()
        # Special handling for config files
        if filepath.endswith('.config.js') or filepath.endswith('.config.ts'):
            return True
            
        # Check extension
        if ext in code_extensions:
            return True
            
        # Check mime type as fallback
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type and mime_type.startswith('text/'):
            return True
            
        return False
    
    def process_folder(self, folder_path):
        """
        Process all files in a folder recursively
        
        Args:
            folder_path (str): Path to the folder
        """
        try:
            for root, dirs, files in os.walk(folder_path):
                # Skip hidden folders, node_modules, and .svelte-kit
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules' and d != '.svelte-kit']
                
                for file in files:
                    # Skip hidden files and package-lock.json
                    if file.startswith('.') or file == 'package-lock.json':
                        continue
                        
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.start_dir)
                    
                    # Skip large files
                    if os.path.getsize(file_path) > self.max_file_size:
                        continue
                    
                    # Process text files
                    if self.is_text_file(file_path):
                        self.add_file_content(file_path, rel_path)
                        self.processed_files += 1
        
        except Exception as e:
            print(f"Error processing folder {folder_path}: {str(e)}")
    
    def add_file_content(self, file_path, rel_path):
        """
        Add file route and content to PDF
        
        Args:
            file_path (str): Path to the file
            rel_path (str): Relative path from start directory
        """
        try:
            # Try different encodings to read the file
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"Warning: Could not decode file {rel_path}")
                return
            
            # Clean content of any non-ASCII characters
            clean_content = ''.join(c if ord(c) < 128 else '_' for c in content)
            clean_path = ''.join(c if ord(c) < 128 else '_' for c in rel_path)
            
            # Add file header - ensure we have enough space
            if self.pdf.get_y() > 250:
                self.pdf.add_page()
            
            # Route header with background
            self.pdf.set_font('Arial', 'B', 12)
            self.pdf.set_fill_color(220, 220, 220)
            self.pdf.multi_cell(0, 10, f'FILE: {clean_path}', 1, 'L', True)
            
            # Content
            self.pdf.set_font('Courier', '', 8)
            
            # Split content into lines and add to PDF
            lines = clean_content.split('\n')
            for line in lines:
                current_y = self.pdf.get_y()
                if current_y > 270:  # Check if near bottom of page
                    self.pdf.add_page()
                
                # Wrap long lines
                while len(line) > 0:
                    line_width = min(120, len(line))
                    self.pdf.cell(0, 5, line[:line_width], 0, 1)
                    line = line[line_width:]
            
            # Add separator
            self.pdf.ln(5)
            self.pdf.cell(0, 0, '', 'T', 1)
            self.pdf.ln(5)
            
        except Exception as e:
            print(f"Error processing file {rel_path}: {str(e)}")
    
    def generate(self):
        """Generate the PDF file"""
        print(f"Scanning folder: {self.start_dir}")
        self.process_folder(self.start_dir)
        
        # Add summary at the end
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 14)
        self.pdf.cell(0, 10, 'Summary', 0, 1, 'C')
        self.pdf.set_font('Arial', '', 12)
        self.pdf.cell(0, 10, f'Files processed: {self.processed_files}', 0, 1)
        
        # Save PDF
        self.pdf.output(self.output_file)
        print(f"PDF generated: {os.path.abspath(self.output_file)}")
        print(f"Processed {self.processed_files} files.")

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Generate a PDF with file routes and their content')
    parser.add_argument('-d', '--directory', default='.', 
                        help='Directory to scan (default: current directory)')
    parser.add_argument('-o', '--output', default='output.pdf', 
                        help='Output PDF filename (default: output.pdf)')
    parser.add_argument('-m', '--max-size', type=int, default=1048576, 
                        help='Maximum file size in bytes (default: 1MB)')
    args = parser.parse_args()
    
    try:
        exporter = SimpleFileExporter(args.directory, args.output, args.max_size)
        exporter.generate()
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())