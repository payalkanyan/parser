"""
MIME Parser & Content Normalization: MIME parsing, content normalization, thread trimming
"""

import re
from email import policy
from email.parser import BytesParser
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import logging

from .attachments import AttachmentRouter


class ParsedContent:
    """Container for parsed email content"""
    
    def __init__(self):
        self.text_content: str = ""
        self.html_content: str = ""
        self.normalized_text: str = ""
        self.attachments: List[Dict] = []
        self.headers: Dict[str, str] = {}
        self.thread_trimmed: bool = False
        self.source_file: Optional[Path] = None


class EMLParser:
    
    def __init__(self):
        self.attachment_router = AttachmentRouter()
        self.logger = logging.getLogger(__name__)
        
        # Regex patterns for thread detection
        self.reply_patterns = [
            r'^From:\s+.*',
            r'^Sent:\s+.*',
            r'^To:\s+.*',
            r'^Subject:\s+.*',
            r'-----Original Message-----',
            r'________________________________',
            r'On .* wrote:',
            r'> .*' 
        ]
        
        self.boilerplate_patterns = [
            r'unsubscribe.*?(?=\n|$)',
            r'confidential.*?(?=\n|$)',
            r'disclaimer.*?(?=\n|$)',
            r'this email.*?confidential.*?(?=\n|$)',
            r'please.*?unsubscribe.*?(?=\n|$)'
        ]
    
    def parse_eml(self, eml_path: Path) -> ParsedContent:
        """Main parsing entry point"""
        content = ParsedContent()
        content.source_file = eml_path
        
        try:
            with open(eml_path, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
            
            
            content.headers = self._extract_headers(msg)
            content.text_content, content.html_content = self._extract_body_content(msg)
            content.attachments = self.attachment_router.extract_attachments(msg)
            if content.html_content:
                content.normalized_text = self._normalize_html_content(content.html_content)
            else:
                content.normalized_text = self._normalize_text_content(content.text_content)
            
            content.normalized_text, content.thread_trimmed = self._trim_thread(content.normalized_text)
            
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error parsing {eml_path}: {str(e)}")
            content.normalized_text = f"Error reading email: {str(e)}"
            return content
    
    def _extract_headers(self, msg: EmailMessage) -> Dict[str, str]:
        """Extract relevant email headers"""
        headers = {}
        
        header_keys = ['From', 'To', 'Subject', 'Date', 'Message-ID']
        for key in header_keys:
            if key in msg:
                headers[key] = str(msg[key])
        
        return headers
    
    def _extract_body_content(self, msg: EmailMessage) -> Tuple[str, str]:
        """
        Extract text and HTML content from MIME message
        Prefer HTML part for structure preservation, fallback to plain text
        """
        text_content = ""
        html_content = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                
                if content_type == "text/plain":
                    try:
                        text_content += part.get_content()
                    except Exception as e:
                        self.logger.warning(f"Could not decode text part: {e}")
                
                elif content_type == "text/html":
                    try:
                        html_content += part.get_content()
                    except Exception as e:
                        self.logger.warning(f"Could not decode HTML part: {e}")
        
        else:
            # Single part message
            content_type = msg.get_content_type()
            try:
                if content_type == "text/plain":
                    text_content = msg.get_content()
                elif content_type == "text/html":
                    html_content = msg.get_content()
            except Exception as e:
                self.logger.warning(f"Could not decode message content: {e}")
        
        return text_content, html_content
    
    def _normalize_html_content(self, html_content: str) -> str:
        """
        Normalize HTML content while preserving structure
        Key: preserve line breaks and table layout
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Convert tables to text while preserving structure
            tables = soup.find_all('table')
            for table in tables:
                self._normalize_table_structure(table)
            
            # Convert to text with preserved line breaks
            text = soup.get_text(separator='\n')
            
            # Clean up excessive whitespace while preserving structure
            text = self._clean_whitespace(text)
            
            return text
            
        except Exception as e:
            self.logger.warning(f"HTML parsing failed, using raw content: {e}")
            return html_content
    
    def _normalize_table_structure(self, table) -> None:
        """Convert HTML table to structured text format"""
        rows = table.find_all('tr')
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) > 1:
                # Adding separators for better parsing
                for i, cell in enumerate(cells[:-1]):
                    cell.append(' | ')
            
            # Add newlines after rows
            if row != rows[-1]: 
                row.append('\n')
    
    def _normalize_text_content(self, text_content: str) -> str:
        """Normalize plain text content"""
        text = self._clean_whitespace(text_content)
        text = self._normalize_unicode(text)
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean whitespace while preserving meaningful structure"""
        text = re.sub(r'\r\n?', '\n', text)
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        lines = []
        for line in text.split('\n'):
            stripped = line.rstrip()
            if stripped:
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 8:  
                    leading_spaces = 8
                lines.append(' ' * leading_spaces + stripped.lstrip())
            else:
                lines.append('')
        
        return '\n'.join(lines)
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize problematic unicode characters"""
        replacements = {
            ''': "'",
            ''': "'", 
            '"': '"',
            '"': '"',
            '–': '-',
            '—': '-',
            '…': '...',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _trim_thread(self, text: str) -> Tuple[str, bool]:
        """
        Trim email threads - keep topmost message unless older content
        contains unique provider blocks
        """
        lines = text.split('\n')
        
        reply_start = None
        for i, line in enumerate(lines):
            for pattern in self.reply_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    reply_start = i
                    break
            if reply_start is not None:
                break
        
        if reply_start is None:
            return text, False
        
        top_content = '\n'.join(lines[:reply_start])
        bottom_content = '\n'.join(lines[reply_start:])
        
        if self._has_unique_provider_blocks(bottom_content, top_content):
            return text, False
        else:
            return top_content.strip(), True
    
    def _has_unique_provider_blocks(self, older_content: str, newer_content: str) -> bool:
        """Check if older content has unique NPIs not in newer content"""
        npi_pattern = r'NPI[:\s]*(\d{10})'
        
        older_npis = set(re.findall(npi_pattern, older_content, re.IGNORECASE))
        newer_npis = set(re.findall(npi_pattern, newer_content, re.IGNORECASE))
        
        return len(older_npis - newer_npis) > 0
    
    def strip_boilerplate(self, text: str) -> str:
        """
        Strip common boilerplate patterns
        Called after extraction to avoid losing signal during parsing
        """
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        return text