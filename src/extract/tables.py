"""
Implements direct field mapping from structured tables
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from rapidfuzz import fuzz, process

from .patterns import ExtractionCandidate


@dataclass
class TableCell:
    """Container for table cell data"""
    value: str
    row: int
    col: int
    confidence: float = 1.0


@dataclass
class TableData:
    """Container for extracted table data"""
    headers: List[str]
    rows: List[List[str]]
    header_mappings: Dict[int, str] 
    confidence: float = 0.0


class TableExtractor:
    """
    Extract structured data from tables (HTML, text-based, attachment-derived)
    Maps table headers to standard fields using fuzzy matching
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.field_mappings = {
            'provider_name': [
                'provider name', 'provider', 'doctor name', 'physician name',
                'dr name', 'name', 'doctor', 'physician', 'practitioner'
            ],
            'npi': [
                'npi', 'npi #', 'npi number', 'national provider identifier',
                'provider id', 'provider identifier', 'npinumber', 'provider npi',
                'group npi'
            ],
            'tin': [
                'tin', 'tax id', 'federal id', 'ein', 'employer id',
                'tax identification', 'federal tax id', 'group tin'
            ],
            'specialty': [
                'specialty', 'speciality', 'practice specialty', 'medical specialty',
                'provider specialty', 'field', 'discipline'
            ],
            'license': [
                'license', 'state license', 'medical license', 'lic #',
                'license number', 'state lic', 'license #'
            ],
            'organization': [
                'organization', 'org', 'group', 'practice', 'medical group',
                'healthcare group', 'clinic', 'facility', 'organization name'
            ],
            'phone': [
                'phone', 'telephone', 'phone number', 'contact number',
                'tel', 'contact', 'phone #'
            ],
            'fax': [
                'fax', 'fax number', 'facsimile', 'fax #'
            ],
            'address': [
                'address', 'practice address', 'location', 'office address',
                'mailing address', 'complete address', 'address change'
            ],
            'ppg': [
                'ppg', 'ppg id', 'practice group', 'group id',
                'provider group', 'ppg number', 'practice group id'
            ],
            'effective_date': [
                'effective date', 'start date', 'begin date', 'effective'
            ],
            'term_date': [
                'term date', 'termination date', 'end date', 'expiration date'
            ],
            'term_reason': [
                'term reason', 'termination reason', 'reason', 'exit reason'
            ],
            'provider_type': [
                'provider type', 'type', 'practitioner type'
            ],
            'lob': [
                'lob', 'line of business', 'business line', 'network',
                'plan type', 'insurance type'
            ]
        }
        
        self.fuzzy_threshold = 60  
    
    def extract_from_html_table(self, html_content: str) -> List[TableData]:
        """Extract data from HTML tables"""
        tables = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            html_tables = soup.find_all('table')
            
            for table_elem in html_tables:
                table_data = self._parse_html_table(table_elem)
                if table_data:
                    tables.append(table_data)
        
        except ImportError:
            self.logger.warning("BeautifulSoup not available for HTML table parsing")
        except Exception as e:
            self.logger.error(f"HTML table extraction failed: {e}")
        
        return tables
    
    def extract_from_text_table(self, text: str) -> List[TableData]:
        """Extract data from text-based tables (both horizontal and vertical)"""
        tables = []
        
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            if self._looks_like_table_header(lines[i]):
                table_data = self._parse_text_table(lines, i)
                if table_data:
                    tables.append(table_data)
                    i = i + len(table_data.rows) + 2  
                else:
                    i += 1
            elif self._looks_like_vertical_table_start(lines, i):
                table_data, rows_processed = self._parse_vertical_table(lines, i)
                if table_data:
                    tables.append(table_data)
                    i = i + rows_processed  
                else:
                    i += 1
            else:
                i += 1
        
        return tables
    
    def extract_candidates_from_tables(self, tables: List[TableData]) -> Dict[str, List[ExtractionCandidate]]:
        """Convert table data to extraction candidates by field"""
        field_candidates = {}
        
        for table in tables:
            for row_idx, row in enumerate(table.rows):
                for col_idx, cell_value in enumerate(row):
                    if col_idx in table.header_mappings:
                        field_name = table.header_mappings[col_idx]
                        
                        if cell_value and cell_value.strip():
                            candidate = ExtractionCandidate(
                                value=cell_value.strip(),
                                confidence=table.confidence,
                                extractor_id=f"table_row_{row_idx}_col_{col_idx}",
                                position=0,  
                                context=f"Table row {row_idx + 1}",
                                validation_passed=True
                            )
                            
                            if field_name not in field_candidates:
                                field_candidates[field_name] = []
                            field_candidates[field_name].append(candidate)
        
        return field_candidates
    
    def _parse_html_table(self, table_elem) -> Optional[TableData]:
        """Parse individual HTML table element (handles both horizontal and vertical tables)"""
        try:
            rows = table_elem.find_all('tr')
            if not rows:
                return None

            is_vertical_table = self._is_html_vertical_table(rows)
            
            if is_vertical_table:
                return self._parse_html_vertical_table(rows)
            else:
                return self._parse_html_horizontal_table(rows)
        
        except Exception as e:
            self.logger.error(f"HTML table parsing failed: {e}")
            return None
    
    def _is_html_vertical_table(self, rows) -> bool:
        """Check if HTML table is vertical format (field: value pairs in rows)"""
        if len(rows) < 2:
            return False

        vertical_rows = 0
        for row in rows[:5]:  # Check first 5 rows
            cells = row.find_all(['td', 'th'])
            if len(cells) == 2:
                first_cell = cells[0].get_text(strip=True).lower()
                field_indicators = [
                    'provider', 'name', 'npi', 'tin', 'specialty', 'license', 
                    'organization', 'phone', 'fax', 'address', 'ppg', 'date', 
                    'reason', 'type', 'lob', 'group', 'effective'
                ]
                if any(indicator in first_cell for indicator in field_indicators):
                    vertical_rows += 1
        
        return vertical_rows >= 2  
    
    def _parse_html_horizontal_table(self, rows) -> Optional[TableData]:
        """Parse horizontal HTML table (traditional format)"""
        header_row = rows[0]
        headers = []
        
        for cell in header_row.find_all(['th', 'td']):
            headers.append(cell.get_text(strip=True))
        
        if not headers:
            return None
        

        data_rows = []
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = []
                for cell in cells:
                    row_data.append(cell.get_text(strip=True))
                data_rows.append(row_data)
        
        header_mappings = self._map_headers_to_fields(headers)
        
        confidence = len(header_mappings) / len(headers) if headers else 0
        
        return TableData(
            headers=headers,
            rows=data_rows,
            header_mappings=header_mappings,
            confidence=confidence
        )
    
    def _parse_html_vertical_table(self, rows) -> Optional[TableData]:
        """Parse vertical HTML table (field: value pairs in rows)"""
        field_value_pairs = []
        
        field_indicators = [
            'provider', 'name', 'npi', 'tin', 'specialty', 'license', 
            'organization', 'phone', 'fax', 'address', 'ppg', 'date', 
            'reason', 'type', 'lob', 'group', 'effective'
        ]
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) == 2:
                field = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                field_lower = field.lower()
                
                if field and value and any(indicator in field_lower for indicator in field_indicators):
                    field_value_pairs.append((field, value))
        
        if len(field_value_pairs) < 2:
            return None
        
        headers = [pair[0] for pair in field_value_pairs]
        data_row = [pair[1] for pair in field_value_pairs]
        
        header_mappings = self._map_headers_to_fields(headers)
        
        confidence = len(header_mappings) / len(headers) if headers else 0
        
        confidence = min(confidence + 0.2, 1.0)
        
        return TableData(
            headers=headers,
            rows=[data_row],  
            header_mappings=header_mappings,
            confidence=confidence
        )
    
    def _parse_text_table(self, lines: List[str], start_idx: int) -> Optional[TableData]:
        """Parse text-based table starting from header line"""
        try:
            header_line = lines[start_idx].strip()
            
            separator = self._detect_table_separator(header_line)
            if not separator:
                return None
            
            headers = self._split_table_row(header_line, separator)
            if not headers:
                return None
            
            data_rows = []
            i = start_idx + 1
            
            while i < len(lines):
                line = lines[i].strip()
                
                if not line:
                    break
                
                if self._is_table_row(line, separator):
                    row_data = self._split_table_row(line, separator)
                    if row_data and len(row_data) <= len(headers):
                        # Pad row to match header count
                        while len(row_data) < len(headers):
                            row_data.append("")
                        data_rows.append(row_data[:len(headers)])
                    i += 1
                else:
                    break
            
            if not data_rows:
                return None
            header_mappings = self._map_headers_to_fields(headers)
            confidence = len(header_mappings) / len(headers) if headers else 0
            
            return TableData(
                headers=headers,
                rows=data_rows,
                header_mappings=header_mappings,
                confidence=confidence
            )
        
        except Exception as e:
            self.logger.error(f"Text table parsing failed: {e}")
            return None
    
    def _looks_like_table_header(self, line: str) -> bool:
        """Check if line looks like a table header"""
        line = line.strip()
        if not line:
            return False
        
        header_indicators = ['provider', 'name', 'npi', 'tin', 'specialty', 'license']
        line_lower = line.lower()
        
        indicator_count = sum(1 for indicator in header_indicators if indicator in line_lower)
        has_separators = any(sep in line for sep in ['|', '\t']) or '  ' in line
        
        return indicator_count >= 2 and has_separators
    
    def _detect_table_separator(self, line: str) -> Optional[str]:
        """Detect the separator used in table row"""
        separators = ['|', '\t']
        
        for sep in separators:
            if line.count(sep) >= 1:
                return sep
        if re.search(r'\s{2,}', line):
            return 'spaces'
        
        return None
    
    def _split_table_row(self, line: str, separator: str) -> List[str]:
        """Split table row by separator"""
        if separator == 'spaces':
            parts = re.split(r'\s{2,}', line)
        else:
            parts = line.split(separator)
        
        # Clean up parts
        return [part.strip() for part in parts if part.strip()]
    
    def _is_table_row(self, line: str, separator: str) -> bool:
        """Check if line is a valid table row"""
        if separator == 'spaces':
            return bool(re.search(r'\s{2,}', line))
        else:
            return separator in line
    
    def _map_headers_to_fields(self, headers: List[str]) -> Dict[int, str]:
        """Map table headers to standard field names using fuzzy matching"""
        mappings = {}
        
        for col_idx, header in enumerate(headers):
            header_clean = header.lower().strip()
            
            best_field = None
            best_score = 0
            for field_name, variants in self.field_mappings.items():
                for variant in variants:
                    # Exact match
                    if header_clean == variant:
                        best_field = field_name
                        best_score = 100
                        break
                    
                    # Fuzzy match
                    score = fuzz.ratio(header_clean, variant)
                    if score > best_score and score >= self.fuzzy_threshold:
                        best_field = field_name
                        best_score = score
                
                if best_score == 100: 
                    break
            
            if best_field and best_score >= self.fuzzy_threshold:
                mappings[col_idx] = best_field
                self.logger.debug(f"Mapped header '{header}' -> '{best_field}' (score: {best_score})")
            else:
                self.logger.debug(f"Could not map header '{header}' (best score: {best_score})")
        
        return mappings
    
    def _looks_like_vertical_table_start(self, lines: List[str], start_idx: int) -> bool:
        """Check if this looks like the start of a vertical table (field: value format)"""
        if start_idx >= len(lines):
            return False
        
        line = lines[start_idx].strip()
        if not line:
            return False
        
       
        if ':' not in line:
            return False
        
    
        field_part = line.split(':', 1)[0].strip()
        if field_part.startswith('-'):
            field_part = field_part[1:].strip()
        field_part = field_part.lower()
        
        
        field_indicators = [
            'provider', 'name', 'npi', 'tin', 'specialty', 'license', 
            'organization', 'phone', 'fax', 'address', 'ppg', 'date', 
            'reason', 'type', 'lob', 'group', 'effective', 'termination'
        ]
        
        if not any(indicator in field_part for indicator in field_indicators):
            return False
        consecutive_pairs = 1
        for i in range(start_idx + 1, min(start_idx + 8, len(lines))):
            next_line = lines[i].strip()
            if next_line and ':' in next_line:
                next_field = next_line.split(':', 1)[0].strip()
                if next_field.startswith('-'):
                    next_field = next_field[1:].strip()
                next_field = next_field.lower()
                if any(indicator in next_field for indicator in field_indicators):
                    consecutive_pairs += 1
            elif not next_line:
                continue  
            else:
                break
        
      
        return consecutive_pairs >= 2
    
    def _parse_vertical_table(self, lines: List[str], start_idx: int) -> Tuple[Optional[TableData], int]:
        """Parse vertical table where each line is 'field: value' (with or without dash prefix)"""
        try:
            field_value_pairs = []
            i = start_idx
            rows_processed = 0
            
            
            field_indicators = [
                'provider', 'name', 'npi', 'tin', 'specialty', 'license', 
                'organization', 'phone', 'fax', 'address', 'ppg', 'date', 
                'reason', 'type', 'lob', 'group', 'effective', 'termination'
            ]
            
            
            while i < len(lines):
                line = lines[i].strip()
                
                if not line:
                    i += 1
                    rows_processed += 1
                    continue
                
                if ':' not in line:
                    break

                parts = line.split(':', 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip()
                    if field.startswith('-'):
                        field = field[1:].strip()
                    
                    field_lower = field.lower()
                    if field and value and any(indicator in field_lower for indicator in field_indicators):
                        field_value_pairs.append((field, value))
                    elif field and value:   
                        break
                
                i += 1
                rows_processed += 1
            
            if len(field_value_pairs) < 2: 
                return None, 0
            
           
            headers = [pair[0] for pair in field_value_pairs]
            # Single data row with all values
            data_row = [pair[1] for pair in field_value_pairs]
            header_mappings = self._map_headers_to_fields(headers)
            confidence = len(header_mappings) / len(headers) if headers else 0
            
            confidence = min(confidence + 0.2, 1.0)
            
            table_data = TableData(
                headers=headers,
                rows=[data_row],  # Single row with all the values
                header_mappings=header_mappings,
                confidence=confidence
            )
            
            return table_data, rows_processed
        
        except Exception as e:
            self.logger.error(f"Vertical table parsing failed: {e}")
            return None, 0