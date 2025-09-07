"""
Sectioner - Multi-transaction Detection
Implements Parser.pdf Section 3: Build provider/transaction blocks using cues
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging


@dataclass
class ProviderBlock:
    """Container for a detected provider/transaction block"""
    text: str
    start_line: int
    end_line: int
    transaction_type: Optional[str] = None
    provider_indicators: List[str] = None
    confidence: float = 0.0
    shared_fields: Dict[str, str] = None
    
    def __post_init__(self):
        if self.provider_indicators is None:
            self.provider_indicators = []
        if self.shared_fields is None:
            self.shared_fields = {}


class BlockSectioner:
    """
    Multi-transaction detection system
    Builds provider/transaction blocks using hard and soft cues
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Hard cues for provider blocks
        self.hard_provider_cues = [
            r'^\s*Provider\s*:',
            r'^\s*Provider\s+Name\s*:',
            r'^\s*Provider\s+\d+\s*:',  # Provider 1:, Provider 2:, etc.
            r'^\s*Doctor\s*:',
            r'^\s*Physician\s*:',
        ]
        
        # Patterns for table detection
        self.table_header_patterns = [
            r'Provider.*Name.*NPI',
            r'Name.*NPI.*TIN',
            r'NPI.*Provider.*Specialty',
        ]
        
        # Bullet/numbered item patterns
        self.list_item_patterns = [
            r'^\s*[-•*]\s+',  # Bullet points
            r'^\s*\d+\.\s+',  # Numbered lists
            r'^\s*\(\d+\)\s+',  # Parenthetical numbers
        ]
        
        # Soft cues - sequences that likely indicate provider blocks
        self.soft_cue_patterns = [
            r'NPI[:\s]*\d{10}',  # NPI followed by number
            r'License[:\s]*[A-Z]\d+',  # License pattern
            r'TIN[:\s]*\d{2}-?\d{7}',  # TIN pattern
        ]
        
        # Transaction scope cues
        self.transaction_scope_cues = {
            'add': [r'add', r'new', r'include', r'enroll', r'join'],
            'term': [r'term', r'terminate', r'remove', r'discontinue', r'end'],
            'update': [r'update', r'change', r'modify', r'revise', r'correct', r'move'],
        }
        
        # Email-scope field patterns (shared across blocks)
        self.email_scope_patterns = {
            'tin': r'TIN[:\s#]*(\d{2}-?\d{7})',
            'ppg': r'PPG[:\s#\']*([A-Za-z0-9]+)',
            'lob': r'(?:Medicare|Medicaid|Commercial|HMO|PPO)',
            'organization': r'(?:Medical Group|Healthcare|Clinic|Practice)',
        }
    
    def section_content(self, text: str) -> List[ProviderBlock]:
        """
        Main sectioning method - detect provider/transaction blocks
        Returns list of detected blocks with metadata
        """
        lines = text.split('\n')
        blocks = []
        
        # First pass: detect email-scope shared fields
        shared_fields = self._extract_shared_fields(text)
        
        # Second pass: detect provider blocks using multiple strategies
        hard_blocks = self._detect_hard_cue_blocks(lines, shared_fields)
        table_blocks = self._detect_table_blocks(lines, shared_fields)
        soft_blocks = self._detect_soft_cue_blocks(lines, shared_fields)
        
        # Combine and deduplicate blocks
        all_blocks = hard_blocks + table_blocks + soft_blocks
        blocks = self._merge_overlapping_blocks(all_blocks)
        
        # Third pass: apply transaction scope to blocks
        blocks = self._apply_transaction_scope(blocks, lines)
        
        # Filter out low-confidence blocks
        blocks = [block for block in blocks if block.confidence >= 0.3]
        
        self.logger.info(f"Detected {len(blocks)} provider blocks")
        return blocks
    
    def _extract_shared_fields(self, text: str) -> Dict[str, str]:
        """Extract email-scope fields that apply to all blocks"""
        shared_fields = {}
        
        for field_name, pattern in self.email_scope_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if field_name in ['lob']:
                    # Multiple values possible
                    shared_fields[field_name] = ', '.join(set(matches))
                else:
                    # Single value
                    shared_fields[field_name] = matches[0]
        
        return shared_fields
    
    def _detect_hard_cue_blocks(self, lines: List[str], shared_fields: Dict[str, str]) -> List[ProviderBlock]:
        """Detect blocks using hard cues like 'Provider:' """
        blocks = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for hard provider cues
            for pattern in self.hard_provider_cues:
                if re.match(pattern, line, re.IGNORECASE):
                    # Found a provider block start
                    start_line = i
                    
                    # Find the end of this block
                    end_line = self._find_block_end(lines, start_line)
                    
                    block_text = '\n'.join(lines[start_line:end_line + 1])
                    
                    block = ProviderBlock(
                        text=block_text,
                        start_line=start_line,
                        end_line=end_line,
                        provider_indicators=['hard_cue'],
                        confidence=0.9,  # High confidence for hard cues
                        shared_fields=shared_fields.copy()
                    )
                    
                    blocks.append(block)
                    i = end_line + 1
                    break
            else:
                i += 1
        
        return blocks
    
    def _detect_table_blocks(self, lines: List[str], shared_fields: Dict[str, str]) -> List[ProviderBlock]:
        """Detect table rows with provider information"""
        blocks = []
        
        # Look for table headers
        for i, line in enumerate(lines):
            for pattern in self.table_header_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Found table header, look for data rows
                    table_blocks = self._extract_table_rows(lines, i, shared_fields)
                    blocks.extend(table_blocks)
                    break
        
        return blocks
    
    def _extract_table_rows(self, lines: List[str], header_line: int, shared_fields: Dict[str, str]) -> List[ProviderBlock]:
        """Extract individual rows from a detected table"""
        blocks = []
        
        # Simple table detection - look for rows with similar structure
        i = header_line + 1
        while i < len(lines) and i < header_line + 20:  # Limit search
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check if line looks like a data row
            if self._looks_like_table_row(line):
                block = ProviderBlock(
                    text=line,
                    start_line=i,
                    end_line=i,
                    provider_indicators=['table_row'],
                    confidence=0.8,
                    shared_fields=shared_fields.copy()
                )
                blocks.append(block)
            
            i += 1
        
        return blocks
    
    def _looks_like_table_row(self, line: str) -> bool:
        """Heuristic to detect if a line looks like a table row"""
        # Look for typical separators and provider indicators
        separators = ['|', '\t', '  ']  # Multiple spaces count as separator
        provider_indicators = [r'\d{10}', r'[A-Z]\d{5}']  # NPI, License patterns
        
        has_separators = any(sep in line for sep in separators)
        has_indicators = any(re.search(pattern, line) for pattern in provider_indicators)
        
        return has_separators and has_indicators
    
    def _detect_soft_cue_blocks(self, lines: List[str], shared_fields: Dict[str, str]) -> List[ProviderBlock]:
        """Detect blocks using soft cues (NPI + License/State + date sequences)"""
        blocks = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for NPI pattern
            if re.search(r'NPI[:\s]*\d{10}', line, re.IGNORECASE):
                # Found potential start, look for supporting evidence
                evidence_score = self._score_soft_evidence(lines, i)
                
                if evidence_score >= 2:  # Need at least 2 pieces of evidence
                    start_line = i
                    end_line = self._find_block_end(lines, start_line, max_lines=5)
                    
                    block_text = '\n'.join(lines[start_line:end_line + 1])
                    
                    block = ProviderBlock(
                        text=block_text,
                        start_line=start_line,
                        end_line=end_line,
                        provider_indicators=['soft_cue'],
                        confidence=min(evidence_score * 0.2, 0.7),  # Cap at 0.7
                        shared_fields=shared_fields.copy()
                    )
                    
                    blocks.append(block)
                    i = end_line + 1
                else:
                    i += 1
            else:
                i += 1
        
        return blocks
    
    def _score_soft_evidence(self, lines: List[str], start_line: int, window: int = 3) -> int:
        """Score the evidence around a potential provider block"""
        score = 0
        
        # Look in a small window around the start line
        for i in range(max(0, start_line - window), min(len(lines), start_line + window + 1)):
            line = lines[i].strip()
            
            for pattern in self.soft_cue_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    score += 1
        
        return score
    
    def _find_block_end(self, lines: List[str], start_line: int, max_lines: int = 10) -> int:
        """Find the end of a provider block"""
        end_line = start_line
        
        for i in range(start_line + 1, min(len(lines), start_line + max_lines + 1)):
            line = lines[i].strip()
            
            # Stop at empty lines or new provider blocks
            if not line:
                # Allow one empty line
                if i + 1 < len(lines) and lines[i + 1].strip():
                    continue
                else:
                    break
            
            # Stop if we hit another provider block start
            for pattern in self.hard_provider_cues:
                if re.match(pattern, line, re.IGNORECASE):
                    return end_line
            
            end_line = i
        
        return end_line
    
    def _merge_overlapping_blocks(self, blocks: List[ProviderBlock]) -> List[ProviderBlock]:
        """Merge overlapping or adjacent blocks"""
        if not blocks:
            return blocks
        
        # Sort by start line
        blocks.sort(key=lambda b: b.start_line)
        
        merged = []
        current = blocks[0]
        
        for next_block in blocks[1:]:
            # Check for overlap or adjacency
            if next_block.start_line <= current.end_line + 2:  # Allow small gaps
                # Merge blocks
                current.end_line = max(current.end_line, next_block.end_line)
                current.text = f"{current.text}\n{next_block.text}"
                current.provider_indicators.extend(next_block.provider_indicators)
                current.confidence = max(current.confidence, next_block.confidence)
            else:
                merged.append(current)
                current = next_block
        
        merged.append(current)
        return merged
    
    def _apply_transaction_scope(self, blocks: List[ProviderBlock], lines: List[str]) -> List[ProviderBlock]:
        """Apply transaction scope cues to blocks"""
        
        # Find transaction scope markers
        transaction_markers = []
        for i, line in enumerate(lines):
            for trans_type, keywords in self.transaction_scope_cues.items():
                for keyword in keywords:
                    if re.search(rf'\b{keyword}\b', line, re.IGNORECASE):
                        transaction_markers.append((i, trans_type))
                        break
        
        # Apply nearest scope to each block
        for block in blocks:
            block.transaction_type = self._find_nearest_transaction_scope(
                block.start_line, transaction_markers
            )
        
        return blocks
    
    def _find_nearest_transaction_scope(self, block_line: int, markers: List[Tuple[int, str]]) -> Optional[str]:
        """Find the nearest transaction scope marker to a block"""
        if not markers:
            return None
        
        # Find the closest marker that comes before this block
        best_marker = None
        best_distance = float('inf')
        
        for marker_line, trans_type in markers:
            if marker_line <= block_line:
                distance = block_line - marker_line
                if distance < best_distance:
                    best_distance = distance
                    best_marker = trans_type
        
        return best_marker