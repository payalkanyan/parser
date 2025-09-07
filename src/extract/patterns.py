"""
Regex Extractors
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging


@dataclass
class ExtractionCandidate:
    """Container for an extraction candidate with metadata"""
    value: str
    confidence: float
    extractor_id: str
    position: int = -1
    context: str = ""
    validation_passed: bool = False


class PatternExtractor:
    """
    High-precision regex extractors with context windows and validation
    Each extractor returns candidates with confidence scores
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # NPI patterns with context windows
        self.npi_patterns = [
            r'NPI[:\s#]*(\d{10})',
            r'National Provider Identifier[:\s]*(\d{10})',
            r'Provider ID[:\s]*(\d{10})',
            r'(?:^|\s)(\d{10})(?=.*(?:provider|NPI|national))',
        ]
        
        # TIN patterns
        self.tin_patterns = [
            r'TIN[:\s#]*(\d{2}-?\d{7})',
            r'Tax\s+ID[:\s]*(\d{2}-?\d{7})',
            r'Federal\s+ID[:\s]*(\d{2}-?\d{7})',
            r'EIN[:\s]*(\d{2}-?\d{7})',
            r'Employer\s+ID[:\s]*(\d{2}-?\d{7})',
            r'(\d{2}-\d{7})',
        ]
        
        # PPG patterns - enhanced for better detection
        self.ppg_patterns = [
            # Standard labeled PPG patterns
            r'PPG[:\s#\']*([A-Za-z0-9]+)',
            r'PPG\s+ID[:\s]*([A-Za-z0-9]+)',
            r'Provider Practice Group[:\s]*([A-Za-z0-9]+)',
            r'Group\s+ID[:\s]*([A-Za-z0-9]+)',
            
            # Shared Risk context patterns (from Sample-2)
            r'Shared\s+Risk:\s*[^\n]*?[-–]\s*([A-Za-z0-9]{2,6})(?=\s|$|\n)',
            
            # PPG in network context - look for codes after PPG mentions
            r'PPG[#\'s]*[^:]*?[-–]\s*([A-Za-z0-9]{2,6})(?=\s|$|\n)',
            
            # General pattern: look for alphanumeric codes near PPG mentions
            r'(?i)(?:ppg|shared\s+risk)[^\n]{0,50}?[-–]\s*([A-Za-z0-9]{2,6})(?=\s|$|\n)',
            
            # Special format from samples (keep existing)
            r'Shared\s+Risk[:\s]*<([^>]+)>\s*[-–]\s*<([^>]+)>',
        ]
        
        # Phone/Fax patterns with labels
        self.phone_patterns = [
            r'Phone[:\s]*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            r'Tel[:\s]*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            r'Contact[:\s]*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            r'Phone\s+Number[:\s]*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            r'\((\d{3})\)\s*(\d{3})[-.\s]*(\d{4})',  # (555) 123-4567 format
        ]
        
        self.fax_patterns = [
            r'Fax[:\s]*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            r'Facsimile[:\s]*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            r'Fax\s+Number[:\s]*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
        ]
        
        # State License patterns
        self.license_patterns = [
            r'License[:\s#]*([A-Z]\d{5,6})',
            r'State\s+License[:\s]*([A-Z]\d{5,6})',
            r'Medical\s+License[:\s]*([A-Z]\d{5,6})',
            r'Lic\s*#[:\s]*([A-Z]\d{5,6})',
            r'State\s+Lic[:\s]*([A-Z]\d{5,6})',
        ]
        
        # Date patterns
        self.date_patterns = [
            r'Effective\s+Date[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            r'Term(?:ination)?\s+Date[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            r'Start\s+Date[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            r'End\s+Date[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',  # Generic date
        ]
        
        # Transaction Type patterns with negation guard
        # Transaction type clues - focus on ACTION words, not entities
        self.terminate_clues = [
            'terminate', 'terminated', 'termination', 'remove', 'discontinue',
            'end', 'stop', 'cease', 'withdraw', 'cancel', 'expire',
            'no longer'
        ]
        
        self.add_clues = [
            'add', 'new', 'include', 'enroll', 'register', 'join',
            'welcome', 'onboard', 'recruit', 'hire', 'bring on'
        ]
        
        self.update_clues = [
            'update', 'modify', 'change', 'revise', 'amend', 'correct',
            'edit', 'adjust', 'alter', 'refresh', 'renew'
        ]
        
        # Build patterns from clues
        self.transaction_patterns = {
            'Term': [
                # Direct termination clues
                r'\bterminate?d?\b',
                r'\btermination\b', 
                r'\bremove\b',
                r'\bdiscontinue\b',
                r'\bend\b',
                r'\bstop\b',
                r'\bcease\b',
                r'\bwithdraw\b',
                r'\bcancel\b',
                r'\bexpire\b',
                # Context patterns with quotes (like "Terminate")
                r'["\']terminate?["\']',
                r'\bno\s+longer\b',
                r'\beffective\s+immediately\b',
                r'\bvoluntary\b',
                # Time-based termination indicators
                r'\bas\s+of\b'
            ],
            'Add': [
                # Direct add clues
                r'\badd\b',
                r'\bnew\b',
                r'\binclude\b',
                r'\benroll\b',
                r'\bregister\b',
                r'\bjoin\b',
                r'\bwelcome\b',
                r'\bonboard\b',
                r'\brecruit\b',
                r'\bhire\b',
                r'\bbring\s+on\b'
            ],
            'Update': [
                # Direct update clues
                r'\bupdate\b',
                r'\bmodify\b',
                r'\bchange\b',
                r'\brevise\b',
                r'\bamend\b',
                r'\bcorrect\b',
                r'\bedit\b',
                r'\badjust\b',
                r'\balter\b',
                r'\brefresh\b',
                r'\brenew\b',
                # Location/practice specific changes
                r'\bmove\b',
                r'\brelocate\b',
                r'\btransfer\b'
            ]
        }
        
        # Negation patterns to ignore
        self.negation_patterns = [
            r'not\s+terminate', r'no\s+changes', r'don\'t\s+', r'will\s+not'
        ]
    
    def extract_npi_candidates(self, text: str) -> List[ExtractionCandidate]:
        """Extract NPI candidates with Luhn validation"""
        candidates = []
        
        for i, pattern in enumerate(self.npi_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                npi = match.group(1) if match.groups() else match.group(0)
                
                # Clean NPI
                npi_clean = re.sub(r'[^\d]', '', npi)
                
                if len(npi_clean) == 10:
                    # Luhn validation
                    validation_passed = self._validate_npi_luhn(npi_clean)
                    
                    candidate = ExtractionCandidate(
                        value=npi_clean,
                        confidence=0.9 if validation_passed else 0.6,
                        extractor_id=f"npi_pattern_{i}",
                        position=match.start(),
                        context=self._get_context(text, match.start(), 20),
                        validation_passed=validation_passed
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def extract_tin_candidates(self, text: str) -> List[ExtractionCandidate]:
        """Extract TIN candidates with length validation"""
        candidates = []
        
        for i, pattern in enumerate(self.tin_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                tin = match.group(1) if match.groups() else match.group(0)
                
                # Clean TIN - keep digits only
                tin_clean = re.sub(r'[^\d]', '', tin)
                
                if len(tin_clean) == 9:
                    # Format with hyphen
                    tin_formatted = f"{tin_clean[:2]}-{tin_clean[2:]}"
                    
                    candidate = ExtractionCandidate(
                        value=tin_formatted,
                        confidence=0.9 if i < 5 else 0.7,  # Higher confidence for labeled patterns
                        extractor_id=f"tin_pattern_{i}",
                        position=match.start(),
                        context=self._get_context(text, match.start(), 20),
                        validation_passed=True
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def extract_ppg_candidates(self, text: str) -> List[ExtractionCandidate]:
        """Extract PPG candidates and combine multiple PPG IDs"""
        candidates = []
        found_ppgs = set()  # Track unique PPG IDs
        
        for i, pattern in enumerate(self.ppg_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if i == len(self.ppg_patterns) - 1:  # Special "Shared Risk" format
                    if len(match.groups()) >= 2:
                        org = match.group(1)
                        ppg = match.group(2)
                    else:
                        continue
                else:
                    ppg = match.group(1) if match.groups() else match.group(0)
                
                # Clean PPG - keep alphanumeric characters
                ppg_clean = re.sub(r'[^\dA-Za-z]', '', ppg)
                
                # Validate PPG ID (2-6 alphanumeric characters, exclude common false positives)
                if ppg_clean and 2 <= len(ppg_clean) <= 6:
                    # Skip obvious false positives
                    false_positives = ['PPG', 'ID', 'TIN', 'NPI', 'MD', 'DR', 'HMO', 'PPO']
                    if ppg_clean.upper() not in false_positives:
                        found_ppgs.add(ppg_clean)
                        
                        candidate = ExtractionCandidate(
                            value=ppg_clean,
                            confidence=0.8 if i < 4 else 0.9,  # Context patterns get higher confidence
                            extractor_id=f"ppg_pattern_{i}",
                            position=match.start(),
                            context=self._get_context(text, match.start(), 30),
                            validation_passed=True
                        )
                        candidates.append(candidate)
        
        # If multiple PPG IDs found, combine them into a single candidate
        if len(found_ppgs) > 1:
            combined_ppg = ', '.join(sorted(found_ppgs))
            # Find the highest confidence candidate to use as template
            best_candidate = max(candidates, key=lambda x: x.confidence) if candidates else None
            
            if best_candidate:
                combined_candidate = ExtractionCandidate(
                    value=combined_ppg,
                    confidence=best_candidate.confidence,
                    extractor_id="ppg_combined",
                    position=best_candidate.position,
                    context=f"Multiple PPG IDs found: {combined_ppg}",
                    validation_passed=True
                )
                return [combined_candidate]
        
        return candidates
    
    def extract_phone_candidates(self, text: str) -> List[ExtractionCandidate]:
        """Extract phone number candidates with NANP validation"""
        candidates = []
        
        for i, pattern in enumerate(self.phone_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if len(match.groups()) == 3:  # (555) 123-4567 format
                    phone = ''.join(match.groups())
                else:
                    phone = match.group(1) if match.groups() else match.group(0)
                
                # Clean phone - digits only
                phone_clean = re.sub(r'[^\d]', '', phone)
                
                if len(phone_clean) == 10:
                    # Format as XXX-XXX-XXXX
                    phone_formatted = f"{phone_clean[:3]}-{phone_clean[3:6]}-{phone_clean[6:]}"
                    
                    candidate = ExtractionCandidate(
                        value=phone_formatted,
                        confidence=0.9 if 'phone' in pattern.lower() else 0.7,
                        extractor_id=f"phone_pattern_{i}",
                        position=match.start(),
                        context=self._get_context(text, match.start(), 20),
                        validation_passed=True
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def extract_fax_candidates(self, text: str) -> List[ExtractionCandidate]:
        """Extract fax number candidates"""
        candidates = []
        
        for i, pattern in enumerate(self.fax_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                fax = match.group(1) if match.groups() else match.group(0)
                
                # Clean fax - digits only
                fax_clean = re.sub(r'[^\d]', '', fax)
                
                if len(fax_clean) == 10:
                    # Format as XXX-XXX-XXXX
                    fax_formatted = f"{fax_clean[:3]}-{fax_clean[3:6]}-{fax_clean[6:]}"
                    
                    candidate = ExtractionCandidate(
                        value=fax_formatted,
                        confidence=0.9,
                        extractor_id=f"fax_pattern_{i}",
                        position=match.start(),
                        context=self._get_context(text, match.start(), 20),
                        validation_passed=True
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def extract_license_candidates(self, text: str) -> List[ExtractionCandidate]:
        """Extract state license candidates"""
        candidates = []
        
        for i, pattern in enumerate(self.license_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                license_num = match.group(1) if match.groups() else match.group(0)
                
                # Validate format (letter followed by digits)
                if re.match(r'^[A-Z]\d{5,6}$', license_num.upper()):
                    candidate = ExtractionCandidate(
                        value=license_num.upper(),
                        confidence=0.9 if 'license' in pattern.lower() else 0.7,
                        extractor_id=f"license_pattern_{i}",
                        position=match.start(),
                        context=self._get_context(text, match.start(), 20),
                        validation_passed=True
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def extract_date_candidates(self, text: str) -> List[ExtractionCandidate]:
        """Extract date candidates with multi-format parsing"""
        candidates = []
        
        for i, pattern in enumerate(self.date_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                date_str = match.group(1) if match.groups() else match.group(0)
                
                # Normalize date format
                normalized_date = self._normalize_date(date_str)
                
                if normalized_date:
                    confidence = 0.9 if i < 4 else 0.6  # Higher confidence for labeled dates
                    
                    candidate = ExtractionCandidate(
                        value=normalized_date,
                        confidence=confidence,
                        extractor_id=f"date_pattern_{i}",
                        position=match.start(),
                        context=self._get_context(text, match.start(), 30),
                        validation_passed=True
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def extract_transaction_type_candidates(self, text: str) -> List[ExtractionCandidate]:
        """Extract transaction type with negation guard"""
        candidates = []
        
        # First check for negation patterns
        text_lower = text.lower()
        has_negation = any(re.search(pattern, text_lower) for pattern in self.negation_patterns)
        
        if has_negation:
            # If negation detected, return low confidence or skip
            return candidates
        
        # Score each transaction type
        type_scores = {}
        
        for trans_type, patterns in self.transaction_patterns.items():
            score = 0
            positions = []
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                score += len(matches)
                positions.extend([match.start() for match in matches])
            
            if score > 0:
                type_scores[trans_type] = (score, positions)
        
        # Return the highest scoring type
        if type_scores:
            best_type = max(type_scores.keys(), key=lambda x: type_scores[x][0])
            score, positions = type_scores[best_type]
            
            candidate = ExtractionCandidate(
                value=best_type.title(),
                confidence=min(score * 0.3, 0.9),  # Cap confidence
                extractor_id="transaction_lexicon",
                position=positions[0] if positions else 0,
                context=self._get_context(text, positions[0], 50) if positions else text[:50],
                validation_passed=True
            )
            candidates.append(candidate)
        
        return candidates
    
    def _validate_npi_luhn(self, npi: str) -> bool:
        """Validate NPI using Luhn algorithm with 80840 prefix"""
        if len(npi) != 10:
            return False
        
        # Add 80840 prefix for Luhn check
        full_number = "80840" + npi[:-1]
        check_digit = int(npi[-1])
        
        # Luhn algorithm
        total = 0
        for i, digit in enumerate(reversed(full_number)):
            n = int(digit)
            if i % 2 == 1:  # Every second digit from right
                n *= 2
                if n > 9:
                    n = n - 9
            total += n
        
        calculated_check = (10 - (total % 10)) % 10
        return calculated_check == check_digit
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date to MM/DD/YYYY format"""
        # Remove extra whitespace
        date_str = date_str.strip()
        
        # Try different date formats
        date_patterns = [
            r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})',  # MM/DD/YY or MM/DD/YYYY
            r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})',    # YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            match = re.match(pattern, date_str)
            if match:
                part1, part2, part3 = match.groups()
                
                # Determine if it's MM/DD/YYYY or YYYY/MM/DD
                if len(part1) == 4:  # YYYY/MM/DD format
                    year, month, day = part1, part2, part3
                else:  # MM/DD/YY or MM/DD/YYYY format
                    month, day, year = part1, part2, part3
                
                # Convert 2-digit year to 4-digit
                if len(year) == 2:
                    year = "20" + year if int(year) < 50 else "19" + year
                
                # Validate ranges
                try:
                    month_int = int(month)
                    day_int = int(day)
                    year_int = int(year)
                    
                    if 1 <= month_int <= 12 and 1 <= day_int <= 31 and 1900 <= year_int <= 2100:
                        return f"{month_int:02d}/{day_int:02d}/{year_int}"
                except ValueError:
                    continue
        
        return None
    
    def _get_context(self, text: str, position: int, window: int = 20) -> str:
        """Get context around a match position"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].replace('\n', ' ').strip()