"""
Hybrid Extraction Engine: Combines deterministic + ML extractors with fusion
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from ..ingest.eml_parser import ParsedContent
from ..sectioner.block_split import BlockSectioner, ProviderBlock
from .patterns import PatternExtractor, ExtractionCandidate
from .ner import NERExtractor
from .tables import TableExtractor


@dataclass
class FieldResult:
    """Final result for a field after fusion"""
    value: str
    confidence: float
    extractor_id: str
    source_block: Optional[int] = None
    candidates: List[ExtractionCandidate] = None
    
    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []


class ExtractionEngine:
    """
    Hybrid extraction engine that combines multiple extraction methods
    with candidate scoring and fusion
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize extractors
        self.sectioner = BlockSectioner()
        self.pattern_extractor = PatternExtractor()
        self.ner_extractor = NERExtractor()
        self.table_extractor = TableExtractor()
        
        self.output_fields = [
            'Transaction Type (Add/Update/Term)',
            'Transaction Attribute',
            'Effective Date',
            'Term Date',
            'Term Reason',
            'Provider Name',
            'Provider NPI',
            'Provider Specialty',
            'State License',
            'Organization Name',
            'TIN',
            'Group NPI',
            'Complete Address',
            'Phone Number',
            'Fax Number',
            'PPG ID',
            'Line Of Business (Medicare/Commercial/Medical)'
        ]
        
        self.extractor_priorities = {
            'table_': 100,  
            'npi_pattern_': 90,  
            'tin_pattern_': 90,
            'phone_pattern_': 90,
            'fax_pattern_': 90,
            'license_pattern_': 90,
            'date_pattern_': 85,
            'ppg_pattern_': 85,
            'transaction_lexicon': 85,
            'spacy_': 70, 
            'domain_': 80,  
            'specialty_gazetteer': 80,
            'lob_gazetteer': 85,
        }
    
    def extract_all_fields(self, parsed_content: ParsedContent) -> List[Dict[str, str]]:
        """
        Main extraction method - extract all fields from parsed content
        Returns list of provider records (one per detected block)
        """
        blocks = self.sectioner.section_content(parsed_content.normalized_text)
        
        if not blocks:
            single_block = ProviderBlock(
                text=parsed_content.normalized_text,
                start_line=0,
                end_line=len(parsed_content.normalized_text.split('\n')),
                confidence=0.5
            )
            blocks = [single_block]
        
        results = []
        for i, block in enumerate(blocks):
            result = self._extract_from_block_smart(block, i, parsed_content)
            results.append(result)
        
        if not results:
            results = [self._create_empty_result()]
        
        return results
    
    
    
    def _extract_from_block_smart(
        self, 
        block: ProviderBlock, 
        block_idx: int,
        parsed_content: ParsedContent
    ) -> Dict[str, str]:
        """
        Smart extraction that uses optimal source prioritization to avoid redundancy.
        """
        output = {}
        for field in self.output_fields:
            output[field] = "Information not found"
        
        table_candidates = self._extract_from_tables_block_aware(block, parsed_content)
        
        output['Transaction Type (Add/Update/Term)'] = self._extract_transaction_type_smart(parsed_content.normalized_text)
        
        output['Provider Name'] = self._extract_provider_name_smart(table_candidates, block.text, parsed_content.normalized_text)
        
        output['Provider NPI'] = self._extract_npi_smart(table_candidates, parsed_content.normalized_text, block.text)
        
        output['TIN'] = self._extract_tin_smart(parsed_content.normalized_text, table_candidates, block.text)
        
        effective_date, term_date = self._extract_dates_smart(table_candidates, parsed_content.normalized_text, block.text, output['Transaction Type (Add/Update/Term)'])
        output['Effective Date'] = effective_date
        output['Term Date'] = term_date
        
        if output['Transaction Type (Add/Update/Term)'].lower() == 'term':
            output['Term Reason'] = self._extract_term_reason_smart(parsed_content.normalized_text, table_candidates, block.text)
        
        output['Provider Specialty'] = self._extract_specialty_smart(table_candidates, block.text, parsed_content.normalized_text)
        
        output['Organization Name'] = self._extract_organization_smart(parsed_content.normalized_text, block.text, table_candidates)
        
        output['PPG ID'] = self._extract_ppg_smart(parsed_content.normalized_text, table_candidates, block.text)
        
        output['Phone Number'] = self._extract_phone_smart(block.text, parsed_content.normalized_text)
        output['Fax Number'] = self._extract_fax_smart(table_candidates, block.text, parsed_content.normalized_text)
        
        output['State License'] = self._extract_license_smart(block.text, parsed_content.normalized_text, table_candidates)
        
        output['Line Of Business (Medicare/Commercial/Medical)'] = self._extract_lob_smart(parsed_content.normalized_text, block.text)
        
        full_text = parsed_content.text_content + '\n' + parsed_content.normalized_text
        output['Transaction Attribute'] = self._extract_transaction_attribute_smart(output['Transaction Type (Add/Update/Term)'], full_text)
        
        output['Complete Address'] = self._extract_address_smart(table_candidates, block.text, parsed_content.normalized_text)
        
        output['Group NPI'] = self._extract_group_npi_smart(table_candidates, parsed_content.normalized_text, block.text)
        
        return output
    
    
    
    
    def _extract_from_tables(self, parsed_content: ParsedContent) -> Dict[str, List[ExtractionCandidate]]:
        candidates = {}
        
        if parsed_content.html_content:
            tables = self.table_extractor.extract_from_html_table(parsed_content.html_content)
            if tables:
                table_candidates = self.table_extractor.extract_candidates_from_tables(tables)
                candidates.update(table_candidates)
        
        text_tables = self.table_extractor.extract_from_text_table(parsed_content.normalized_text)
        if text_tables:
            text_table_candidates = self.table_extractor.extract_candidates_from_tables(text_tables)
            for field, field_candidates in text_table_candidates.items():
                if field not in candidates:
                    candidates[field] = []
                candidates[field].extend(field_candidates)
        
        return candidates
    
    def _extract_from_tables_block_aware(self, block: ProviderBlock, parsed_content: ParsedContent) -> Dict[str, List[ExtractionCandidate]]:
        """Extract candidates from tables, but filter to only include data from the current block"""
        candidates = {}
        
        all_candidates = self._extract_from_tables(parsed_content)
        
        block_text_lower = block.text.lower()
        
        for field, field_candidates in all_candidates.items():
            block_specific_candidates = []
            
            for candidate in field_candidates:
                candidate_value = candidate.value.lower()
                
                if candidate_value in block_text_lower:
                    block_specific_candidates.append(candidate)
                elif not self._is_provider_specific_field(field):
                    candidate_copy = ExtractionCandidate(
                        value=candidate.value,
                        confidence=candidate.confidence * 0.6,
                        extractor_id=candidate.extractor_id,
                        position=candidate.position,
                        context=candidate.context,
                        validation_passed=candidate.validation_passed
                    )
                    block_specific_candidates.append(candidate_copy)
            
            if block_specific_candidates:
                candidates[field] = block_specific_candidates
        
        return candidates
    
    def _is_provider_specific_field(self, field: str) -> bool:
        """Determine if a field is provider-specific (should be unique per provider block)"""
        provider_specific_fields = {
            'provider_name', 'npi', 'specialty', 'license', 'phone', 'fax', 
            'ppg', 'organization'
        }
        return field in provider_specific_fields
    
    
    
    
    
    
    def _map_lob_to_canonical(self, lob: str) -> str:
        """Map LOB variant to canonical form (copied from NER extractor)"""
        lob_lower = lob.lower()
        
        if any(x in lob_lower for x in ['medicare', 'part a', 'part b', 'part c', 'part d']):
            return 'Medicare'
        elif any(x in lob_lower for x in ['medicaid', 'medi-cal']):
            return 'Medicaid'
        elif any(x in lob_lower for x in ['commercial', 'hmo', 'ppo', 'epo', 'pos', 'exchange']):
            return 'Commercial'
        else:
            return lob.title()
    
    def _extract_transaction_attribute(self, text: str) -> str:
        """
        Extract what specific attribute is being updated using contextual analysis
        Returns exactly one of: Specialty, Provider, Address, PPG, Phone Number, LOB
        For Add/Term defaults to 'Provider' unless specific attribute mentioned
        """
        return self._analyze_transaction_attribute_context(text)
    
    def _extract_explicit_transaction_attribute(self, text: str) -> Optional[str]:
        """Extract explicitly stated transaction attributes from email content"""
        text_lower = text.lower()
        
        patterns = [
            r'transaction\s+attribute\s*:\s*([^\n\r]+)',
            r'attribute\s*:\s*([^\n\r]+)',
            r'changed\s+attribute\s*:\s*([^\n\r]+)',
            r'update\s+type\s*:\s*([^\n\r]+)'
        ]
        
        for pattern in patterns:
            import re
            match = re.search(pattern, text_lower)
            if match:
                attr_value = match.group(1).strip()
                
                attr_mappings = {
                    'not applicable': 'Not Applicable',
                    'n/a': 'Not Applicable', 
                    'na': 'Not Applicable',
                    'none': 'Not Applicable',
                    'address': 'Address',
                    'location': 'Address',
                    'specialty': 'Specialty',
                    'specialization': 'Specialty',
                    'phone': 'Phone Number',
                    'phone number': 'Phone Number',
                    'telephone': 'Phone Number',
                    'contact': 'Phone Number',
                    'ppg': 'PPG',
                    'ppg id': 'PPG',
                    'practice group': 'PPG',
                    'lob': 'LOB',
                    'line of business': 'LOB',
                    'network': 'LOB',
                    'provider': 'Provider',
                    'general': 'Provider',
                    'demographic': 'Provider'
                }
                
                for key, standard_value in attr_mappings.items():
                    if key in attr_value:
                        return standard_value
                
                if len(attr_value) < 50:  
                    return attr_value.title()
        
        return None
    
    def _analyze_transaction_attribute_context(self, text: str) -> str:
        """
        Advanced contextual analysis to determine which attribute is being changed
        Uses weighted scoring and context patterns to identify the primary attribute
        """
        text_lower = text.lower()
        
        attribute_scores = {
            'Address': 0,
            'Specialty': 0, 
            'Phone Number': 0,
            'PPG': 0,
            'LOB': 0,
            'Provider': 0
        }
        
        context_patterns = {
            'Address': {
                'explicit': [  
                    ('address change', 3.0), ('address update', 3.0), ('location change', 3.0),
                    ('office change', 3.0), ('practice location', 2.8), ('new address', 2.5),
                    ('relocate', 2.5), ('relocation', 2.5), ('move', 2.3), ('transfer', 2.0)
                ],
                'contextual': [ 
                    ('address', 1.5), ('location', 1.2), ('street', 1.8), ('suite', 1.8),
                    ('building', 1.5), ('zip', 1.8), ('city', 1.0), ('state', 0.8)
                ],
                'subjects': [  
                    (r'address.*change', 2.5), (r'location.*update', 2.5), (r'move.*office', 2.0)
                ]
            },
            'Specialty': {
                'explicit': [
                    ('specialty change', 3.0), ('specialty update', 3.0), ('specialization change', 2.8),
                    ('practice change', 2.5), ('field change', 2.5), ('new specialty', 2.3)
                ],
                'contextual': [
                    ('specialty', 1.8), ('specialization', 1.5), ('practice area', 1.5),
                    ('medical field', 1.3), ('discipline', 1.2), ('board certified', 1.0)
                ],
                'subjects': [
                    (r'specialty.*change', 2.5), (r'practice.*update', 2.0)
                ]
            },
            'Phone Number': {
                'explicit': [
                    ('phone change', 3.0), ('phone update', 3.0), ('contact change', 2.8),
                    ('phone number change', 3.2), ('telephone change', 3.0), ('fax change', 2.8),
                    ('contact update', 2.5), ('new phone', 2.3), ('new contact', 2.0)
                ],
                'contextual': [
                    ('phone', 1.8), ('telephone', 1.5), ('fax', 1.5), ('contact', 1.2),
                    ('number', 1.0)
                ],
                'subjects': [
                    (r'phone.*change', 2.5), (r'contact.*update', 2.5), (r'fax.*change', 2.5)
                ]
            },
            'PPG': {
                'explicit': [
                    ('ppg change', 3.0), ('ppg update', 3.0), ('group change', 2.5),
                    ('practice group change', 3.2), ('ppg id change', 3.0), ('new ppg', 2.3)
                ],
                'contextual': [
                    ('ppg', 2.0), ('practice group', 1.8), ('group id', 1.5), ('ppg id', 2.0)
                ],
                'subjects': [
                    (r'ppg.*change', 2.5), (r'group.*update', 2.0)
                ]
            },
            'LOB': {
                'explicit': [
                    ('lob change', 3.0), ('network change', 2.8), ('line of business change', 3.2),
                    ('plan change', 2.5), ('coverage change', 2.5), ('insurance change', 2.3)
                ],
                'contextual': [
                    ('line of business', 2.0), ('lob', 1.8), ('network', 1.5), ('medicare', 1.3),
                    ('commercial', 1.3), ('medicaid', 1.3), ('insurance', 1.2), ('plan', 1.0)
                ],
                'subjects': [
                    (r'lob.*change', 2.5), (r'network.*update', 2.5), (r'plan.*change', 2.0)
                ]
            },
            'Provider': {
                'explicit': [ 
                    ('provider change', 2.0), ('provider update', 2.0), ('provider information', 1.8),
                    ('demographic change', 2.2), ('information update', 1.5)
                ],
                'contextual': [
                    ('provider', 1.0), ('doctor', 0.8), ('physician', 0.8), ('demographic', 1.2),
                    ('information', 0.5), ('data', 0.5)
                ],
                'subjects': [
                    (r'provider.*update', 2.0), (r'information.*change', 1.5)
                ]
            }
        }
        
        for attribute, patterns in context_patterns.items():
            for phrase, weight in patterns['explicit']:
                if phrase in text_lower:
                    attribute_scores[attribute] += weight
            
            for phrase, weight in patterns['contextual']:
                if phrase in text_lower:
                    attribute_scores[attribute] += weight
            
            first_100_chars = text_lower[:100]
            for pattern, weight in patterns['subjects']:
                import re
                if re.search(pattern, first_100_chars):
                    attribute_scores[attribute] += weight * 1.2 
        
        top_scores = sorted(attribute_scores.items(), key=lambda x: x[1], reverse=True)
        
        if top_scores[0][1] > 0:
            primary_attribute = top_scores[0][0]
            primary_score = top_scores[0][1]
            
            if len(top_scores) > 1 and top_scores[1][1] > 0:
                secondary_attribute = top_scores[1][0]
                secondary_score = top_scores[1][1]
                
                if abs(primary_score - secondary_score) < 1.0:
                    primary_attribute = self._resolve_attribute_conflict(
                        text_lower, primary_attribute, secondary_attribute, primary_score, secondary_score
                    )
            
            return primary_attribute
        
        return 'Provider'
    
    def _resolve_attribute_conflict(self, text_lower: str, attr1: str, attr2: str, score1: float, score2: float) -> str:
        """Resolve conflicts when multiple attributes have similar scores"""
        
        strong_indicators = {
            'Address': ['street', 'suite', 'zip code', 'city', 'state', 'building', 'floor'],
            'Specialty': ['board certified', 'residency', 'fellowship', 'medical school', 'practice area'],
            'Phone Number': ['extension', 'ext', 'area code', 'toll free', 'direct line'],
            'PPG': ['group number', 'practice id', 'group code'],
            'LOB': ['effective date', 'coverage', 'eligibility', 'enrollment'],
            'Provider': ['credentials', 'license', 'certification', 'npi', 'name change']
        }
        
        score_adjustments = {attr1: 0, attr2: 0}
        
        for attr in [attr1, attr2]:
            if attr in strong_indicators:
                for indicator in strong_indicators[attr]:
                    if indicator in text_lower:
                        score_adjustments[attr] += 0.5
        
        final_score1 = score1 + score_adjustments[attr1]
        final_score2 = score2 + score_adjustments[attr2]
        
        return attr1 if final_score1 >= final_score2 else attr2
    
    def _extract_term_reason(self, text: str) -> str:
        """Enhanced termination reason extraction with comprehensive patterns"""
        text_lower = text.lower()
        
        reason_patterns = [
            (['voluntary', 'voluntarily', 'by choice', 'provider choice', 'own choice'], 'Voluntary'),
            (['retired', 'retirement', 'retiring', 'end of career'], 'Retired'),
            (['contract end', 'contract ended', 'contract expir', 'agreement end', 'term of contract'], 'Contract Ended'),
            (['non-renewal', 'not renewed', 'renewal denied'], 'Contract Not Renewed'),
            (['performance', 'quality concern', 'quality issue', 'disciplinary'], 'Performance Issues'),
            (['credentialing', 'credential', 'licensing issue', 'license problem'], 'Credentialing Issues'),
            (['relocat', 'relocation', 'moved', 'moving', 'geographic', 'out of area'], 'Relocation'),
            (['business', 'financial', 'practice sold', 'practice closed', 'consolidation'], 'Business Decision'),
            (['deceased', 'death', 'disability', 'unable to practice'], 'Death/Disability'),
            (['network change', 'panel', 'network restructur', 'plan change'], 'Network Changes'),
            (['involuntary', 'terminated', 'dismissal', 'termination'], 'Involuntary'),\
            (['administrative', 'clerical', 'other'], 'Other')
        ]
        
        for keywords, reason_label in reason_patterns:
            for keyword in keywords:
                if keyword in text_lower:
                    return reason_label
        
        reason_context_patterns = [
            r'reason[:\s]+([^,.\n]+)',
            r'term(?:ination)?\s+reason[:\s]+([^,.\n]+)', 
            r'due\s+to[:\s]+([^,.\n]+)',
            r'because\s+of[:\s]+([^,.\n]+)',
            r'result\s+of[:\s]+([^,.\n]+)'
        ]
        
        for pattern in reason_context_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                reason_text = match.group(1).strip()
                mapped_reason = self._map_reason_text(reason_text)
                if mapped_reason != "Information not found":
                    return mapped_reason
        
        return 'Information not found'
    
    def _map_reason_text(self, reason_text: str) -> str:
        """Map extracted reason text to standardized categories"""
        reason_lower = reason_text.lower()
        
        if any(word in reason_lower for word in ['voluntary', 'choice']):
            return 'Voluntary'
        elif any(word in reason_lower for word in ['retired', 'retirement']):
            return 'Retired'
        elif any(word in reason_lower for word in ['contract', 'agreement']):
            return 'Contract Ended'
        elif any(word in reason_lower for word in ['relocation', 'moved', 'moving']):
            return 'Relocation'
        elif any(word in reason_lower for word in ['performance', 'quality']):
            return 'Performance Issues'
        elif len(reason_text.strip()) > 2: 
            return reason_text.strip().title()
        else:
            return 'Information not found'

    
    def _extract_transaction_type_smart(self, full_text: str) -> str:
        """Extract transaction type from full email content only"""
        candidates = self.ner_extractor.extract_transaction_types(full_text)
        return candidates[0].value if candidates else "Information not found"
    
    def _extract_provider_name_smart(self, table_candidates: Dict, block_text: str, full_text: str) -> str:
        """Provider Name: Tables → Block NER → Email patterns"""
        if 'provider_name' in table_candidates and table_candidates['provider_name']:
            return table_candidates['provider_name'][0].value
        
        block_candidates = self.ner_extractor.extract_provider_names(block_text)
        if block_candidates:
            return block_candidates[0].value
        
        email_candidates = self.ner_extractor.extract_provider_names(full_text)
        return email_candidates[0].value if email_candidates else "Information not found"
    
    def _extract_npi_smart(self, table_candidates: Dict, full_text: str, block_text: str) -> str:
        """NPI: Tables → Email patterns → Block patterns"""
        if 'npi' in table_candidates and table_candidates['npi']:
            return table_candidates['npi'][0].value
        
        email_candidates = self.pattern_extractor.extract_npi_candidates(full_text)
        if email_candidates:
            return email_candidates[0].value
        
        block_candidates = self.pattern_extractor.extract_npi_candidates(block_text)
        return block_candidates[0].value if block_candidates else "Information not found"
    
    def _extract_tin_smart(self, full_text: str, table_candidates: Dict, block_text: str) -> str:
        """TIN: Email patterns → Tables → Block patterns"""
        email_candidates = self.pattern_extractor.extract_tin_candidates(full_text)
        if email_candidates:
            return email_candidates[0].value
        
        if 'tin' in table_candidates and table_candidates['tin']:
            return table_candidates['tin'][0].value
        
        block_candidates = self.pattern_extractor.extract_tin_candidates(block_text)
        return block_candidates[0].value if block_candidates else "Information not found"
    
    def _extract_dates_smart(self, table_candidates: Dict, full_text: str, block_text: str, transaction_type: str) -> tuple:
        """Dates: Tables → Email patterns → Block NER"""
        effective_date = "Information not found"
        term_date = "Information not found"
        
        if 'term_date' in table_candidates and table_candidates['term_date']:
            term_date = table_candidates['term_date'][0].value
        
        if 'dates' in table_candidates and table_candidates['dates'] and term_date == "Information not found":
            date_value = table_candidates['dates'][0].value
            if transaction_type.lower() == 'term':
                term_date = date_value
            else:
                effective_date = date_value
        
        if effective_date == "Information not found" and term_date == "Information not found":
            email_pattern_candidates = self.pattern_extractor.extract_date_candidates(full_text)
            email_ner_candidates = self.ner_extractor.extract_dates(full_text)
            
            all_email_candidates = email_pattern_candidates + email_ner_candidates
            if all_email_candidates:
                best_candidate = max(all_email_candidates, key=lambda x: x.confidence)
                date_value = best_candidate.value
                
                if transaction_type.lower() == 'term':
                    term_date = date_value
                else:
                    effective_date = date_value
        
        if effective_date == "Information not found" and term_date == "Information not found":
            block_candidates = self.ner_extractor.extract_dates(block_text)
            if block_candidates:
                date_value = block_candidates[0].value
                if transaction_type.lower() == 'term':
                    term_date = date_value
                else:
                    effective_date = date_value
        
        return effective_date, term_date
    
    def _extract_term_reason_smart(self, full_text: str, table_candidates: Dict, block_text: str) -> str:
        """Term Reason: Email patterns → Tables → Block patterns"""
        email_reason = self._extract_term_reason(full_text)
        if email_reason != "Information not found":
            return email_reason
        
        if 'term_reason' in table_candidates and table_candidates['term_reason']:
            return table_candidates['term_reason'][0].value
        
        return self._extract_term_reason(block_text)
    
    def _extract_specialty_smart(self, table_candidates: Dict, block_text: str, full_text: str) -> str:
        """Specialties: Tables → Block NER → Email patterns"""
        if 'specialty' in table_candidates and table_candidates['specialty']:
            return table_candidates['specialty'][0].value
        
        block_candidates = self.ner_extractor.extract_specialties(block_text)
        if block_candidates:
            return block_candidates[0].value
        
        email_candidates = self.ner_extractor.extract_specialties(full_text)
        return email_candidates[0].value if email_candidates else "Information not found"
    
    def _extract_organization_smart(self, full_text: str, block_text: str, table_candidates: Dict) -> str:
        """Organizations: Email patterns → Block NER → Tables"""
        email_candidates = self.ner_extractor.extract_organizations(full_text)
        if email_candidates:
            return email_candidates[0].value
        
        block_candidates = self.ner_extractor.extract_organizations(block_text)
        if block_candidates:
            return block_candidates[0].value
        
        if 'organization' in table_candidates and table_candidates['organization']:
            return table_candidates['organization'][0].value
        
        return "Information not found"
    
    def _extract_ppg_smart(self, full_text: str, table_candidates: Dict, block_text: str) -> str:
        """PPG: Email patterns → Tables → Block patterns"""
        email_candidates = self.pattern_extractor.extract_ppg_candidates(full_text)
        if email_candidates:
            return email_candidates[0].value
        
        if 'ppg' in table_candidates and table_candidates['ppg']:
            return table_candidates['ppg'][0].value
        
        block_candidates = self.pattern_extractor.extract_ppg_candidates(block_text)
        return block_candidates[0].value if block_candidates else "Information not found"
    
    def _extract_phone_smart(self, block_text: str, full_text: str) -> str:
        """Phone: Block patterns → Email patterns"""
        block_candidates = self.pattern_extractor.extract_phone_candidates(block_text)
        if block_candidates:
            return block_candidates[0].value
        
        email_candidates = self.pattern_extractor.extract_phone_candidates(full_text)
        return email_candidates[0].value if email_candidates else "Information not found"
    
    def _extract_fax_smart(self, table_candidates: Dict, block_text: str, full_text: str) -> str:
        """Fax: Tables → Block patterns → Email patterns"""
        if 'fax' in table_candidates and table_candidates['fax']:
            return table_candidates['fax'][0].value
        
        block_candidates = self.pattern_extractor.extract_fax_candidates(block_text)
        if block_candidates:
            return block_candidates[0].value
        
        email_candidates = self.pattern_extractor.extract_fax_candidates(full_text)
        return email_candidates[0].value if email_candidates else "Information not found"
    
    def _extract_license_smart(self, block_text: str, full_text: str, table_candidates: Dict) -> str:
        """License: Block patterns → Email patterns → Tables"""
        block_candidates = self.pattern_extractor.extract_license_candidates(block_text)
        if block_candidates:
            return block_candidates[0].value
        
        email_candidates = self.pattern_extractor.extract_license_candidates(full_text)
        if email_candidates:
            return email_candidates[0].value
        
        if 'license' in table_candidates and table_candidates['license']:
            return table_candidates['license'][0].value
        
        return "Information not found"
    
    def _extract_lob_smart(self, full_text: str, block_text: str) -> str:
        """Line of Business: Email patterns → Block NER"""
        email_candidates = self.ner_extractor.extract_line_of_business(full_text)
        if email_candidates:
            lobs = [c.value for c in email_candidates]
            return ", ".join(lobs)
        
        block_candidates = self.ner_extractor.extract_line_of_business(block_text)
        if block_candidates:
            lobs = [c.value for c in block_candidates]
            return ", ".join(lobs)
        
        return "Information not found"
    
    def _extract_transaction_attribute_smart(self, transaction_type: str, full_text: str) -> str:
        """Transaction Attribute based on transaction type and context"""
        explicit_attr = self._extract_explicit_transaction_attribute(full_text)
        if explicit_attr:
            return explicit_attr
        
        transaction_type_lower = transaction_type.lower()
        
        if transaction_type_lower == 'term':
            full_text_lower = full_text.lower()
            
            explicit_attr_terminations = [
                'address termination', 'phone termination', 'ppg termination', 
                'lob termination', 'terminate address only', 'terminate phone only',
                'terminate ppg only', 'terminate lob only', 'specialty termination'
            ]
            
            for phrase in explicit_attr_terminations:
                if phrase in full_text_lower:
                    if 'address' in phrase:
                        return 'Address'
                    elif 'specialty' in phrase:
                        return 'Specialty'
                    elif 'phone' in phrase:
                        return 'Phone Number'
                    elif 'ppg' in phrase:
                        return 'PPG'
                    elif 'lob' in phrase:
                        return 'LOB'
            
            return 'Provider'
            
        elif transaction_type_lower == 'add':
            full_text_lower = full_text.lower()
            
            explicit_attr_additions = [
                'add new address to', 'add new specialty to', 'add new phone to',
                'add new ppg to', 'add new lob to', 'include new address in',
                'include new specialty in', 'include new phone in'
            ]
            
            for phrase in explicit_attr_additions:
                if phrase in full_text_lower:
                    if 'address' in phrase:
                        return 'Address'
                    elif 'specialty' in phrase:
                        return 'Specialty'
                    elif 'phone' in phrase:
                        return 'Phone Number'
                    elif 'ppg' in phrase:
                        return 'PPG'
                    elif 'lob' in phrase:
                        return 'LOB'
            
            return 'Provider'
        elif transaction_type_lower == 'update':
            return self._extract_transaction_attribute(full_text)
        else:
            return "Not Applicable"
    
    def _extract_address_smart(self, table_candidates: Dict, block_text: str, full_text: str) -> str:
        if 'address' in table_candidates and table_candidates['address']:
            return table_candidates['address'][0].value
        
        return "Information not found"
    
    def _extract_group_npi_smart(self, table_candidates: Dict, full_text: str, block_text: str) -> str:
        """Group NPI: Tables → Email patterns → Block patterns"""
        if 'npi' in table_candidates and len(table_candidates['npi']) > 1:
            return table_candidates['npi'][1].value
        
        email_candidates = self.pattern_extractor.extract_npi_candidates(full_text)
        if len(email_candidates) > 1:
            return email_candidates[1].value
        
        block_candidates = self.pattern_extractor.extract_npi_candidates(block_text)
        if len(block_candidates) > 1:
            return block_candidates[1].value
        
        return "Information not found"
    
    def _create_empty_result(self) -> Dict[str, str]:
        """Create empty result with all fields set to 'Information not found'"""
        result = {}
        for field in self.output_fields:
            result[field] = "Information not found"
        return result