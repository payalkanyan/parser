import re
from typing import Dict, List, Optional
import logging
import yaml
from pathlib import Path
import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English
from .patterns import ExtractionCandidate
from dateutil import parser as date_parser

# Optional spaCy import with graceful fallback
try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.lang.en import English
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

class NERExtractor:
    """
    Local ML/NER using spaCy with domain vocabulary
    Provides fuzzy extraction for names, organizations, specialties
    """
    
    def __init__(self):
        global HAS_SPACY
        self.logger = logging.getLogger(__name__)
        self.nlp = None
        self.matcher = None
        
        if HAS_SPACY:
            try:
                try:
                    self.nlp = spacy.load("en_core_web_trf")
                    self.logger.info("Loaded spaCy transformer model")
                except OSError:
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                        self.logger.info("Loaded spaCy small model")
                    except OSError:
                        self.logger.warning("No spaCy model available, using basic English")
                        self.nlp = English()
                        
                self.matcher = Matcher(self.nlp.vocab)
                self._setup_domain_patterns()
                
            except Exception as e:
                self.logger.error(f"Failed to initialize spaCy: {e}")
                HAS_SPACY = False
        
        self.specialties_config = self._load_specialties_config()
        self.medical_specialties = list(self.specialties_config.keys()) if self.specialties_config else []
        
        self.specialty_synonyms = self._build_specialty_synonyms()
          
        self.taxonomy_codes = self._build_taxonomy_mappings()
        
        self.organization_aliases = [
            "Medical Group", "Healthcare", "Clinic", "Practice", "Associates",
            "Physicians", "Hospital", "Health System", "Medical Center"
        ]
        
        self.lob_variants = [
            "Medicare", "Medicaid", "Commercial", "HMO", "PPO", "EPO", "POS",
            "Exchange", "Medi-Cal", "Part A", "Part B", "Part C", "Part D",
            "Advantage", "Supplement"
        ]
    
    def _setup_domain_patterns(self):
        """Setup domain-specific patterns for the matcher"""
        if not self.matcher:
            return
        
        specialty_patterns = []
        for specialty in self.medical_specialties:
            words = specialty.split()
            if len(words) == 1:
                specialty_patterns.append([{"LOWER": words[0].lower()}])
            else:
                pattern = [{"LOWER": word.lower()} for word in words]
                specialty_patterns.append(pattern)
        
        if specialty_patterns:
            self.matcher.add("MEDICAL_SPECIALTY", specialty_patterns)
        
        org_patterns = []
        for alias in self.organization_aliases:
            words = alias.split()
            if len(words) == 1:
                org_patterns.append([{"LOWER": words[0].lower()}])
            else:
                pattern = [{"LOWER": word.lower()} for word in words]
                org_patterns.append(pattern)
        
        if org_patterns:
            self.matcher.add("HEALTHCARE_ORG", org_patterns)
        
        lob_patterns = []
        for lob in self.lob_variants:
            lob_patterns.append([{"LOWER": lob.lower()}])
        
        if lob_patterns:
            self.matcher.add("LINE_OF_BUSINESS", lob_patterns)
        
        title_patterns = [
            [{"LOWER": "dr"}, {"IS_PUNCT": True, "OP": "?"}],
            [{"LOWER": "doctor"}],
            [{"LOWER": "physician"}],
            [{"LOWER": "md"}],
            [{"LOWER": "do"}],
        ]
        self.matcher.add("PROVIDER_TITLE", title_patterns)
    
    def extract_provider_names(self, text: str) -> List[ExtractionCandidate]:
        """Extract provider names using NER + patterns"""
        candidates = []
        
        if not HAS_SPACY or not self.nlp:
            return self._extract_names_fallback(text)
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    if self._is_likely_provider_name(ent.text):
                        candidate = ExtractionCandidate(
                            value=self._normalize_name(ent.text),
                            confidence=0.7,
                            extractor_id="spacy_person",
                            position=ent.start_char,
                            context=self._get_surrounding_context(doc, ent),
                            validation_passed=True
                        )
                        candidates.append(candidate)
            
            if self.matcher:
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id]
                    
                    if label == "PROVIDER_TITLE":
                        name_candidates = self._find_names_near_title(doc, start, end)
                        candidates.extend(name_candidates)
        
        except Exception as e:
            self.logger.error(f"NER extraction failed: {e}")
            return self._extract_names_fallback(text)
        
        return candidates
    
    def extract_organizations(self, text: str) -> List[ExtractionCandidate]:
        """
        Organization extraction restricted to body content and tables using NER only
        """
        candidates = []
        
        body_content = self._extract_body_content(text)
        
        if body_content and HAS_SPACY and self.nlp:
            candidates = self._extract_organizations_from_body(body_content)
        
        if not candidates:
            table_content = self._extract_table_content(text)
            if table_content:
                candidates = self._extract_organizations_from_body(table_content)
        
        return candidates
    
    def _extract_body_content(self, text: str) -> str:
        """Extract main body content excluding salutation and closing notes"""
        lines = text.split('\n')
        
        body_start = 0
        salutation_patterns = [
            r'^dear\s+', r'^hi\s*,?$', r'^hello\s*,?$', r'^greetings\s*,?$',
            r'^to\s+whom', r'^attention', r'^regarding'
        ]
        
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            if line_lower and not any(re.match(pattern, line_lower) for pattern in salutation_patterns):
                if not re.match(r'^(from|to|subject|date|received):', line_lower) and \
                   not re.match(r'^[a-z\s,&]+:$', line_lower):
                    body_start = i
                    break
        
        body_end = len(lines)
        closing_patterns = [
            r'^best\s+(regards?|wishes)', r'^sincerely', r'^thank\s+you',
            r'^regards?$', r'^thanks?$', r'^cheers?$', r'^yours?\s+',
            r'^respectfully', r'^cordially'
        ]
        
        for i in range(len(lines) - 1, body_start, -1):
            line_lower = lines[i].strip().lower()
            if line_lower and any(re.match(pattern, line_lower) for pattern in closing_patterns):
                body_end = i
                break
        
        body_lines = lines[body_start:body_end]
        
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        
        return '\n'.join(body_lines)
    
    def _extract_table_content(self, text: str) -> str:
        """Extract content that appears to be in table format"""
        lines = text.split('\n')
        table_lines = []
        
        for line in lines:
            if '|' in line and line.count('|') >= 2:
                table_lines.append(line)
            elif re.search(r'\s+\|\s+', line) or re.search(r'\w+:\s*\w+\s*\|\s*\w+:', line):
                table_lines.append(line)
        
        return '\n'.join(table_lines)
    
    def _extract_organizations_from_body(self, text: str) -> List[ExtractionCandidate]:
        """Extract organizations from body/table text using NER only"""
        candidates = []
        
        if not HAS_SPACY or not self.nlp:
            return candidates
        
        try:
            cleaned_text = self._clean_text_for_ner(text)
            doc = self.nlp(cleaned_text)
            
            all_candidates = []
            
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    org_name = ent.text.strip()
                    
                    if not self._is_healthcare_related(org_name):
                        continue
                    
                    if self._is_health_plan_organization(org_name):
                        continue
                    
                    expanded_org = self._expand_organization_name(doc, ent, cleaned_text)
                    if expanded_org:
                        if self._is_healthcare_related(expanded_org):
                            org_name = expanded_org
                    
                    context = self._get_surrounding_context(doc, ent)
                    
                    confidence = self._calculate_org_confidence(org_name, context)
                    
                    candidate = ExtractionCandidate(
                        value=self._normalize_org_name(org_name),
                        confidence=confidence,
                        extractor_id="body_ner_org",
                        position=ent.start_char,
                        context=context,
                        validation_passed=True
                    )
                    all_candidates.append(candidate)
            
            candidates = self._filter_best_organizations(all_candidates)
        
        except Exception as e:
            self.logger.error(f"Body NER organization extraction failed: {e}")
        
        return candidates
    
    def _filter_best_organizations(self, candidates: List[ExtractionCandidate]) -> List[ExtractionCandidate]:
        """Filter organization candidates to keep the most specific/complete ones"""
        if not candidates:
            return candidates
        
        candidates.sort(key=lambda x: len(x.value), reverse=True)
        
        filtered = []
        processed_names = set()
        
        for candidate in candidates:
            candidate_lower = candidate.value.lower()
            
            is_subset = False
            for processed_name in processed_names:
                if candidate_lower in processed_name or processed_name in candidate_lower:
                    if len(candidate_lower) > len(processed_name):
                        filtered = [c for c in filtered if c.value.lower() != processed_name]
                        processed_names.discard(processed_name)
                        break
                    else:
                        is_subset = True
                        break
            
            if not is_subset:
                filtered.append(candidate)
                processed_names.add(candidate_lower)
        
        if filtered:
            best_candidate = max(filtered, key=lambda x: x.confidence)
            return [best_candidate]
        
        return filtered
    
    def _clean_text_for_ner(self, text: str) -> str:
        """Clean text to improve NER recognition"""
        cleaned = re.sub(r'\n+', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        return cleaned
    
    def _expand_organization_name(self, doc, ent, original_text: str) -> Optional[str]:
        """Try to expand organization name to include adjacent parts like '& RCSSD'"""
        start_char = ent.start_char
        end_char = ent.end_char
        
        remaining_text = original_text[end_char:end_char + 20]
        
        expansion_match = re.match(r'\s*&\s+([A-Z][A-Z\s&]+)', remaining_text)
        if expansion_match:
            expansion = expansion_match.group(1).strip()
            if len(expansion) <= 10 and not any(word in expansion.lower() for word in ['the', 'and', 'or']):
                expanded = f"{ent.text} & {expansion}"
                return expanded
        
        return None
    
    def _is_healthcare_related(self, org_name: str) -> bool:
        """Check if organization name is healthcare-related"""
        if not org_name or len(org_name) < 3:
            return False
        
        org_lower = org_name.lower()
        
        healthcare_terms = [
            'medical', 'clinic', 'hospital', 'practice', 'physicians',
            'health', 'healthcare', 'group', 'associates', 'center'
        ]
        
        non_healthcare_terms = [
            'microsoft', 'google', 'email', 'outlook', 'exchange',
            'best regards', 'thank you', 'sincerely'
        ]
        
        if any(term in org_lower for term in non_healthcare_terms):
            return False
        
        if any(term in org_lower for term in healthcare_terms):
            return True
        
        if len(org_name) >= 8:
            return True
        
        if re.match(r'^[A-Z]+(\s*&\s*[A-Z]+)*$', org_name):
            return True
        
        return False
    
    def _calculate_org_confidence(self, org_name: str, context: str) -> float:
        """Calculate confidence score for organization based on name and context"""
        base_confidence = 0.7
        org_lower = org_name.lower()
        context_lower = context.lower()
        
        if any(term in org_lower for term in ['medical', 'clinic', 'hospital', 'practice']):
            base_confidence += 0.1
        
        if any(term in context_lower for term in ['terminated with', 'affiliated with', 'practices at']):
            base_confidence += 0.15
        
        if any(term in org_lower for term in ['group', 'associates', 'partners']):
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)
    
    
    def _is_bad_org_match(self, org_name: str) -> bool:
        """Filter out obviously bad organization name matches"""
        org_lower = org_name.lower()
        
        bad_patterns = [
            'provider', 'affiliation', 'network', 'best regards', 'hi', 'please',
            'terminate', 'effective', 'date', 'tax id', 'license', 'npi'
        ]
        
        return any(bad_pattern in org_lower for bad_pattern in bad_patterns)
    
    def _clean_provider_org_name(self, org_name: str) -> str:
        """Clean and normalize provider organization name"""
        if not org_name:
            return ""
        
        org_name = re.sub(r'^(the\s+)', '', org_name, flags=re.IGNORECASE)
        org_name = re.sub(r'\s+(effective|on|as\s+of).*$', '', org_name, flags=re.IGNORECASE)
        
        org_name = re.sub(r'[.,;:\s]+$', '', org_name)
        org_name = org_name.strip()
        
        skip_patterns = [
            r'^(effective|on|as\s+of|date|time)$',
            r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$',  # Dates
            r'^(january|february|march|april|may|june|july|august|september|october|november|december)$',
            r'^\w{1,2}$',  # Single/double letters
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, org_name, re.IGNORECASE):
                return ""
        
        return org_name
    
    def _classify_organization_context(self, context: str, org_name: str) -> str:
        """Classify whether organization is a provider, health plan, or unknown"""
        context_lower = context.lower()
        org_lower = org_name.lower()
        
        health_plan_indicators = [
            'insurance', 'health plan', 'hmo', 'ppo', 'epo', 'pos',
            'medicare', 'medicaid', 'coverage', 'benefits', 'plan',
            'payor', 'payer', 'insurer'
        ]
        
        provider_indicators = [
            'medical group', 'medical center', 'clinic', 'hospital', 'practice',
            'physicians', 'associates', 'health system', 'healthcare',
            'terminated with', 'employed by', 'affiliated with', 'works at',
            'practices at', 'joined', 'leaving'
        ]
        
        for indicator in provider_indicators:
            if indicator in context_lower:
                return "provider"
        
        provider_org_patterns = [
            'medical', 'clinic', 'hospital', 'practice', 'physicians', 
            'health center', 'medical center', 'group', 'associates'
        ]
        if any(word in org_lower for word in provider_org_patterns):
            return "provider"
        
        for indicator in health_plan_indicators:
            if indicator in context_lower or indicator in org_lower:
                return "health_plan"
        
        return "unknown"
    
    def _is_health_plan_organization(self, org_name: str) -> bool:
        """Check if organization name indicates a health plan rather than provider using generic patterns"""
        org_lower = org_name.lower()
        
        health_plan_patterns = [
            'insurance', 'health plan', 'hmo', 'ppo', 'epo', 'pos',
            'medicare', 'medicaid', 'coverage', 'benefits plan',
            'health net', 'health care plan', 'managed care'
        ]
        
        provider_override_patterns = [
            'medical group', 'medical center', 'clinic', 'hospital',
            'practice', 'physicians', 'health center', 'associates'
        ]
        
        if any(pattern in org_lower for pattern in provider_override_patterns):
            return False
        
        return any(pattern in org_lower for pattern in health_plan_patterns)
    
    
    
    def extract_specialties(self, text: str) -> List[ExtractionCandidate]:
        """
        Enhanced specialty extraction with YAML config, synonyms, taxonomy codes, and fuzzy matching
        """
        candidates = []
        found_specialties = set()
        
        candidates.extend(self._extract_specialties_by_synonyms(text, found_specialties))
        
        candidates.extend(self._extract_specialties_by_taxonomy_codes(text, found_specialties))
        
        candidates.extend(self._extract_specialties_by_fuzzy_matching(text, found_specialties))
        
        candidates.extend(self._extract_specialties_with_spacy_ner(text, found_specialties))
        
        candidates.extend(self._extract_specialties_exact_canonical(text, found_specialties))
        
        return candidates
    
    def _extract_specialties_by_synonyms(self, text: str, found_specialties: set) -> List[ExtractionCandidate]:
        """Extract specialties using comprehensive synonym matching"""
        candidates = []
        
        for synonym, canonical_name in self.specialty_synonyms.items():
            if canonical_name in found_specialties:
                continue
            
            if len(synonym) <= 2:
                patterns = [rf'\b{re.escape(synonym)}\b']
            else:
                patterns = [
                    rf'\b{re.escape(synonym)}\b',  # Word boundary match
                    rf'{re.escape(synonym)}(?=\s|$|,|\.)',  # End of phrase match
                    rf'(?:^|[:\s]){re.escape(synonym)}(?=\s|$|,|\.)'  # After colon/space match
                ]
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    match = matches[0]
                    
                    if len(synonym) <= 2:
                        context_before = text[max(0, match.start()-20):match.start()].lower()
                        context_after = text[match.end():match.end()+20].lower()
                        
                        if any(word in context_before + context_after for word in ['provider', 'deliver', 'other', 'over', 'under', 'after', 'never', 'number', 'management', 'different', 'treatment', 'department', 'agreement', 'statement']):
                            continue
                    
                    candidate = ExtractionCandidate(
                        value=canonical_name,
                        confidence=0.95 if len(synonym) > 2 else 0.7,
                        extractor_id="enhanced_synonym_match",
                        position=match.start(),
                        context=text[max(0, match.start()-30):match.end()+30],
                        validation_passed=True
                    )
                    candidates.append(candidate)
                    found_specialties.add(canonical_name)
                    break
        
        return candidates
    
    def _extract_specialties_by_taxonomy_codes(self, text: str, found_specialties: set) -> List[ExtractionCandidate]:
        """Extract specialties by recognizing taxonomy codes"""
        candidates = []
        
        for taxonomy_code, canonical_name in self.taxonomy_codes.items():
            if canonical_name in found_specialties:
                continue
                
            pattern = rf'\b{re.escape(taxonomy_code)}\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                candidate = ExtractionCandidate(
                    value=canonical_name,
                    confidence=0.9,
                    extractor_id="taxonomy_code_match",
                    position=match.start(),
                    context=text[max(0, match.start()-30):match.end()+30],
                    validation_passed=True
                )
                candidates.append(candidate)
                found_specialties.add(canonical_name)
                break
        
        return candidates
    
    def _extract_specialties_by_fuzzy_matching(self, text: str, found_specialties: set) -> List[ExtractionCandidate]:
        """Extract specialties using fuzzy matching for variations"""
        candidates = []
        
        try:
            from rapidfuzz import fuzz, process
            
            medical_keywords = ['medicine', 'surgery', 'ology', 'ics', 'ist', 'ian', 'specialty', 'field']
            potential_phrases = []
            
            for keyword in medical_keywords:
                pattern = rf'(\w+(?:\s+\w+)*\s+{keyword}|\w*{keyword}\w*(?:\s+\w+)*)'
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    phrase = match.group(0).strip()
                    if len(phrase) > 3:
                        potential_phrases.append((phrase, match.start()))
            
            specialty_context_patterns = [
                r'specialty[:\s]+([^,.\n]+)',
                r'field[:\s]+([^,.\n]+)',
                r'specialization[:\s]+([^,.\n]+)',
                r'area[:\s]+([^,.\n]+)',
                r'practice[:\s]+([^,.\n]+)'
            ]
            
            for pattern in specialty_context_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    phrase = match.group(1).strip()
                    if len(phrase) > 3 and phrase not in [p[0] for p in potential_phrases]:
                        potential_phrases.append((phrase, match.start()))
            
            for phrase, position in potential_phrases:
                best_match = process.extractOne(
                    phrase, 
                    self.medical_specialties, 
                    scorer=fuzz.ratio,
                    score_cutoff=80
                )
                
                if not best_match:
                    best_match = process.extractOne(
                        phrase, 
                        self.medical_specialties, 
                        scorer=fuzz.partial_ratio,
                        score_cutoff=85
                    )
                
                if best_match and best_match[0] not in found_specialties:
                    canonical_name = best_match[0]
                    similarity_score = best_match[1]
                    
                    if similarity_score >= 90:
                        confidence = 0.8
                    elif similarity_score >= 85:
                        confidence = 0.7
                    else:
                        confidence = 0.6
                    
                    candidate = ExtractionCandidate(
                        value=canonical_name,
                        confidence=confidence,
                        extractor_id="fuzzy_match",
                        position=position,
                        context=text[max(0, position-30):position+len(phrase)+30],
                        validation_passed=True
                    )
                    candidates.append(candidate)
                    found_specialties.add(canonical_name)
        
        except ImportError:
            self.logger.warning("rapidfuzz not available for fuzzy specialty matching")
        except Exception as e:
            self.logger.error(f"Fuzzy specialty matching failed: {e}")
        
        return candidates
    
    def _extract_specialties_with_spacy_ner(self, text: str, found_specialties: set) -> List[ExtractionCandidate]:
        """Extract specialties using spaCy NER with domain matcher"""
        candidates = []
        
        if not HAS_SPACY or not self.nlp or not self.matcher:
            return candidates
            
        try:
            doc = self.nlp(text)
            
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                label = self.nlp.vocab.strings[match_id]
                
                if label == "MEDICAL_SPECIALTY":
                    span = doc[start:end]
                    specialty_text = span.text
                    
                    canonical_name = self.specialty_synonyms.get(specialty_text.lower())
                    if not canonical_name:
                        canonical_name = specialty_text
                    
                    if canonical_name not in found_specialties:
                        candidate = ExtractionCandidate(
                            value=canonical_name,
                            confidence=0.85,
                            extractor_id="spacy_domain_ner",
                            position=span.start_char,
                            context=self._get_surrounding_context(doc, span),
                            validation_passed=True
                        )
                        candidates.append(candidate)
                        found_specialties.add(canonical_name)
        
        except Exception as e:
            self.logger.error(f"spaCy specialty NER failed: {e}")
        
        return candidates
    
    def _extract_specialties_exact_canonical(self, text: str, found_specialties: set) -> List[ExtractionCandidate]:
        """Fallback: exact matching on canonical specialty names"""
        candidates = []
        
        for specialty in self.medical_specialties:
            if specialty in found_specialties:
                continue
                
            pattern = rf'\b{re.escape(specialty)}\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                candidate = ExtractionCandidate(
                    value=specialty,
                    confidence=0.8,
                    extractor_id="canonical_exact_match",
                    position=match.start(),
                    context=text[max(0, match.start()-20):match.end()+20],
                    validation_passed=True
                )
                candidates.append(candidate)
                found_specialties.add(specialty)
                break
        
        return candidates
    
    def extract_line_of_business(self, text: str) -> List[ExtractionCandidate]:
        """Extract Line of Business mentions"""
        candidates = []
        found_lobs = set()
        
        for lob in self.lob_variants:
            pattern = r'\b' + re.escape(lob) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                canonical_lob = self._map_lob_to_canonical(lob)
                
                if canonical_lob not in found_lobs:
                    found_lobs.add(canonical_lob)
                    
                    candidate = ExtractionCandidate(
                        value=canonical_lob,
                        confidence=0.9,
                        extractor_id="lob_gazetteer",
                        position=match.start(),
                        context=text[max(0, match.start()-20):match.end()+20],
                        validation_passed=True
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _extract_dates_fallback(self, text: str) -> List[ExtractionCandidate]:
        """Fallback date extraction using regex patterns for word formats"""
        candidates = []
        
        months = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        
        month_pattern = '|'.join(months)
        date_patterns = [
            # "22 September 2025"
            rf'(\d{{1,2}})\s+({month_pattern})\s+(\d{{4}})',
            # "September 22, 2025" 
            rf'({month_pattern})\s+(\d{{1,2}}),?\s+(\d{{4}})',
            # "22nd September 2025"
            rf'(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern})\s+(\d{{4}})',
            # "September 2025" (month and year only)
            rf'({month_pattern})\s+(\d{{4}})'
        ]
        
        for i, pattern in enumerate(date_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                date_text = match.group(0)
                normalized_date = self._normalize_word_date(date_text)
                
                if normalized_date:
                    candidate = ExtractionCandidate(
                        value=normalized_date,
                        confidence=0.7,
                        extractor_id=f"date_fallback_{i}",
                        position=match.start(),
                        context=text[max(0, match.start()-20):match.end()+20],
                        validation_passed=True
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _normalize_word_date(self, date_str: str) -> Optional[str]:
        """Normalize word format dates to MM/DD/YYYY"""
        import re
        
        date_str = date_str.strip()
        
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
            'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
            'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        try:
            # Handle "Effective 22nd September 2025" format (extract date part)
            effective_match = re.search(r'effective\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})', date_str, re.IGNORECASE)
            if effective_match:
                date_str = effective_match.group(1)
            
            # "22 September 2025" or "22nd September 2025"
            match = re.match(r'(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)\s+(\d{4})', date_str, re.IGNORECASE)
            if match:
                day, month_name, year = match.groups()
                month = month_map.get(month_name.lower())
                if month:
                    return f"{month:02d}/{int(day):02d}/{year}"
            
            # "September 22, 2025"
            match = re.match(r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', date_str, re.IGNORECASE)
            if match:
                month_name, day, year = match.groups()
                month = month_map.get(month_name.lower())
                if month:
                    return f"{month:02d}/{int(day):02d}/{year}"
            
            # "September 2025" (assume 1st of month)
            match = re.match(r'(\w+)\s+(\d{4})', date_str, re.IGNORECASE)
            if match:
                month_name, year = match.groups()
                month = month_map.get(month_name.lower())
                if month:
                    return f"{month:02d}/01/{year}"
            
            parsed_date = date_parser.parse(date_str, fuzzy=True)
            return f"{parsed_date.month:02d}/{parsed_date.day:02d}/{parsed_date.year}"
                
        except Exception:
            pass
        
        return None
    
    def _load_specialties_config(self) -> Optional[Dict]:
        """Load medical specialties from YAML configuration file"""
        try:
            config_path = Path(__file__).parent.parent.parent / "configs" / "specialties.yml"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    return config.get('specialties', {})
            else:
                self.logger.warning(f"Specialties config file not found: {config_path}")
                return self._get_fallback_specialties()
                
        except Exception as e:
            self.logger.error(f"Failed to load specialties config: {e}")
            return self._get_fallback_specialties()
    
    def _get_fallback_specialties(self) -> Dict:
        """Fallback hardcoded specialties if YAML loading fails"""
        return {
            "Internal Medicine": {"synonyms": ["internal medicine", "internal med", "im", "internist"]},
            "Family Medicine": {"synonyms": ["family medicine", "family practice", "fp", "primary care"]},
            "Emergency Medicine": {"synonyms": ["emergency medicine", "emergency", "er", "em"]},
            "Pediatric Emergency Medicine": {"synonyms": ["pediatric emergency medicine", "pediatric emergency", "peds emergency"]},
            "Cardiology": {"synonyms": ["cardiology", "cardiac", "heart", "cardiovascular"]},
            "Dermatology": {"synonyms": ["dermatology", "derm", "skin"]},
            "Neurology": {"synonyms": ["neurology", "neuro", "neurologist"]},
            "General Surgery": {"synonyms": ["general surgery", "surgery", "surgeon"]},
            "Orthopedic Surgery": {"synonyms": ["orthopedic surgery", "orthopedics", "ortho"]},
            "Anesthesiology": {"synonyms": ["anesthesiology", "anesthesia"]},
            "Radiology": {"synonyms": ["radiology", "radiologist", "diagnostic radiology"]},
            "Pathology": {"synonyms": ["pathology", "pathologist"]},
            "Psychiatry": {"synonyms": ["psychiatry", "psychiatric", "mental health"]},
            "Obstetrics": {"synonyms": ["obstetrics", "ob", "obgyn", "ob/gyn"]},
            "Oncology": {"synonyms": ["oncology", "oncologist", "cancer"]},
            "Ophthalmology": {"synonyms": ["ophthalmology", "eye", "vision"]},
            "Urology": {"synonyms": ["urology", "urologist"]},
            "Gastroenterology": {"synonyms": ["gastroenterology", "gi", "gastro"]},
            "Endocrinology": {"synonyms": ["endocrinology", "endo", "diabetes"]},
            "Nephrology": {"synonyms": ["nephrology", "kidney", "renal"]},
            "Allergy": {"synonyms": ["allergy", "allergist"]},
            "Immunology": {"synonyms": ["immunology", "immunologist"]}
        }
    
    def _build_specialty_synonyms(self) -> Dict[str, str]:
        """Build reverse mapping from synonyms to canonical specialty names"""
        synonym_map = {}
        
        for canonical_name, config in self.specialties_config.items():
            synonym_map[canonical_name.lower()] = canonical_name
            
            synonyms = config.get('synonyms', [])
            for synonym in synonyms:
                synonym_map[synonym.lower()] = canonical_name
        
        return synonym_map
    
    def _build_taxonomy_mappings(self) -> Dict[str, str]:
        """Build mapping from taxonomy codes to specialty names"""
        taxonomy_map = {}
        
        for canonical_name, config in self.specialties_config.items():
            synonyms = config.get('synonyms', [])
            for synonym in synonyms:
                if re.match(r'^[0-9]{3,4}[a-z]*[0-9]{4,}x$', synonym.lower()):
                    taxonomy_map[synonym.lower()] = canonical_name
        
        return taxonomy_map
    
    def extract_dates(self, text: str) -> List[ExtractionCandidate]:
        """Extract dates using NER (handles word formats like '22 September 2025')"""
        candidates = []
        
        if not HAS_SPACY or not self.nlp:
            return self._extract_dates_fallback(text)
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    normalized_date = self._normalize_word_date(ent.text)
                    
                    if normalized_date:
                        context = self._get_surrounding_context(doc, ent)
                        context_lower = context.lower()
                        
                        confidence = 0.8
                        if any(keyword in context_lower for keyword in ['effective', 'start', 'begin']):
                            confidence = 0.9
                        elif any(keyword in context_lower for keyword in ['term', 'end', 'finish', 'expir']):
                            confidence = 0.9
                        
                        candidate = ExtractionCandidate(
                            value=normalized_date,
                            confidence=confidence,
                            extractor_id="spacy_date",
                            position=ent.start_char,
                            context=context,
                            validation_passed=True
                        )
                        candidates.append(candidate)
        
        except Exception as e:
            self.logger.error(f"Date NER extraction failed: {e}")
            return self._extract_dates_fallback(text)
        
        return candidates

    def extract_transaction_types(self, text: str) -> List[ExtractionCandidate]:
        """Extract transaction types using advanced contextual analysis"""
        candidates = []
        text_lower = text.lower()
        
        explicit_phrases = {
            'Term': [
                'provider termination', 'terminate provider', 'provider term',
                'discontinue provider', 'remove provider', 'end provider',
                'provider withdrawal', 'cancel provider', 'provider departure'
            ],
            'Update': [
                'address change', 'phone change', 'information change',
                'provider update', 'update provider', 'modify provider',
                'change provider', 'provider modification', 'address update',
                'phone update', 'demographic change', 'contact change',
                'location change', 'practice change', 'office change'
            ],
            'Add': [
                'new provider', 'add provider', 'provider enrollment', 
                'provider addition', 'include provider', 'onboard provider',
                'welcome provider', 'provider registration', 'provider credentialing'
            ]
        }
        
        for transaction_type, phrases in explicit_phrases.items():
            for phrase in phrases:
                if phrase in text_lower:
                    position = text_lower.find(phrase)
                    candidate = ExtractionCandidate(
                        value=transaction_type,
                        confidence=0.95,
                        extractor_id=f"transaction_explicit_{transaction_type.lower()}",
                        position=position,
                        context=text[max(0, position-30):position+len(phrase)+30],
                        validation_passed=True
                    )
                    candidates.append(candidate)
                    return candidates
        
        contextual_score = self._analyze_transaction_context(text_lower)
        if contextual_score['type'] and contextual_score['confidence'] > 0.6:
            candidate = ExtractionCandidate(
                value=contextual_score['type'],
                confidence=contextual_score['confidence'],
                extractor_id="transaction_contextual",
                position=contextual_score['position'],
                context=contextual_score['context'],
                validation_passed=True
            )
            candidates.append(candidate)
            return candidates
        
        terminate_patterns = [
            r'\bterminate\b', r'\bterminated\b', r'\btermination\b',
            r'\bremove\b', r'\bdiscontinue\b', r'\bwithdraw\b',
            r'\bcancel\b', r'\bexpire\b', r'\bcease\b', r'\bend\b',
            r'\bstop\b', r'no longer'
        ]
        
        for pattern in terminate_patterns:
            match = re.search(pattern, text_lower)
            if match:
                candidate = ExtractionCandidate(
                    value='Term',
                    confidence=0.9,
                    extractor_id="transaction_ner_term",
                    position=match.start(),
                    context=text[max(0, match.start()-30):match.end()+30],
                    validation_passed=True
                )
                candidates.append(candidate)
                return candidates
        
        # Update keywords
        update_patterns = [
            r'\bupdate\b', r'\bmodify\b', r'\bchange\b', r'\brevise\b',
            r'\bamend\b', r'\bcorrect\b', r'\bedit\b', r'\badjust\b',
            r'\balter\b', r'\brefresh\b', r'\bmove\b', r'\brelocate\b'
        ]
        
        update_found = False
        for pattern in update_patterns:
            match = re.search(pattern, text_lower)
            if match:
                context_window = text_lower[max(0, match.start()-50):match.end()+50]
                change_indicators = ['address', 'phone', 'contact', 'location', 'information', 'demographic', 'details']
                
                if any(indicator in context_window for indicator in change_indicators):
                    confidence = 0.85
                else:
                    confidence = 0.7
                
                candidate = ExtractionCandidate(
                    value='Update',
                    confidence=confidence,
                    extractor_id="transaction_ner_update",
                    position=match.start(),
                    context=text[max(0, match.start()-30):match.end()+30],
                    validation_passed=True
                )
                candidates.append(candidate)
                update_found = True
                break
        
        # Add keywords
        if not update_found:
            add_patterns = [
                r'\badd\b', r'\bnew\b', r'\binclude\b', r'\benroll\b',
                r'\bregister\b', r'\bjoin\b', r'\bwelcome\b', r'\bonboard\b',
                r'\brecruit\b', r'\bhire\b'
            ]
            
            for pattern in add_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    context_window = text_lower[max(0, match.start()-50):match.end()+50]
                    update_indicators = ['change', 'modify', 'update', 'alter', 'correct']
                    
                    if any(indicator in context_window for indicator in update_indicators):
                        continue
                    
                    candidate = ExtractionCandidate(
                        value='Add',
                        confidence=0.8,
                        extractor_id="transaction_ner_add",
                        position=match.start(),
                        context=text[max(0, match.start()-30):match.end()+30],
                        validation_passed=True
                    )
                    candidates.append(candidate)
                    break
        
        return candidates
    
    def _analyze_transaction_context(self, text_lower: str) -> dict:
        """Advanced contextual analysis to determine transaction type"""
        
        scores = {'Add': 0, 'Update': 0, 'Term': 0}
        best_match = {'type': None, 'confidence': 0, 'position': 0, 'context': ''}
        
        context_indicators = {
            'Update': {
                'strong': [
                    ('effective date', 2.0), ('address change', 2.0), ('phone change', 2.0),
                    ('contact change', 2.0), ('location change', 2.0), ('move', 1.8),
                    ('relocate', 1.8), ('transfer', 1.5), ('modify', 1.5)
                ],
                'medium': [
                    ('change', 1.2), ('update', 1.2), ('revise', 1.0), ('correct', 1.0),
                    ('edit', 1.0), ('adjust', 1.0), ('alter', 1.0)
                ],
                'weak': [
                    ('different', 0.5), ('new address', 0.8), ('new phone', 0.8),
                    ('updated', 0.7), ('current', 0.3)
                ]
            },
            'Add': {
                'strong': [
                    ('new provider', 2.5), ('welcome', 2.0), ('enrollment', 2.0),
                    ('credentialing', 2.0), ('onboard', 1.8), ('recruit', 1.8),
                    ('joined our network', 2.2), ('joining our network', 2.2),
                    ('has joined', 2.0), ('will be joining', 2.0)
                ],
                'medium': [
                    ('new', 1.0), ('add', 1.2), ('include', 1.0), ('join', 1.2),
                    ('register', 1.2), ('enroll', 1.5), ('joined', 1.4), 
                    ('joining', 1.4), ('please add', 1.8)
                ],
                'weak': [
                    ('first time', 0.8), ('initial', 0.5), ('begin', 0.5)
                ]
            },
            'Term': {
                'strong': [
                    ('termination', 2.5), ('terminated', 2.0), ('departure', 2.0),
                    ('discontinue', 2.0), ('withdraw', 1.8), ('cease', 1.8),
                    ('no longer be associated', 2.2), ('will no longer', 2.0),
                    ('process the termination', 2.3)
                ],
                'medium': [
                    ('remove', 1.2), ('end', 1.0), ('stop', 1.2), ('cancel', 1.5),
                    ('expire', 1.3), ('no longer associated', 1.6)
                ],
                'weak': [
                    ('no longer', 1.5), ('final', 0.5), ('last', 0.3)
                ]
            }
        }
        
        for trans_type, categories in context_indicators.items():
            for indicators in categories.values():
                for indicator, weight in indicators:
                    if indicator in text_lower:
                        scores[trans_type] += weight
                        
                        if scores[trans_type] > best_match['confidence']:
                            position = text_lower.find(indicator)
                            best_match.update({
                                'type': trans_type,
                                'confidence': min(scores[trans_type] / 3.0, 0.9),
                                'position': position,
                                'context': text_lower[max(0, position-30):position+len(indicator)+30]
                            })

        subject_patterns = {
            'Add': [r'new\s+provider', r'provider\s+enrollment', r'welcome', r'onboard'],
            'Update': [r'address\s+change', r'update', r'change', r'modify', r'move'],
            'Term': [r'termination', r'terminate', r'end', r'discontinue']
        }

        first_50_chars = text_lower[:50]
        for trans_type, patterns in subject_patterns.items():
            for pattern in patterns:
                if re.search(pattern, first_50_chars):
                    scores[trans_type] += 1.5
                    if scores[trans_type] > best_match['confidence'] * 3.0:
                        best_match.update({
                            'type': trans_type,
                            'confidence': min(scores[trans_type] / 3.0, 0.9),
                            'position': 0,
                            'context': first_50_chars
                        })
        
        if scores['Add'] > 0 and scores['Update'] > 0:
            if any(word in text_lower for word in ['existing provider', 'current provider', 'already enrolled']):
                scores['Update'] += 1.0
                scores['Add'] -= 0.5
            elif any(word in text_lower for word in ['first time', 'never been', 'not currently']):
                scores['Add'] += 1.0 
                scores['Update'] -= 0.5
        
        max_score = max(scores.values())
        if max_score > 0:
            best_type = max(scores, key=scores.get)
            if best_type != best_match['type'] or scores[best_type] > best_match['confidence'] * 3.0:
                key_indicators = [item[0] for sublist in context_indicators[best_type].values() for item in sublist]
                position = 0
                context = text_lower[:50]
                for indicator in key_indicators:
                    if indicator in text_lower:
                        position = text_lower.find(indicator)
                        context = text_lower[max(0, position-30):position+len(indicator)+30]
                        break
                
                best_match.update({
                    'type': best_type,
                    'confidence': min(scores[best_type] / 3.0, 0.9),
                    'position': position,
                    'context': context
                })
        
        return best_match
    
    def _extract_names_fallback(self, text: str) -> List[ExtractionCandidate]:
        """Fallback name extraction without spaCy"""
        candidates = []
        
        name_patterns = [
            r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?\s+)*[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+M\.?D\.?',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+D\.?O\.?',
        ]
        
        for i, pattern in enumerate(name_patterns):
            matches = re.finditer(pattern, text)
            
            for match in matches:
                name = match.group(1) if match.groups() else match.group(0)
                
                candidate = ExtractionCandidate(
                    value=self._normalize_name(name),
                    confidence=0.6,
                    extractor_id=f"name_pattern_{i}",
                    position=match.start(),
                    context=text[max(0, match.start()-20):match.end()+20],
                    validation_passed=True
                )
                candidates.append(candidate)
        
        return candidates
    
    
    def _is_likely_provider_name(self, name: str) -> bool:
        """Check if a detected name is likely a healthcare provider"""
        name_lower = name.lower()
        
        skip_patterns = ['best regards', 'thank you', 'sincerely', 'email', 'phone']
        if any(pattern in name_lower for pattern in skip_patterns):
            return False
        
        parts = name.split()
        return len(parts) >= 2 and all(len(part) > 1 for part in parts)
    
    def _is_healthcare_org(self, org: str) -> bool:
        """Check if organization is healthcare-related"""
        org_lower = org.lower()
        healthcare_indicators = ['medical', 'health', 'clinic', 'hospital', 'practice', 'physicians', 'doctors']
        return any(indicator in org_lower for indicator in healthcare_indicators)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize provider name"""
        name = ' '.join(name.split())
        
        name = re.sub(r',\s*(M\.?D\.?|D\.?O\.?)$', r', \1', name, flags=re.IGNORECASE)
        
        return name.strip()
    
    def _normalize_org_name(self, org: str) -> str:
        """Normalize organization name"""
        org = ' '.join(org.split())
        
        words = org.split()
        normalized_words = []
        
        for word in words:
            if word.lower() in ['and', 'the', 'of', 'for', 'in', 'on', 'at']:
                normalized_words.append(word.lower())
            else:
                normalized_words.append(word.title())
        
        return ' '.join(normalized_words)
    
    def _map_lob_to_canonical(self, lob: str) -> str:
        """Map LOB variant to canonical form"""
        lob_lower = lob.lower()
        
        if any(x in lob_lower for x in ['medicare', 'part a', 'part b', 'part c', 'part d']):
            return 'Medicare'
        elif any(x in lob_lower for x in ['medicaid', 'medi-cal']):
            return 'Medicaid'
        elif any(x in lob_lower for x in ['commercial', 'hmo', 'ppo', 'epo', 'pos', 'exchange']):
            return 'Commercial'
        else:
            return lob.title()
    
    def _get_surrounding_context(self, doc, span) -> str:
        """Get context around a spaCy span"""
        start = max(0, span.start - 10)
        end = min(len(doc), span.end + 10)
        return doc[start:end].text.replace('\n', ' ')
    
    def _find_names_near_title(self, doc, title_start, title_end):
        """Find names near provider titles"""
        candidates = []
        
        window_start = max(0, title_start - 5)
        window_end = min(len(doc), title_end + 5)
        
        for token in doc[window_start:window_end]:
            if token.ent_type_ == "PERSON":
                ent = token.ent
                if self._is_likely_provider_name(ent.text):
                    candidate = ExtractionCandidate(
                        value=self._normalize_name(ent.text),
                        confidence=0.9,
                        extractor_id="title_adjacent_name",
                        position=ent.start_char,
                        context=self._get_surrounding_context(doc, ent),
                        validation_passed=True
                    )
                    candidates.append(candidate)
                    break
        
        return candidates
    
    def _find_full_org_name(self, doc, match_start, match_end):
        """Find full organization name around a healthcare keyword"""
        candidates = []
        
        start_idx = match_start
        while start_idx > 0 and doc[start_idx - 1].is_title:
            start_idx -= 1
        
        org_span = doc[start_idx:match_end]
        org_name = org_span.text.strip()
        
        if len(org_name) > 5:
            candidate = ExtractionCandidate(
                value=self._normalize_org_name(org_name),
                confidence=0.8,
                extractor_id="pattern_org_extended",
                position=org_span.start_char,
                context=self._get_surrounding_context(doc, org_span),
                validation_passed=True
            )
            candidates.append(candidate)
        
        return candidates