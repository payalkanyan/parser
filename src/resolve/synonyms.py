import re
import yaml
from pathlib import Path
import logging

class SynonymMapper:
    """
    Maps synonyms to canonical forms for LOB, specialties, and organizations
    Supports fuzzy matching and configurable mappings
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.config_dir = Path(__file__).parent.parent.parent / "configs"
        
        self.lob_mappings = self._load_lob_mappings()
        self.specialty_mappings = self._load_specialty_mappings()
        self.org_type_mappings = self._load_organization_type_mappings()
    
    def _load_lob_mappings(self) -> dict[str, list[str]]:
        """Load Line of Business mappings from YAML config"""
        try:
            lob_file = self.config_dir / "lob_map.yml"
            with open(lob_file, 'r', encoding='utf-8') as f:
                lob_config = yaml.safe_load(f)
            
            mappings = {}
            for lob_type, config in lob_config.items():
                if isinstance(config, dict) and 'synonyms' in config:
                    mappings[lob_type] = config['synonyms']
                    
            self.logger.info(f"Loaded {len(mappings)} LOB mappings from {lob_file}")
            return mappings
            
        except Exception as e:
            self.logger.error(f"Failed to load LOB mappings: {e}")
            return {}
    
    def _load_specialty_mappings(self) -> dict[str, list[str]]:
        """Load medical specialty mappings from YAML config"""
        try:
            specialty_file = self.config_dir / "specialties.yml"
            with open(specialty_file, 'r', encoding='utf-8') as f:
                specialty_config = yaml.safe_load(f)
            
            mappings = {}
            if 'specialties' in specialty_config:
                for specialty, config in specialty_config['specialties'].items():
                    if isinstance(config, dict) and 'synonyms' in config:
                        mappings[specialty] = config['synonyms']
                        
            self.logger.info(f"Loaded {len(mappings)} specialty mappings from {specialty_file}")
            return mappings
            
        except Exception as e:
            self.logger.error(f"Failed to load specialty mappings: {e}")
            return {}
    
    def _load_organization_type_mappings(self) -> dict[str, list[str]]:
        """Load organization type mappings from YAML config"""
        try:
            org_file = self.config_dir / "organization_types.yml"
            with open(org_file, 'r', encoding='utf-8') as f:
                org_config = yaml.safe_load(f)
            
            mappings = {}
            if 'organization_types' in org_config:
                for org_type, config in org_config['organization_types'].items():
                    if isinstance(config, dict) and 'synonyms' in config:
                        mappings[org_type] = config['synonyms']
                        
            self.logger.info(f"Loaded {len(mappings)} organization type mappings from {org_file}")
            return mappings
            
        except Exception as e:
            self.logger.error(f"Failed to load organization type mappings: {e}")
            return {}

    def normalize_lob(self, lob_text: str) -> list[str]:
        """
        Normalize Line of Business text to canonical forms
        Returns list of canonical LOB values found
        """
        if not lob_text or lob_text == "Information not found":
            return []
        
        lob_lower = lob_text.lower().strip()
        canonical_lobs = []
        found_lobs = set()
        
        for canonical, synonyms in self.lob_mappings.items():
            for synonym in synonyms:
                if synonym in lob_lower and canonical not in found_lobs:
                    if canonical == 'medicare':
                        canonical_lobs.append('Medicare')
                    elif canonical == 'medicaid':
                        canonical_lobs.append('Medicaid')
                    elif canonical in ['commercial', 'hmo', 'ppo', 'epo', 'pos', 'exchange']:
                        canonical_lobs.append('Commercial')
                    
                    found_lobs.add(canonical)
                    break
        
        if not canonical_lobs:
            if re.search(r'\bhmo\b', lob_lower):
                canonical_lobs.append('Commercial')
            elif re.search(r'\bppo\b', lob_lower):
                canonical_lobs.append('Commercial')
            elif re.search(r'\bmedicare\b', lob_lower):
                canonical_lobs.append('Medicare')
            elif re.search(r'\bmedicaid\b', lob_lower):
                canonical_lobs.append('Medicaid')
        
        return canonical_lobs if canonical_lobs else ['Commercial']
    
    def normalize_specialty(self, specialty_text: str) -> str:
        """
        Normalize medical specialty to canonical form
        """
        if not specialty_text or specialty_text == "Information not found":
            return "Information not found"
        
        specialty_lower = specialty_text.lower().strip()
        
        for canonical, synonyms in self.specialty_mappings.items():
            for synonym in synonyms:
                if synonym == specialty_lower:
                    return canonical
        
        for canonical, synonyms in self.specialty_mappings.items():
            for synonym in synonyms:
                if synonym in specialty_lower or specialty_lower in synonym:
                    # Additional confidence check
                    if len(synonym) > 3:
                        return canonical
        
        return self._title_case_specialty(specialty_text)
    
    def normalize_organization_name(self, org_text: str) -> str:
        """
        Normalize organization name with proper casing and format
        """
        if not org_text or org_text == "Information not found":
            return "Information not found"
        
        org_clean = ' '.join(org_text.split())
        
        words = org_clean.split()
        normalized_words = []
        
        for word in words:
            word_lower = word.lower()
            
            if word_lower in ['and', 'the', 'of', 'for', 'in', 'on', 'at', 'by', 'with']:
                normalized_words.append(word_lower)
            elif word_lower in ['md', 'do', 'llc', 'inc', 'corp', 'pa', 'pc']:
                normalized_words.append(word.upper())
            elif re.match(r'^[a-z]+$', word_lower) and len(word) <= 4:
                normalized_words.append(word.upper())
            else:
                normalized_words.append(word.title())
        
        if normalized_words:
            normalized_words[0] = normalized_words[0].title()
        
        return ' '.join(normalized_words)
    
    def normalize_provider_name(self, name_text: str) -> str:
        """
        Normalize provider name with proper formatting
        """
        if not name_text or name_text == "Information not found":
            return "Information not found"
        
        name_clean = ' '.join(name_text.split())
        
        if ',' in name_clean:
            parts = name_clean.split(',', 1)
            if len(parts) == 2:
                last_name = parts[0].strip().title()
                first_part = parts[1].strip()
                
                first_part = self._normalize_name_titles(first_part)
                
                return f"{last_name}, {first_part}"
        
        else:
            words = name_clean.split()
            normalized_words = []
            
            for word in words:
                word_normalized = self._normalize_name_titles(word)
                normalized_words.append(word_normalized)
            
            return ' '.join(normalized_words)
    
    def _title_case_specialty(self, specialty: str) -> str:
        """Apply proper title case to specialty"""
        words = specialty.split()
        titled_words = []
        
        for word in words:
            if word.lower() in ['and', 'of', 'the', 'in', 'on', 'for']:
                titled_words.append(word.lower())
            else:
                titled_words.append(word.title())
        
        if titled_words:
            titled_words[0] = titled_words[0].title()
        
        return ' '.join(titled_words)
    
    def _normalize_name_titles(self, name_part: str) -> str:
        """Normalize name titles and suffixes"""
        title_mappings = {
            'dr': 'Dr.',
            'dr.': 'Dr.',
            'doctor': 'Dr.',
            'md': 'M.D.',
            'm.d.': 'M.D.',
            'm.d': 'M.D.',
            'do': 'D.O.',
            'd.o.': 'D.O.',
            'd.o': 'D.O.',
            'jr': 'Jr.',
            'jr.': 'Jr.',
            'sr': 'Sr.',
            'sr.': 'Sr.',
            'ii': 'II',
            'iii': 'III',
            'iv': 'IV'
        }
        
        name_lower = name_part.lower().strip()
        
        if name_lower in title_mappings:
            return title_mappings[name_lower]
        
        return name_part.title()
    
    def apply_all_normalizations(self, data: dict[str, str]) -> dict[str, str]:
        """
        Apply all normalizations to a data dictionary
        Modifies the dictionary in place and returns it
        """
        if 'Line Of Business (Medicare/Commercial/Medical)' in data:
            lob_text = data['Line Of Business (Medicare/Commercial/Medical)']
            normalized_lobs = self.normalize_lob(lob_text)
            if normalized_lobs:
                data['Line Of Business (Medicare/Commercial/Medical)'] = ', '.join(normalized_lobs)
        
        if 'Provider Specialty' in data:
            data['Provider Specialty'] = self.normalize_specialty(data['Provider Specialty'])
        
        if 'Organization Name' in data:
            data['Organization Name'] = self.normalize_organization_name(data['Organization Name'])
        
        if 'Provider Name' in data:
            data['Provider Name'] = self.normalize_provider_name(data['Provider Name'])
        
        return data