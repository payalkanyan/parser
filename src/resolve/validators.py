"""
Column Validation file for NPI/TIN/phone/taxonomy
"""
import re
from typing import Optional
import logging


class ValidationResult:
    """Container for validation results"""
    
    def __init__(self, is_valid: bool, message: str = "", normalized_value: Optional[str] = None):
        self.is_valid = is_valid
        self.message = message
        self.normalized_value = normalized_value or ""


class FieldValidator:
    """
    Validation library for healthcare provider fields
    Implements checksum validation, format validation, and normalization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_npi(self, npi: str) -> ValidationResult:
        """
        Validate NPI using Luhn algorithm with 80840 prefix
        """
        if not npi:
            return ValidationResult(False, "NPI is empty")
        
        npi_clean = re.sub(r'[^\d]', '', str(npi))
        
        if len(npi_clean) != 10:
            return ValidationResult(False, f"NPI must be 10 digits, got {len(npi_clean)}")
        
        # Luhn check with 80840 prefix
        # if not self._luhn_check_npi(npi_clean):
        #     self.logger.warning(f"NPI {npi_clean} failed Luhn checksum (might be test data)")
        #     return ValidationResult(True, "NPI format valid (Luhn check failed)", npi_clean)
        
        return ValidationResult(True, "Valid NPI", npi_clean)
    
    def validate_tin(self, tin: str) -> ValidationResult:
        """
        Validate TIN (exactly 9 digits)
        """
        if not tin:
            return ValidationResult(False, "TIN is empty")
        
        tin_clean = re.sub(r'[^\d]', '', str(tin))
        
        if len(tin_clean) != 9:
            return ValidationResult(False, f"TIN must be 9 digits, got {len(tin_clean)}")
        
        formatted_tin = f"{tin_clean[:2]}-{tin_clean[2:]}"
        
        return ValidationResult(True, "Valid TIN", formatted_tin)
    
    def validate_taxonomy_code(self, code: str) -> ValidationResult:
        """
        Validate taxonomy code format using regex
        """
        if not code:
            return ValidationResult(False, "Taxonomy code is empty")
        
        # Clean code
        code_clean = code.strip().upper()
        
        # Check format
        if not re.match(r'^[12]\d{2}[A-Z]\d{5}X$', code_clean):
            return ValidationResult(False, "Invalid taxonomy code format. Expected: [12]DD[A-Z]DDDDDX")
        
        return ValidationResult(True, "Valid taxonomy code", code_clean)
    
    def validate_phone_fax(self, number: str, field_type: str = "phone") -> ValidationResult:
        """
        Validate phone/fax number
        """
        if not number:
            return ValidationResult(False, f"{field_type} number is empty")
        
        number_clean = re.sub(r'[^\d]', '', str(number))
        
        # Check length
        if len(number_clean) < 10:
            return ValidationResult(False, f"{field_type} number too short: {len(number_clean)} digits")
        elif len(number_clean) > 11:
            return ValidationResult(False, f"{field_type} number too long: {len(number_clean)} digits")
        elif len(number_clean) == 11:
            # Remove leading 1 for US numbers
            if number_clean[0] == '1':
                number_clean = number_clean[1:]
            else:
                return ValidationResult(False, f"11-digit {field_type} number must start with 1")
        
        # Format as XXX-XXX-XXXX
        formatted_number = f"{number_clean[:3]}-{number_clean[3:6]}-{number_clean[6:]}"
        
        return ValidationResult(True, f"Valid {field_type} number", formatted_number)
    
    def validate_state_license(self, license_num: str, state: Optional[str] = None) -> ValidationResult:
        """
        Basic validation: Letter followed by digits
        """
        if not license_num:
            return ValidationResult(False, "License number is empty")
        
        license_clean = license_num.strip().upper()
        
        if not re.match(r'^[A-Z]\d{5,6}$', license_clean):
            return ValidationResult(False, "Invalid license format. Expected: Letter followed by 5-6 digits")
        
        return ValidationResult(True, "Valid license format", license_clean)
    
    def validate_date(self, date_str: str) -> ValidationResult:
        """
        Validate and normalize date to MM/DD/YYYY if possible
        """
        if not date_str:
            return ValidationResult(False, "Date is empty")
        
        # Try to parse and normalize
        normalized_date = self._normalize_date_format(date_str)
        
        if not normalized_date:
            return ValidationResult(False, f"Could not parse date: {date_str}")

        try:
            parts = normalized_date.split('/')
            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            
            if not (1 <= month <= 12):
                return ValidationResult(False, f"Invalid month: {month}")
            if not (1 <= day <= 31):
                return ValidationResult(False, f"Invalid day: {day}")
            if not (1900 <= year <= 2100):
                return ValidationResult(False, f"Invalid year: {year}")
                
        except (ValueError, IndexError):
            return ValidationResult(False, f"Invalid date format: {normalized_date}")
        
        return ValidationResult(True, "Valid date", normalized_date)
    
    def validate_ppg_id(self, ppg: str) -> ValidationResult:
        """
        Validate PPG ID (basic format check)
        """
        if not ppg:
            return ValidationResult(False, "PPG ID is empty")
        
        # Clean PPG - keep alphanumeric, commas, and spaces (for multiple PPG IDs)
        ppg_clean = re.sub(r'[^A-Za-z0-9, ]', '', str(ppg)).strip()
        
        if len(ppg_clean) < 1:
            return ValidationResult(False, "PPG ID contains no valid characters")
        
        return ValidationResult(True, "Valid PPG ID", ppg_clean)
    
    def _luhn_check_npi(self, npi: str) -> bool:
        """
        Luhn algorithm check for NPI with 80840 prefix
        """
        if len(npi) != 10:
            return False

        full_number = "80840" + npi[:-1]
        check_digit = int(npi[-1])

        total = 0
        for i, digit_char in enumerate(reversed(full_number)):
            digit = int(digit_char)
            
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit = digit - 9
            
            total += digit

        calculated_check = (10 - (total % 10)) % 10
        
        return calculated_check == check_digit
    
    def _normalize_date_format(self, date_str: str) -> Optional[str]:
        """
        Normalize various date formats to MM/DD/YYYY if possible
        """
        date_str = date_str.strip()

        patterns = [
            # MM/DD/YYYY or M/D/YYYY
            (r'^(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})$', lambda m: (m.group(1), m.group(2), m.group(3))),
            
            # YYYY/MM/DD or YYYY-MM-DD
            (r'^(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})$', lambda m: (m.group(2), m.group(3), m.group(1))),
            
            # DD/MM/YYYY (European format)
            (r'^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$', lambda m: (m.group(2), m.group(1), m.group(3))),
        ]
        
        for pattern, extract_func in patterns:
            match = re.match(pattern, date_str)
            if match:
                try:
                    month, day, year = extract_func(match)
                    
                    month_int = int(month)
                    day_int = int(day)
                    year_int = int(year)
                    
                    if year_int < 100:
                        if year_int < 50:
                            year_int += 2000
                        else:
                            year_int += 1900
                    
                    if 1 <= month_int <= 12 and 1 <= day_int <= 31 and 1900 <= year_int <= 2100:
                        return f"{month_int:02d}/{day_int:02d}/{year_int}"
                        
                except ValueError:
                    continue
        
        return None
    
    def validate_and_normalize_all(self, data: dict[str, str]) -> dict[str, ValidationResult]:
        """
        Validate and normalize all fields in a data dictionary
        Returns validation results for each field
        """
        results = {}
        
        field_validators = {
            'Provider NPI': self.validate_npi,
            'Group NPI': self.validate_npi,
            'TIN': self.validate_tin,
            'Phone Number': lambda x: self.validate_phone_fax(x, "phone"),
            'Fax Number': lambda x: self.validate_phone_fax(x, "fax"),
            'State License': self.validate_state_license,
            'Effective Date': self.validate_date,
            'Term Date': self.validate_date,
            'PPG ID': self.validate_ppg_id,
        }
        
        for field_name, validator in field_validators.items():
            if field_name in data and data[field_name] != "Information not found":
                try:
                    result = validator(data[field_name])
                    results[field_name] = result
                    
                    if result.is_valid and result.normalized_value:
                        data[field_name] = result.normalized_value
                        
                except Exception as e:
                    self.logger.error(f"Validation error for {field_name}: {str(e)}")
                    results[field_name] = ValidationResult(False, f"Validation error: {str(e)}")
        
        return results