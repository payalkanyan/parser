from typing import Optional
import logging
from dataclasses import dataclass

@dataclass
class ColumnSpec:
    """Data Structure for a single column"""
    name: str
    description: str
    example: str
    allowed_values: Optional[list[str]] = None
    required_for_transaction_types: Optional[list[str]] = None

class ColumnValidator:
    """
    Class for validating column values against Output Format.xlsx specifications
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.column_specs = {
            'Transaction Type (Add/Update/Term)': ColumnSpec(
                name='Transaction Type (Add/Update/Term)',
                description='Identifies the type of change in the provider record. For example, Add for a new provider, Update for changes, and Term for termination.',
                example='Add, Update, Term',
                allowed_values=['Add', 'Update', 'Term']
            ),
            
            'Transaction Attribute': ColumnSpec(
                name='Transaction Attribute',
                description='Specifies which attribute changed in case of an Update (e.g., Address, Specialty). For Add/Term, may be marked as Not Applicable.',
                example='Specialty, Provider, Address, PPG, Phone Number, LOB',
                allowed_values=['Specialty', 'Provider', 'Address', 'PPG', 'Phone Number', 'LOB', 'Not Applicable']
            ),
            
            'Effective Date': ColumnSpec(
                name='Effective Date',
                description='The date on which the transaction (Add or Update) becomes active.',
                example='6/25/2025'
            ),
            
            'Term Date': ColumnSpec(
                name='Term Date',
                description='The date on which a termination is effective. Required if Transaction Type is Term.',
                example='6/25/2025',
                required_for_transaction_types=['Term']
            ),
            
            'Term Reason': ColumnSpec(
                name='Term Reason',
                description='Reason why a provider is being terminated (e.g., Provider Left Group, Contract Ended, License Expired).',
                example='Provider is retired',
                required_for_transaction_types=['Term']
            ),
            
            'Provider Name': ColumnSpec(
                name='Provider Name',
                description='Full legal name of the provider (First, Middle, Last).',
                example='John Doe'
            ),
            
            'Provider NPI': ColumnSpec(
                name='Provider NPI',
                description='National Provider Identifier (10-digit unique ID assigned to healthcare providers in the U.S.).',
                example='1638549275'
            ),
            
            'Provider Specialty': ColumnSpec(
                name='Provider Specialty',
                description='The provider\'s primary specialty (e.g., Pediatrics, Cardiology, Family Medicine).',
                example='Internal Medicine'
            ),
            
            'State License': ColumnSpec(
                name='State License',
                description='The state medical license number associated with the provider.',
                example='G68269'
            ),
            
            'Organization Name': ColumnSpec(
                name='Organization Name',
                description='The name of the organization, practice group, or facility where the provider is affiliated.',
                example='UCSD Health'
            ),
            
            'TIN': ColumnSpec(
                name='TIN',
                description='Tax Identification Number of the organization or group billing for the provider.',
                example='649264027'
            ),
            
            'Group NPI': ColumnSpec(
                name='Group NPI',
                description='National Provider Identifier assigned to the group or organization (distinct from the individual provider NPI).',
                example='1794739829'
            ),
            
            'Complete Address': ColumnSpec(
                name='Complete Address',
                description='Full practice location address, including street, city, state, and ZIP code.',
                example='123 Main Street, Atlanta, Georgia, 63929'
            ),
            
            'Phone Number': ColumnSpec(
                name='Phone Number',
                description='Main contact number for the provider or practice location.',
                example='2685562965'
            ),
            
            'Fax Number': ColumnSpec(
                name='Fax Number',
                description='Fax contact for the provider or practice (if available).',
                example='2845582945'
            ),
            
            'PPG ID': ColumnSpec(
                name='PPG ID',
                description='Provider Practice Group Identifier, a unique code used internally by health plans or provider groups.',
                example='P04, 1104, 569'
            ),
            
            'Line Of Business (Medicare/Commercial/Medical)': ColumnSpec(
                name='Line Of Business (Medicare/Commercial/Medical)',
                description='Indicates which line(s) of business the provider participates in (e.g., Medicare Advantage, Medicaid, Commercial Insurance).',
                example='Medicare, Medicaid, Commercial',
                allowed_values=['Medicare', 'Medicaid', 'Commercial']
            )
        }
    
    def validate_record(self, record: dict[str, str]) -> dict[str, list[str]]:
        """
        Validate a complete record against column specifications
        Returns dictionary of field_name -> list of validation errors
        """
        errors = {}
        
        transaction_type = record.get('Transaction Type (Add/Update/Term)', '').strip()
        
        for field_name, spec in self.column_specs.items():
            field_errors = []
            value = record.get(field_name, '').strip()
            
            if value == "Information not found":
                continue
            
            if spec.allowed_values and value:
                if field_name == 'Line Of Business (Medicare/Commercial/Medical)':
                    lob_values = [v.strip() for v in value.split(',')]
                    for lob in lob_values:
                        if lob and lob not in spec.allowed_values:
                            field_errors.append(f"'{lob}' is not an allowed value. Must be one of: {', '.join(spec.allowed_values)}")
                else:
                    if value not in spec.allowed_values:
                        field_errors.append(f"'{value}' is not an allowed value. Must be one of: {', '.join(spec.allowed_values)}")
            
            if spec.required_for_transaction_types:
                if transaction_type in spec.required_for_transaction_types:
                    if not value or value == "Information not found":
                        field_errors.append(f"Field is required when Transaction Type is '{transaction_type}'")
            
            if field_name == 'Provider NPI' and value and value != "Information not found":
                if not self._validate_npi_format(value):
                    field_errors.append("NPI must be 10 digits")
            
            elif field_name == 'Group NPI' and value and value != "Information not found":
                if not self._validate_npi_format(value):
                    field_errors.append("Group NPI must be 10 digits")
            
            elif field_name == 'TIN' and value and value != "Information not found":
                if not self._validate_tin_format(value):
                    field_errors.append("TIN must be 9 digits, formatted as XX-XXXXXXX")
            
            if field_errors:
                errors[field_name] = field_errors
        
        return errors
    
    def _validate_npi_format(self, npi: str) -> bool:
        """Validate NPI format (10 digits)"""
        digits_only = ''.join(filter(str.isdigit, npi))
        return len(digits_only) == 10
    
    def _validate_tin_format(self, tin: str) -> bool:
        """Validate TIN format (9 digits)"""
        digits_only = ''.join(filter(str.isdigit, tin))
        return len(digits_only) == 9
    
    def get_column_names(self) -> list[str]:
        """Get ordered list of column names"""
        return list(self.column_specs.keys())
    
    def get_allowed_values(self, column_name: str) -> Optional[list[str]]:
        """Get allowed values for a specific column"""
        spec = self.column_specs.get(column_name)
        return spec.allowed_values if spec else None