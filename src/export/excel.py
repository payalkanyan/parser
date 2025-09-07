"""
Excel writer with template matching
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from ..resolve.validators import FieldValidator
from ..resolve.synonyms import SynonymMapper
from ..resolve.column_validator import ColumnValidator


class ExcelExporter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = FieldValidator()
        self.synonym_mapper = SynonymMapper()
        self.column_validator = ColumnValidator()
        
        self.expected_columns = self.column_validator.get_column_names()
    
    def export_to_excel(
        self, 
        extracted_data: List[Dict[str, str]], 
        template_path: Path, 
        output_path: Path
    ) -> bool:

        try:
            template_columns = self._read_template_columns(template_path)
            
            if template_columns:
                self.expected_columns = template_columns
                self.logger.info(f"Using template column order: {len(template_columns)} columns")
            else:
                self.logger.warning("Could not read template, using default column order")
            
            processed_records = []
            for i, record in enumerate(extracted_data):
                processed_record = self._process_record(record, i)
                
                validation_errors = self.column_validator.validate_record(processed_record)
                if validation_errors:
                    self.logger.warning(f"Record {i} validation errors:")
                    for field, errors in validation_errors.items():
                        for error in errors:
                            self.logger.warning(f"  {field}: {error}")
                
                processed_records.append(processed_record)
            
            df = pd.DataFrame(processed_records, columns=self.expected_columns)
            
            for col in self.expected_columns:
                if col not in df.columns:
                    df[col] = "Information not found"
            
            df = df[self.expected_columns]
            
            self._write_excel_file(df, output_path)
            
            self.logger.info(f"Successfully exported {len(processed_records)} records to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Excel export failed: {str(e)}")
            return False
    
    def _read_template_columns(self, template_path: Path) -> Optional[List[str]]:
        """Read column headers from template file"""
        try:
            if not template_path.exists():
                self.logger.warning(f"Template file not found: {template_path}")
                return None
            
            df = pd.read_excel(template_path, sheet_name='Output')
            return list(df.columns)
            
        except Exception as e:
            self.logger.error(f"Could not read template columns: {str(e)}")
            try:
                df = pd.read_excel(template_path)
                return list(df.columns)
            except Exception as e2:
                self.logger.error(f"Fallback template read also failed: {str(e2)}")
                return None
    
    def _process_record(self, record: Dict[str, str], record_index: int) -> Dict[str, str]:
        """
        Process a single record: validate, normalize, and ensure completeness
        """
        processed = record.copy()
        
        processed = self.synonym_mapper.apply_all_normalizations(processed)
        
        validation_results = self.validator.validate_and_normalize_all(processed)
        
        for field, result in validation_results.items():
            if not result.is_valid:
                self.logger.warning(f"Record {record_index} - {field}: {result.message}")
        
        for col in self.expected_columns:
            if col not in processed:
                processed[col] = "Information not found"
        
        processed = self._apply_business_rules(processed)
        
        return processed
    
    def _apply_business_rules(self, record: Dict[str, str]) -> Dict[str, str]:
        """
        Apply business rules and cross-field logic
        """
        transaction_type = record.get('Transaction Type (Add/Update/Term)', '').lower()
        
        if transaction_type == 'term':
            if record.get('Term Date') == "Information not found" and record.get('Effective Date') != "Information not found":
                
                record['Term Date'] = record['Effective Date']
                record['Effective Date'] = "Information not found"
        else:
            record['Term Date'] = "Information not found"
        
        if transaction_type == 'term':
            record['Transaction Attribute'] = 'Provider'
        elif transaction_type == 'add':
            record['Transaction Attribute'] = "Information not found"
        
        if transaction_type != 'term':
            record['Term Reason'] = "Information not found"
        
        for field in ['Phone Number', 'Fax Number']:
            if field in record and record[field] != "Information not found":
                digits = ''.join(filter(str.isdigit, record[field]))
                if len(digits) == 10:
                    record[field] = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        
        for field in ['Provider NPI', 'Group NPI']:
            if field in record and record[field] != "Information not found":
                record[field] = ''.join(filter(str.isdigit, record[field]))
        
        if 'TIN' in record and record['TIN'] != "Information not found":
            tin_digits = ''.join(filter(str.isdigit, record['TIN']))
            if len(tin_digits) == 9:
                record['TIN'] = f"{tin_digits[:2]}-{tin_digits[2:]}"
        
        if 'PPG ID' in record and record['PPG ID'] != "Information not found":
            record['PPG ID'] = ''.join(c for c in record['PPG ID'] if c.isalnum() or c in ', ')
        
        return record
    
    def _write_excel_file(self, df: pd.DataFrame, output_path: Path):
        """
        Write DataFrame to Excel with proper formatting
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(
            output_path,
            engine='openpyxl'
        ) as writer:
            
            df.to_excel(
                writer,
                sheet_name='Output',
                index=False,
                header=True
            )
            
            workbook = writer.book
            worksheet = writer.sheets['Output']
            
            self._format_worksheet(worksheet, df)
    
    def _format_worksheet(self, worksheet, df: pd.DataFrame):
        """Apply basic formatting to the worksheet"""
        try:
            from openpyxl.styles import Font, Alignment
            
            header_font = Font(bold=True)
            for cell in worksheet[1]: 
                cell.font = header_font
                cell.alignment = Alignment(horizontal='left', vertical='center')
            
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            worksheet.freeze_panes = 'A2'
            
        except ImportError:
            self.logger.warning("openpyxl not available for Excel formatting")
        except Exception as e:
            self.logger.warning(f"Excel formatting failed: {str(e)}")
    
    def validate_output(self, output_path: Path) -> bool:
        """
        Validate the created Excel file
        """
        try:
            df = pd.read_excel(output_path, sheet_name='Output')
            
            missing_columns = set(self.expected_columns) - set(df.columns)
            if missing_columns:
                self.logger.error(f"Output file missing columns: {missing_columns}")
                return False
            
            if len(df) == 0:
                self.logger.warning("Output file contains no data rows")
                return False
            
            self.logger.info(f"Output validation passed: {len(df)} rows, {len(df.columns)} columns")
            return True
            
        except Exception as e:
            self.logger.error(f"Output validation failed: {str(e)}")
            return False