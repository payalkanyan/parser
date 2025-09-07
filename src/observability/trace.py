"""
Trace Logger for Provenance and Debugging
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging


@dataclass
class ExtractionTrace:
    """Trace information for a field extraction"""
    field_name: str
    extracted_value: str
    extractor_id: str
    confidence: float
    source_text: str = ""
    position: int = -1
    timestamp: float = field(default_factory=time.time)
    validation_passed: bool = False
    validation_message: str = ""


@dataclass
class FileProcessingTrace:
    """Complete trace for a file processing session"""
    file_path: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    
    # Stage timing
    stage_traces: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Field extractions
    field_traces: List[ExtractionTrace] = field(default_factory=list)
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    blocks_detected: int = 0
    tables_found: int = 0
    attachments_processed: int = 0


class TraceLogger:
    """
    Provides detailed tracing and provenance tracking
    Enables audit trails and debugging support
    """
    
    def __init__(self, save_traces: bool = False, trace_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.save_traces = save_traces
        self.trace_dir = trace_dir or Path("traces")
        
        # Current active trace
        self.current_trace: Optional[FileProcessingTrace] = None
        
        # All traces for batch processing
        self.all_traces: List[FileProcessingTrace] = []
        
        if self.save_traces:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
    
    def start_trace(self, file_path: str) -> FileProcessingTrace:
        """Start tracing for a file"""
        self.current_trace = FileProcessingTrace(
            file_path=file_path,
            start_time=time.time()
        )
        
        self.logger.debug(f"Started trace for {file_path}")
        return self.current_trace
    
    def log_stage(self, stage_name: str, **kwargs):
        """Log the start of a processing stage"""
        if not self.current_trace:
            return
        
        stage_info = {
            "start_time": time.time(),
            "stage_data": kwargs
        }
        
        self.current_trace.stage_traces[stage_name] = stage_info
        self.logger.debug(f"Stage started: {stage_name}")
    
    def complete_stage(self, stage_name: str, **kwargs):
        """Complete a processing stage"""
        if not self.current_trace or stage_name not in self.current_trace.stage_traces:
            return
        
        stage_info = self.current_trace.stage_traces[stage_name]
        stage_info["end_time"] = time.time()
        stage_info["duration"] = stage_info["end_time"] - stage_info["start_time"]
        stage_info.update(kwargs)
        
        self.logger.debug(f"Stage completed: {stage_name} ({stage_info['duration']:.3f}s)")
    
    def log_extraction(
        self,
        field_name: str,
        extracted_value: str,
        extractor_id: str,
        confidence: float,
        source_text: str = "",
        position: int = -1,
        validation_passed: bool = False,
        validation_message: str = ""
    ):
        """Log a field extraction event"""
        if not self.current_trace:
            return
        
        trace = ExtractionTrace(
            field_name=field_name,
            extracted_value=extracted_value,
            extractor_id=extractor_id,
            confidence=confidence,
            source_text=source_text[:200],  # Limit source text length
            position=position,
            validation_passed=validation_passed,
            validation_message=validation_message
        )
        
        self.current_trace.field_traces.append(trace)
        
        self.logger.debug(
            f"Extraction: {field_name}='{extracted_value}' "
            f"via {extractor_id} (conf: {confidence:.2f})"
        )
    
    def log_block_detection(self, blocks_count: int):
        """Log block detection results"""
        if self.current_trace:
            self.current_trace.blocks_detected = blocks_count
            self.logger.debug(f"Detected {blocks_count} provider blocks")
    
    def log_table_detection(self, tables_count: int):
        """Log table detection results"""
        if self.current_trace:
            self.current_trace.tables_found = tables_count
            self.logger.debug(f"Found {tables_count} tables")
    
    def log_attachment_processing(self, attachments_count: int):
        """Log attachment processing"""
        if self.current_trace:
            self.current_trace.attachments_processed = attachments_count
            self.logger.debug(f"Processed {attachments_count} attachments")
    
    def log_error(self, error_message: str):
        """Log an error during processing"""
        if self.current_trace:
            self.current_trace.errors.append(f"{time.time()}: {error_message}")
        
        self.logger.error(error_message)
    
    def log_warning(self, warning_message: str):
        """Log a warning during processing"""
        if self.current_trace:
            self.current_trace.warnings.append(f"{time.time()}: {warning_message}")
        
        self.logger.warning(warning_message)
    
    def end_trace(self, success: bool = True):
        """End the current trace"""
        if not self.current_trace:
            return
        
        self.current_trace.end_time = time.time()
        self.current_trace.success = success
        
        duration = self.current_trace.end_time - self.current_trace.start_time
        self.logger.debug(f"Trace completed: {self.current_trace.file_path} ({duration:.3f}s)")
        
        # Save trace if enabled
        if self.save_traces:
            self._save_trace(self.current_trace)
        
        # Add to all traces
        self.all_traces.append(self.current_trace)
        
        # Clear current trace
        self.current_trace = None
    
    def get_field_provenance(self, field_name: str) -> List[Dict[str, Any]]:
        """
        Get provenance information for a specific field
        Returns all extraction attempts for the field
        """
        if not self.current_trace:
            return []
        
        field_extractions = []
        for trace in self.current_trace.field_traces:
            if trace.field_name == field_name:
                field_extractions.append({
                    "value": trace.extracted_value,
                    "extractor": trace.extractor_id,
                    "confidence": trace.confidence,
                    "source": trace.source_text,
                    "position": trace.position,
                    "timestamp": trace.timestamp,
                    "validation_passed": trace.validation_passed,
                    "validation_message": trace.validation_message
                })
        
        # Sort by confidence descending
        field_extractions.sort(key=lambda x: x["confidence"], reverse=True)
        return field_extractions
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of current processing trace"""
        if not self.current_trace:
            return {}
        
        total_extractions = len(self.current_trace.field_traces)
        successful_extractions = sum(
            1 for trace in self.current_trace.field_traces
            if trace.validation_passed and trace.confidence > 0.5
        )
        
        # Stage timing summary
        stage_timings = {}
        for stage_name, stage_info in self.current_trace.stage_traces.items():
            if "duration" in stage_info:
                stage_timings[stage_name] = round(stage_info["duration"], 3)
        
        return {
            "file_path": self.current_trace.file_path,
            "total_extractions": total_extractions,
            "successful_extractions": successful_extractions,
            "success_rate": successful_extractions / max(total_extractions, 1),
            "blocks_detected": self.current_trace.blocks_detected,
            "tables_found": self.current_trace.tables_found,
            "attachments_processed": self.current_trace.attachments_processed,
            "stage_timings": stage_timings,
            "errors": len(self.current_trace.errors),
            "warnings": len(self.current_trace.warnings)
        }
    
    def get_extraction_candidates_report(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get a report of all extraction candidates by field
        Useful for understanding which extractors found what
        """
        if not self.current_trace:
            return {}
        
        candidates_by_field = {}
        
        for trace in self.current_trace.field_traces:
            field = trace.field_name
            if field not in candidates_by_field:
                candidates_by_field[field] = []
            
            candidates_by_field[field].append({
                "value": trace.extracted_value,
                "extractor": trace.extractor_id,
                "confidence": trace.confidence,
                "validation_passed": trace.validation_passed
            })
        
        # Sort candidates by confidence for each field
        for field in candidates_by_field:
            candidates_by_field[field].sort(key=lambda x: x["confidence"], reverse=True)
        
        return candidates_by_field
    
    def _save_trace(self, trace: FileProcessingTrace):
        """Save trace to file"""
        try:
            # Create filename from file path and timestamp
            file_stem = Path(trace.file_path).stem
            timestamp = int(trace.start_time)
            trace_filename = f"trace_{file_stem}_{timestamp}.json"
            trace_path = self.trace_dir / trace_filename
            
            # Convert to dictionary for JSON serialization
            trace_dict = asdict(trace)
            
            with open(trace_path, 'w') as f:
                json.dump(trace_dict, f, indent=2, default=str)
            
            self.logger.debug(f"Trace saved to {trace_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save trace: {str(e)}")
    
    def export_all_traces(self, output_path: Path):
        """Export all collected traces to a single file"""
        try:
            traces_data = {
                "export_timestamp": time.time(),
                "total_traces": len(self.all_traces),
                "traces": [asdict(trace) for trace in self.all_traces]
            }
            
            with open(output_path, 'w') as f:
                json.dump(traces_data, f, indent=2, default=str)
            
            self.logger.info(f"All traces exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export traces: {str(e)}")
    
    def clear_traces(self):
        """Clear all stored traces"""
        self.all_traces.clear()
        self.current_trace = None
        self.logger.debug("All traces cleared")