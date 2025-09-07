"""
Metrics Collection
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import logging


@dataclass
class ProcessingMetrics:
    """Container for processing metrics"""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    
    # Stage-wise timing
    stage_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Field extraction success rates
    field_success_rates: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"success": 0, "total": 0}))
    
    # Extractor performance
    extractor_performance: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"success": 0, "total": 0}))


class MetricsCollector:
    """
    Collect and analyze processing metrics
    Provides TAT analysis and field-level success tracking
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = ProcessingMetrics()
        self.session_start_time = time.time()
        
        # Expected fields for success rate calculation
        self.tracked_fields = [
            'Transaction Type (Add/Update/Term)',
            'Provider Name',
            'Provider NPI',
            'TIN',
            'Organization Name',
            'Provider Specialty',
            'State License',
            'Phone Number',
            'Fax Number',
            'PPG ID',
            'Line Of Business (Medicare/Commercial/Medical)',
            'Effective Date',
            'Term Date'
        ]
    
    def record_processing_time(self, processing_time: float):
        """Record total processing time for a file"""
        self.metrics.total_processing_time += processing_time
        self.metrics.total_files += 1
        
        if self.metrics.total_files > 0:
            self.metrics.average_processing_time = self.metrics.total_processing_time / self.metrics.total_files
    
    def record_stage_time(self, stage_name: str, stage_time: float):
        """Record time for a specific processing stage"""
        self.metrics.stage_times[stage_name].append(stage_time)
    
    def record_file_success(self, success: bool):
        """Record file processing success/failure"""
        if success:
            self.metrics.successful_files += 1
        else:
            self.metrics.failed_files += 1
    
    def record_field_success_rates(self, extracted_data: List[Dict[str, str]]):
        """
        Record field-level success rates
        Determines success based on whether field has actual value vs "Information not found"
        """
        for record in extracted_data:
            for field in self.tracked_fields:
                self.metrics.field_success_rates[field]["total"] += 1
                
                if field in record and record[field] != "Information not found":
                    # Additional validation for some fields
                    if self._is_valid_field_value(field, record[field]):
                        self.metrics.field_success_rates[field]["success"] += 1
    
    def record_extractor_performance(self, field_results: Dict[str, Any]):
        """Record which extractors are most successful"""
        for field, result in field_results.items():
            if hasattr(result, 'extractor_id') and hasattr(result, 'confidence'):
                extractor_id = result.extractor_id
                
                self.metrics.extractor_performance[extractor_id]["total"] += 1
                
                # Consider successful if confidence > 0.5
                if result.confidence > 0.5:
                    self.metrics.extractor_performance[extractor_id]["success"] += 1
    
    def get_tat_analysis(self) -> Dict[str, Any]:
        """
        Get Turn-Around-Time analysis
        Returns detailed timing breakdown
        """
        analysis = {
            "summary": {
                "total_files": self.metrics.total_files,
                "total_time_seconds": round(self.metrics.total_processing_time, 2),
                "average_time_per_file": round(self.metrics.average_processing_time, 2),
                "files_per_minute": round(60 / self.metrics.average_processing_time, 2) if self.metrics.average_processing_time > 0 else 0
            },
            "stage_breakdown": {},
            "performance_classification": self._classify_performance()
        }
        
        # Calculate stage averages
        for stage, times in self.metrics.stage_times.items():
            if times:
                analysis["stage_breakdown"][stage] = {
                    "average_seconds": round(sum(times) / len(times), 3),
                    "min_seconds": round(min(times), 3),
                    "max_seconds": round(max(times), 3),
                    "total_calls": len(times)
                }
        
        return analysis
    
    def get_field_success_analysis(self) -> Dict[str, Any]:
        """
        Get field-level extraction success analysis
        """
        analysis = {
            "overall_success_rate": 0.0,
            "field_breakdown": {},
            "problem_fields": [],
            "top_performing_fields": []
        }
        
        success_rates = []
        field_performance = []
        
        for field, stats in self.metrics.field_success_rates.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                success_rates.append(success_rate)
                
                field_info = {
                    "field": field,
                    "success_rate": round(success_rate, 3),
                    "successful": stats["success"],
                    "total": stats["total"]
                }
                
                analysis["field_breakdown"][field] = field_info
                field_performance.append((field, success_rate))
                
                # Identify problem fields (< 50% success)
                if success_rate < 0.5:
                    analysis["problem_fields"].append(field_info)
        
        # Calculate overall success rate
        if success_rates:
            analysis["overall_success_rate"] = round(sum(success_rates) / len(success_rates), 3)
        
        # Top performing fields (> 80% success)
        field_performance.sort(key=lambda x: x[1], reverse=True)
        for field, rate in field_performance[:5]:
            if rate > 0.8:
                analysis["top_performing_fields"].append({
                    "field": field,
                    "success_rate": round(rate, 3)
                })
        
        return analysis
    
    def get_extractor_analysis(self) -> Dict[str, Any]:
        """
        Analyze extractor performance
        """
        analysis = {
            "extractor_breakdown": {},
            "best_extractors": [],
            "underperforming_extractors": []
        }
        
        extractor_performance = []
        
        for extractor, stats in self.metrics.extractor_performance.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                
                extractor_info = {
                    "extractor": extractor,
                    "success_rate": round(success_rate, 3),
                    "successful": stats["success"],
                    "total": stats["total"]
                }
                
                analysis["extractor_breakdown"][extractor] = extractor_info
                extractor_performance.append((extractor, success_rate, stats["total"]))
        
        # Sort by success rate and usage
        extractor_performance.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Best extractors (> 70% success rate and used at least 5 times)
        for extractor, rate, total in extractor_performance:
            if rate > 0.7 and total >= 5:
                analysis["best_extractors"].append({
                    "extractor": extractor,
                    "success_rate": round(rate, 3),
                    "usage_count": total
                })
            elif rate < 0.3 and total >= 5:
                analysis["underperforming_extractors"].append({
                    "extractor": extractor,
                    "success_rate": round(rate, 3),
                    "usage_count": total
                })
        
        return analysis
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        session_duration = time.time() - self.session_start_time
        
        report = {
            "session_info": {
                "start_time": self.session_start_time,
                "duration_seconds": round(session_duration, 2),
                "files_processed": self.metrics.total_files,
                "success_count": self.metrics.successful_files,
                "failure_count": self.metrics.failed_files,
                "overall_success_rate": round(self.metrics.successful_files / max(self.metrics.total_files, 1), 3)
            },
            "tat_analysis": self.get_tat_analysis(),
            "field_success_analysis": self.get_field_success_analysis(),
            "extractor_analysis": self.get_extractor_analysis()
        }
        
        return report
    
    def export_metrics(self, output_path: str):
        """Export metrics to JSON file"""
        try:
            report = self.generate_full_report()
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
    
    def print_summary(self):
        """Print a summary of metrics to console"""
        report = self.generate_full_report()
        
        print("\n" + "="*50)
        print("PROCESSING METRICS SUMMARY")
        print("="*50)
        
        session = report["session_info"]
        print(f"Files Processed: {session['files_processed']}")
        print(f"Success Rate: {session['overall_success_rate']:.1%}")
        print(f"Total Time: {session['duration_seconds']:.2f}s")
        
        tat = report["tat_analysis"]
        print(f"Avg Time/File: {tat['summary']['average_time_per_file']:.2f}s")
        print(f"Throughput: {tat['summary']['files_per_minute']:.1f} files/minute")
        
        field_analysis = report["field_success_analysis"]
        print(f"Field Success Rate: {field_analysis['overall_success_rate']:.1%}")
        
        if field_analysis["problem_fields"]:
            print(f"Problem Fields ({len(field_analysis['problem_fields'])}):")
            for field_info in field_analysis["problem_fields"][:3]:
                print(f"  - {field_info['field']}: {field_info['success_rate']:.1%}")
        
        print("="*50)
    
    def _classify_performance(self) -> str:
        """Classify overall performance"""
        avg_time = self.metrics.average_processing_time
        
        if avg_time < 0.5:
            return "Fast (< 0.5s/file)"
        elif avg_time < 1.5:
            return "Medium (0.5-1.5s/file)"
        elif avg_time < 5.0:
            return "Slow (1.5-5s/file)"
        else:
            return "Very Slow (> 5s/file)"
    
    def _is_valid_field_value(self, field: str, value: str) -> bool:
        """Additional validation for field values"""
        if not value or value.strip() == "":
            return False
        
        # Field-specific validation
        if "NPI" in field:
            return len(''.join(filter(str.isdigit, value))) == 10
        elif field == "TIN":
            return len(''.join(filter(str.isdigit, value))) == 9
        elif "Phone" in field or "Fax" in field:
            return len(''.join(filter(str.isdigit, value))) >= 10
        elif "Date" in field:
            return "/" in value or "-" in value
        
        return True