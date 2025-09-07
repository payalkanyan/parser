"""
CLI Entry Point
"""

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from .ingest.eml_parser import EMLParser
from .extract.extraction_engine import ExtractionEngine
from .export.excel import ExcelExporter
from .observability.metrics import MetricsCollector
from .observability.trace import TraceLogger


class RosterParserCLI:
    """Main CLI interface for roster parsing"""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.trace = TraceLogger()
        self.parser = EMLParser()
        self.engine = ExtractionEngine()
        self.exporter = ExcelExporter()
        
        # Get template path - stored in templates directory
        self.template_path = Path(__file__).parent.parent / "templates" / "Output Format.xlsx"
    
    def parse_single(self, eml_path: Path, output_path: Path) -> bool:
        """Parse single EML file"""
        start_time = time.time()
        
        try:
            self.trace.start_trace(str(eml_path))
            
            # Stage 1: Parse EML
            self.trace.log_stage("mime_parsing")
            parsed_content = self.parser.parse_eml(eml_path)
            
            # Stage 2: Extract data
            self.trace.log_stage("extraction")
            extracted_data = self.engine.extract_all_fields(parsed_content)
            
            # Stage 3: Export to Excel
            self.trace.log_stage("export")
            success = self.exporter.export_to_excel(extracted_data, self.template_path, output_path)
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics.record_processing_time(processing_time)
            self.metrics.record_field_success_rates(extracted_data)
            
            self.trace.end_trace(success)
            
            print(f"✓ Processed {eml_path.name} -> {output_path.name} ({processing_time:.2f}s)")
            return success
            
        except Exception as e:
            print(f"✗ Error processing {eml_path.name}: {str(e)}")
            self.trace.log_error(str(e))
            return False
    
    def parse_batch(self, eml_dir: Path, output_dir: Path, workers: int = 4) -> bool:
        """Parse multiple EML files in parallel"""
        eml_files = list(eml_dir.glob("*.eml"))
        if not eml_files:
            print(f"No .eml files found in {eml_dir}")
            return False
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {len(eml_files)} files with {workers} workers...")
        batch_start_time = time.time()
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for eml_file in eml_files:
                output_file = output_dir / f"{eml_file.stem}_output.xlsx"
                future = executor.submit(_batch_worker_function_with_metrics, eml_file, self.template_path, output_file)
                futures.append((eml_file, future))
            
            success_count = 0
            all_results = []
            
            for eml_file, future in futures:
                try:
                    result = future.result()
                    if result['success']:
                        success_count += 1
                        # Record metrics from worker
                        self.metrics.record_processing_time(result['processing_time'])
                        self.metrics.record_field_success_rates(result['extracted_data'])
                        self.metrics.record_file_success(True)
                        
                        # Record stage timings if available
                        for stage, timing in result.get('stage_timings', {}).items():
                            self.metrics.record_stage_time(stage, timing)
                    else:
                        self.metrics.record_file_success(False)
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"✗ Error processing {eml_file.name}: {str(e)}")
                    self.metrics.record_file_success(False)
        
        batch_duration = time.time() - batch_start_time
        
        # Print comprehensive analysis
        print(f"\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Files Processed: {success_count}/{len(eml_files)} ({success_count/len(eml_files)*100:.1f}% success rate)")
        print(f"Total Batch Time: {batch_duration:.2f}s")
        
        # Print TAT Analysis
        tat_analysis = self.metrics.get_tat_analysis()
        print(f"\nTAT ANALYSIS:")
        print(f"  Average time per file: {tat_analysis['summary']['average_time_per_file']:.2f}s")
        print(f"  Throughput: {tat_analysis['summary']['files_per_minute']:.1f} files/minute")
        print(f"  Performance: {tat_analysis['performance_classification']}")
        
        # Print stage breakdown if available
        if tat_analysis['stage_breakdown']:
            print(f"\n  Stage Breakdown:")
            for stage, timing in tat_analysis['stage_breakdown'].items():
                print(f"    {stage}: {timing['average_seconds']:.3f}s avg ({timing['total_calls']} calls)")
        
        # Print field success analysis
        field_analysis = self.metrics.get_field_success_analysis()
        print(f"\nFIELD SUCCESS ANALYSIS:")
        print(f"  Overall field success rate: {field_analysis['overall_success_rate']:.1%}")
        
        if field_analysis['problem_fields']:
            print(f"  Problem fields ({len(field_analysis['problem_fields'])}):")
            for field_info in field_analysis['problem_fields'][:5]:
                print(f"    - {field_info['field']}: {field_info['success_rate']:.1%} ({field_info['successful']}/{field_info['total']})")
        
        if field_analysis['top_performing_fields']:
            print(f"  Top performing fields:")
            for field_info in field_analysis['top_performing_fields'][:3]:
                print(f"    + {field_info['field']}: {field_info['success_rate']:.1%}")
        
        print("="*60)
        
        return success_count == len(eml_files)
    
    def _worker_process(self, eml_path: Path, template_path: Path, output_path: Path) -> bool:
        """Worker process for batch processing"""
        # Create new instances for each worker to avoid shared state
        parser = EMLParser()
        engine = ExtractionEngine()
        exporter = ExcelExporter()
        
        try:
            parsed_content = parser.parse_eml(eml_path)
            extracted_data = engine.extract_all_fields(parsed_content)
            return exporter.export_to_excel(extracted_data, template_path, output_path)
        except Exception:
            return False


def _batch_worker_function_with_metrics(eml_path: Path, template_path: Path, output_path: Path) -> dict:
    start_time = time.time()
    stage_timings = {}
    
    try:
        parser = EMLParser()
        engine = ExtractionEngine()
        exporter = ExcelExporter()
        
        parse_start = time.time()
        parsed_content = parser.parse_eml(eml_path)
        stage_timings['mime_parsing'] = time.time() - parse_start
        
        extract_start = time.time()
        extracted_data = engine.extract_all_fields(parsed_content)
        stage_timings['extraction'] = time.time() - extract_start
        
        export_start = time.time()
        success = exporter.export_to_excel(extracted_data, template_path, output_path)
        stage_timings['export'] = time.time() - export_start
        
        processing_time = time.time() - start_time
        
        if success:
            print(f"✓ {eml_path.name} -> {output_path.name} ({processing_time:.2f}s)")
        
        return {
            'success': success,
            'processing_time': processing_time,
            'stage_timings': stage_timings,
            'extracted_data': extracted_data,
            'file_name': eml_path.name
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"✗ {eml_path.name}: {str(e)}")
        return {
            'success': False,
            'processing_time': processing_time,
            'stage_timings': stage_timings,
            'extracted_data': [],
            'file_name': eml_path.name,
            'error': str(e)
        }




def main():
    parser = argparse.ArgumentParser(
        description="HiLabs Roster Parser - Extract provider data from EML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python -m src.cli parse --eml sample.eml --out output.xlsx
  
  # Batch processing  
  python -m src.cli batch --eml-dir ./emails --out-dir ./outputs --workers 4
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single file parsing
    parse_parser = subparsers.add_parser('parse', help='Parse single EML file')
    parse_parser.add_argument('--eml', required=True, type=Path, help='Input EML file')
    parse_parser.add_argument('--out', required=True, type=Path, help='Output Excel file')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Parse multiple EML files')
    batch_parser.add_argument('--eml-dir', required=True, type=Path, help='Directory containing EML files')
    batch_parser.add_argument('--out-dir', required=True, type=Path, help='Output directory')
    batch_parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = RosterParserCLI()
    
    if args.command == 'parse':
        if not args.eml.exists():
            print(f"Error: EML file not found: {args.eml}")
            return 1
        
        success = cli.parse_single(args.eml, args.out)
        return 0 if success else 1
    
    elif args.command == 'batch':
        if not args.eml_dir.exists():
            print(f"Error: EML directory not found: {args.eml_dir}")
            return 1
        
        success = cli.parse_batch(args.eml_dir, args.out_dir, args.workers)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())