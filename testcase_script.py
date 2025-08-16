import json
import os
import re
import random
import string
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import List, Dict, Any, Tuple, Optional
import sys
import shutil

class SPOJTestCaseGenerator:
    def __init__(self, questions_folder: str, output_folder: str = "questions_with_testcases", max_workers: int = 8):
        self.questions_folder = Path(questions_folder)
        self.output_folder = Path(output_folder)
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.processed_count = 0
        self.failed_count = 0
        
        # Create output folder
        self.setup_output_folder()
        
        print(f"🧪 SPOJ Test Case Generator Initialized")
        print(f"   📁 Input folder: {self.questions_folder}")
        print(f"   📁 Output folder: {self.output_folder}")
        print(f"   💪 Workers: {self.max_workers}")
        print(f"   📝 Mode: Creating new folder with test cases")

    def setup_output_folder(self) -> None:
        """Create output folder structure."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.output_folder}_{timestamp}"
        self.output_folder = Path(folder_name)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created output folder: {self.output_folder}")
        
        # Create metadata file
        metadata = {
            "generation_started": timestamp,
            "source_folder": str(self.questions_folder),
            "output_folder": str(self.output_folder),
            "total_questions": 0,
            "successfully_processed": 0,
            "failed": 0,
            "test_cases_per_question": 20
        }
        
        metadata_file = self.output_folder / "generation_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def parse_sample_input_output(self, sample_input: str, sample_output: str) -> List[Tuple[str, str]]:
        """Parse the sample input/output which might be in 'Input:...Output:...' format."""
        samples = []
        
        # Handle sample_input
        if sample_input and sample_input != 'Not found':
            if 'Input:' in sample_input:
                # Format could be "Input:213Output:16" or just "Input:213"
                if 'Output:' in sample_input:
                    parts = sample_input.split('Output:')
                    if len(parts) >= 2:
                        actual_input = parts[0].replace('Input:', '').strip()
                        actual_output = parts[1].strip()
                        if actual_input:
                            samples.append((actual_input, actual_output))
                else:
                    actual_input = sample_input.replace('Input:', '').strip()
                    if actual_input:
                        samples.append((actual_input, ""))
            else:
                # Plain input without "Input:" prefix
                actual_input = sample_input.strip()
                if actual_input:
                    samples.append((actual_input, ""))
        
        # Handle sample_output
        if sample_output and sample_output != 'Not found':
            if 'Input:' in sample_output and 'Output:' in sample_output:
                # This is another example in format "Input:42414Output:109"
                parts = sample_output.split('Output:')
                if len(parts) >= 2:
                    sample_input_2 = parts[0].replace('Input:', '').strip()
                    sample_output_2 = parts[1].strip()
                    if sample_input_2:
                        samples.append((sample_input_2, sample_output_2))
            elif not any(sample_output == out for _, out in samples):
                # If sample_output is just output and we have input without output
                if samples and not samples[0][1]:
                    # Update first sample with this output
                    samples[0] = (samples[0][0], sample_output.strip())
                else:
                    # Standalone output (shouldn't happen but handle gracefully)
                    pass
        
        return samples if samples else [("", "")]

    def analyze_input_format(self, problem_text: str, sample_input: str, sample_output: str) -> Dict[str, Any]:
        """Analyze problem text and sample input to understand the format."""
        format_info = {
            'input_lines': [],
            'constraints': {},
            'data_types': [],
            'special_patterns': [],
            'array_size': None,
            'is_array_problem': False
        }
        
        # Parse sample input to understand structure
        samples = self.parse_sample_input_output(sample_input, sample_output)
        if samples and samples[0][0]:
            lines = samples[0][0].strip().split('\n')
            format_info['input_lines'] = lines
            
            # Analyze each line
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line contains numbers
                numbers = re.findall(r'-?\d+(?:\.\d+)?', line)
                if numbers:
                    format_info['data_types'].append({
                        'line': i,
                        'type': 'numeric',
                        'values': numbers,
                        'count': len(numbers)
                    })
                    
                    # If first line is a single number, might be array size
                    if i == 0 and len(numbers) == 1:
                        try:
                            format_info['array_size'] = int(numbers[0])
                            format_info['is_array_problem'] = True
                        except ValueError:
                            pass
                
                # Check for strings
                words = line.split()
                if any(not word.replace('.', '').replace('-', '').isdigit() for word in words):
                    format_info['data_types'].append({
                        'line': i,
                        'type': 'string',
                        'words': words
                    })
        
        # Extract constraints from problem text
        if problem_text and problem_text != 'Not found':
            # Look for common constraint patterns
            constraint_patterns = [
                r'1\s*[≤<=]\s*([A-Za-z]+)\s*[≤<=]\s*(\d+(?:\s*[\*x]\s*10\s*\^\s*\d+|\s*\d+)*)',
                r'([A-Za-z]+)\s*[≤<=]\s*(\d+(?:\s*[\*x]\s*10\s*\^\s*\d+|\s*\d+)*)',
                r'(\d+)\s*[≤<=]\s*([A-Za-z]+)\s*[≤<=]\s*(\d+(?:\s*[\*x]\s*10\s*\^\s*\d+|\s*\d+)*)',
                r'1\s*≤\s*([A-Za-z]+)\s*≤\s*(\d+(?:\s*[\*x]\s*10\s*\^\s*\d+|\s*\d+)*)',
                r'([A-Za-z]+)\s*≤\s*(\d+(?:\s*[\*x]\s*10\s*\^\s*\d+|\s*\d+)*)'
            ]
            
            for pattern in constraint_patterns:
                matches = re.findall(pattern, problem_text, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 2:
                        var_name = match[0] if len(match) == 2 else match[1]
                        max_val_str = match[1] if len(match) == 2 else match[2]
                        
                        # Parse values like "500 000", "10^8", "10 ^ 8"
                        try:
                            if '10' in max_val_str and ('^' in max_val_str or '*' in max_val_str):
                                # Handle scientific notation like "10^8", "10 ^ 8", "10*10^5"
                                exp_match = re.search(r'10\s*[\^\*x]\s*(\d+)', max_val_str)
                                if exp_match:
                                    exp = int(exp_match.group(1))
                                    max_val = 10 ** exp
                                else:
                                    # Try to find just the number after 10^
                                    exp_match = re.search(r'(\d+)', max_val_str.split('10')[-1])
                                    if exp_match:
                                        exp = int(exp_match.group(1))
                                        max_val = 10 ** exp
                                    else:
                                        max_val = int(re.sub(r'[^\d]', '', max_val_str))
                            else:
                                # Remove spaces and non-digits except for the number itself
                                cleaned = re.sub(r'[^\d]', '', max_val_str)
                                max_val = int(cleaned) if cleaned else 1000
                            
                            format_info['constraints'][var_name.upper()] = min(max_val, 10**6)  # Cap for safety
                        except (ValueError, AttributeError):
                            pass
        
        return format_info

    def generate_numeric_value(self, constraints: Dict[str, int], var_name: str = 'N', 
                             min_val: int = 1, test_case_type: str = 'medium') -> int:
        """Generate a numeric value based on constraints and test case type."""
        var_name = var_name.upper()
        max_constraint = constraints.get(var_name, 1000)
        
        # Different ranges based on test case type
        if test_case_type == 'small':
            max_val = min(max_constraint, 10)
            return random.randint(min_val, max_val)
        elif test_case_type == 'medium':
            max_val = min(max_constraint, 100)
            return random.randint(min_val, max_val)
        elif test_case_type == 'large':
            max_val = min(max_constraint, max_constraint // 2)
            return random.randint(max(min_val, max_val // 10), max_val)
        elif test_case_type == 'max':
            return min(max_constraint, 10**6)
        elif test_case_type == 'edge':
            return random.choice([min_val, min_val + 1, max_constraint - 1, max_constraint])
        else:
            return random.randint(min_val, min(max_constraint, 1000))

    def generate_array(self, size: int, element_type: str = 'int', min_val: int = 1, 
                      max_val: int = 1000, test_case_type: str = 'medium') -> List[int]:
        """Generate an array of values."""
        if test_case_type == 'small':
            max_val = min(max_val, 10)
        elif test_case_type == 'large':
            max_val = min(max_val, 100000)
        elif test_case_type == 'max':
            max_val = min(max_val, 10**8)
        elif test_case_type == 'edge':
            # Create arrays with edge values
            return [random.choice([min_val, max_val, random.randint(min_val, max_val)]) for _ in range(size)]
        
        if element_type == 'int':
            return [random.randint(min_val, max_val) for _ in range(size)]
        elif element_type == 'float':
            return [round(random.uniform(min_val, max_val), 2) for _ in range(size)]
        return []

    def generate_test_cases_for_problem(self, problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate 20 test cases for a single problem."""
        try:
            problem_code = problem_data.get('problem_code', 'UNKNOWN')
            title = problem_data.get('title', 'Unknown Problem')
            problem_text = problem_data.get('text', '')
            sample_input = problem_data.get('sample_input', '')
            sample_output = problem_data.get('sample_output', '')
            
            print(f"🧪 Generating test cases for {problem_code}: {title[:50]}...")
            
            # Analyze input format
            format_info = self.analyze_input_format(problem_text, sample_input, sample_output)
            
            test_cases = []
            
            # Parse existing samples
            samples = self.parse_sample_input_output(sample_input, sample_output)
            
            # Add original samples as test cases
            for i, (inp, out) in enumerate(samples):
                if inp and inp.strip():
                    test_cases.append({
                        'input': inp.strip(),
                        'expected_output': out.strip() if out else 'Unknown',
                        'type': f'sample_{i+1}',
                        'description': f'Original sample {i+1} from problem statement'
                    })
            
            # Generate additional test cases to reach 20 total
            remaining_cases = max(0, 20 - len(test_cases))
            
            # Define test case distribution
            case_types = []
            if remaining_cases > 0:
                case_types.extend(['small'] * min(3, remaining_cases))
            if remaining_cases > 3:
                case_types.extend(['medium'] * min(5, remaining_cases - 3))
            if remaining_cases > 8:
                case_types.extend(['large'] * min(7, remaining_cases - 8))
            if remaining_cases > 15:
                case_types.extend(['max'] * min(2, remaining_cases - 15))
            if remaining_cases > 17:
                case_types.extend(['edge'] * min(2, remaining_cases - 17))
            if remaining_cases > 19:
                case_types.extend(['random'] * (remaining_cases - 19))
            
            for i, case_type in enumerate(case_types):
                test_case = self.generate_single_test_case(format_info, len(test_cases) + 1, case_type)
                test_cases.append(test_case)
            
            return test_cases
            
        except Exception as e:
            print(f"❌ Error generating test cases for {problem_data.get('problem_code', 'unknown')}: {e}")
            return []

    def generate_single_test_case(self, format_info: Dict[str, Any], case_number: int, 
                                case_type: str = 'medium') -> Dict[str, str]:
        """Generate a single test case based on format analysis."""
        test_input_lines = []
        description = f"Generated {case_type} test case"
        
        if format_info['is_array_problem'] and format_info['array_size'] is not None:
            # Handle array-based problems
            if case_type == 'small':
                n = random.randint(1, 5)
                description = "Small array (N ≤ 5)"
            elif case_type == 'medium':
                n = random.randint(5, 20)
                description = "Medium array (5 < N ≤ 20)"
            elif case_type == 'large':
                n = random.randint(20, min(500, format_info['constraints'].get('N', 500)))
                description = "Large array (20 < N ≤ 500)"
            elif case_type == 'max':
                n = min(format_info['constraints'].get('N', 1000), 1000)
                description = f"Maximum size array (N = {n})"
            elif case_type == 'edge':
                n = random.choice([1, 2, min(format_info['constraints'].get('N', 1000), 1000)])
                description = f"Edge case array (N = {n})"
            else:
                n = random.randint(1, 50)
                description = "Random array"
            
            test_input_lines.append(str(n))
            
            # Generate array elements
            element_constraints = format_info['constraints'].copy()
            max_element = element_constraints.get('elements', 10**8)  # Default max for array elements
            
            array_elements = self.generate_array(n, 'int', 1, max_element, case_type)
            for element in array_elements:
                test_input_lines.append(str(element))
        
        elif not format_info['data_types']:
            # Fallback: generate basic test case
            if case_type == 'small':
                test_input_lines.append(str(random.randint(1, 5)))
                description = "Small value (≤ 5)"
            elif case_type == 'medium':
                test_input_lines.append(str(random.randint(5, 50)))
                description = "Medium value (5-50)"
            elif case_type == 'large':
                test_input_lines.append(str(random.randint(50, 1000)))
                description = "Large value (50-1000)"
            elif case_type == 'max':
                test_input_lines.append(str(random.randint(1000, 10000)))
                description = "Maximum value (≥ 1000)"
            else:
                test_input_lines.append(str(random.randint(1, 100)))
                description = "Random value"
        
        else:
            # Generate based on detected format
            for line_idx, data_type_info in enumerate(format_info['data_types']):
                line_type = data_type_info['type']
                
                if line_type == 'numeric':
                    count = data_type_info['count']
                    
                    # Generate multiple values on same line
                    generated_values = []
                    for _ in range(count):
                        val = self.generate_numeric_value(format_info['constraints'], 'N', 1, case_type)
                        generated_values.append(str(val))
                    test_input_lines.append(' '.join(generated_values))
                
                elif line_type == 'string':
                    words = data_type_info['words']
                    if len(words) == 1:
                        # Single word/string
                        length = random.randint(1, min(20, len(words[0]) * 2))
                        test_input_lines.append(self.generate_string_value(length))
                    else:
                        # Multiple words
                        generated_words = []
                        for word in words:
                            if word.isdigit():
                                val = self.generate_numeric_value(format_info['constraints'], 'N', 1, case_type)
                                generated_words.append(str(val))
                            else:
                                length = random.randint(1, max(5, len(word)))
                                generated_words.append(self.generate_string_value(length))
                        test_input_lines.append(' '.join(generated_words))
        
        return {
            'input': '\n'.join(test_input_lines),
            'expected_output': 'To be computed',
            'type': f'{case_type}_case_{case_number}',
            'description': description
        }

    def generate_string_value(self, length: Optional[int] = None, charset: str = 'lowercase') -> str:
        """Generate a random string value."""
        if length is None:
            length = random.randint(1, 10)
        
        if charset == 'lowercase':
            chars = string.ascii_lowercase
        elif charset == 'uppercase':
            chars = string.ascii_uppercase
        elif charset == 'alpha':
            chars = string.ascii_letters
        elif charset == 'alnum':
            chars = string.ascii_letters + string.digits
        else:
            chars = string.ascii_lowercase
        
        return ''.join(random.choice(chars) for _ in range(length))

    def process_single_question(self, question_file: Path) -> bool:
        """Process a single question file and generate test cases."""
        try:
            # Read original question
            with open(question_file, 'r', encoding='utf-8') as f:
                problem_data = json.load(f)
            
            # Generate test cases
            test_cases = self.generate_test_cases_for_problem(problem_data)
            
            if test_cases:
                # Add test cases to the problem data
                enhanced_problem_data = problem_data.copy()
                enhanced_problem_data['test_cases'] = test_cases
                enhanced_problem_data['test_cases_generated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                enhanced_problem_data['total_test_cases'] = len(test_cases)
                
                # Save to new file in output folder
                output_file = self.output_folder / question_file.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_problem_data, f, indent=2, ensure_ascii=False)
                
                with self.lock:
                    self.processed_count += 1
                
                return True
            else:
                with self.lock:
                    self.failed_count += 1
                return False
                
        except Exception as e:
            print(f"❌ Error processing {question_file}: {e}")
            with self.lock:
                self.failed_count += 1
            return False

    def generate_all_test_cases(self) -> None:
        """Generate test cases for all questions in the folder."""
        print(f"\n🚀 Starting test case generation...")
        print(f"📁 Scanning folder: {self.questions_folder}")
        
        # Find all question files
        question_files = list(self.questions_folder.glob("q*.json"))
        
        if not question_files:
            print(f"❌ No question files found in {self.questions_folder}")
            return
        
        print(f"📊 Found {len(question_files)} question files")
        print(f"🎯 Target: Creating {len(question_files)} enhanced files with 20 test cases each")
        print(f"💾 Output: {self.output_folder}")
        
        start_time = time.time()
        
        # Process files in parallel
        print(f"\n⚡ Processing with {self.max_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_single_question, qfile): qfile 
                for qfile in question_files
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 50 == 0 or completed == len(question_files):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    progress = (completed / len(question_files)) * 100
                    eta = (elapsed / completed * (len(question_files) - completed)) if completed > 0 else 0
                    
                    print(f"  🔥 [{progress:5.1f}%] {completed}/{len(question_files)} | "
                          f"✅ {self.processed_count} | ❌ {self.failed_count} | "
                          f"{rate:.1f}/sec | ETA: {eta:.0f}s")
        
        # Update metadata
        self.update_final_metadata(len(question_files), time.time() - start_time)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎉 Test Case Generation Completed!")
        print("=" * 60)
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"✅ Successfully processed: {self.processed_count} questions")
        print(f"❌ Failed: {self.failed_count} questions")
        print(f"🧪 Total test cases generated: {self.processed_count * 20}")
        print(f"📁 Output folder: {self.output_folder}")
        
        # Show sample enhanced structure
        if self.processed_count > 0:
            self.show_sample_structure()

    def show_sample_structure(self) -> None:
        """Display sample structure and file statistics."""
        print(f"\n📋 Enhanced question structure:")
        print(f"{{")
        print(f'  "problem_code": "NORMA2",')
        print(f'  "title": "Norma",')
        print(f'  "tags": [],')
        print(f'  "text": "...",')
        print(f'  "sample_input": "Input:213Output:16",')
        print(f'  "sample_output": "Input:42414Output:109",')
        print(f'  "link": "...",')
        print(f'  "scraped_at": "...",')
        print(f'  "test_cases": [')
        print(f'    {{')
        print(f'      "input": "2\\n1\\n3",')
        print(f'      "expected_output": "16",')
        print(f'      "type": "sample_1",')
        print(f'      "description": "Original sample 1 from problem statement"')
        print(f'    }},')
        print(f'    {{')
        print(f'      "input": "3\\n5\\n2\\n8",')
        print(f'      "expected_output": "To be computed",')
        print(f'      "type": "small_case_3",')
        print(f'      "description": "Small array (N ≤ 5)"')
        print(f'    }},')
        print(f'    ... 18 more test cases')
        print(f'  ],')
        print(f'  "test_cases_generated_at": "2024-12-17 15:30:45",')
        print(f'  "total_test_cases": 20')
        print(f"}}")
        
        # Show first few generated files
        generated_files = list(self.output_folder.glob("q*.json"))
        if generated_files:
            print(f"\n📂 Sample generated files:")
            for i, file in enumerate(generated_files[:5]):
                file_size = file.stat().st_size
                print(f"  {file.name} ({file_size:,} bytes)")
            if len(generated_files) > 5:
                print(f"  ... and {len(generated_files) - 5} more files")
            
            total_size = sum(f.stat().st_size for f in generated_files)
            print(f"\n📊 Total output size: {total_size / (1024*1024):.1f} MB")

    def update_final_metadata(self, total_questions: int, duration: float) -> None:
        """Update metadata with final statistics."""
        try:
            metadata_file = self.output_folder / "generation_metadata.json"
            
            metadata = {
                "generation_started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generation_completed": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_folder": str(self.questions_folder),
                "output_folder": str(self.output_folder),
                "total_questions": total_questions,
                "successfully_processed": self.processed_count,
                "failed": self.failed_count,
                "test_cases_per_question": 20,
                "total_test_cases_generated": self.processed_count * 20,
                "generation_duration_seconds": duration,
                "success_rate": (self.processed_count / total_questions * 100) if total_questions > 0 else 0
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error updating metadata: {e}")


def main() -> None:
    """Main function to run the test case generator."""
    if len(sys.argv) < 2:
        print("Usage: python test_case_generator.py <questions_folder> [output_folder] [max_workers]")
        print("Example: python test_case_generator.py questions_20241217_143022 questions_with_testcases 8")
        print("\nNote: This creates a NEW folder with enhanced questions (original files unchanged)")
        return
    
    questions_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "questions_with_testcases"
    
    try:
        max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    except ValueError:
        print("❌ max_workers must be an integer")
        return
    
    # Validate input folder
    if not Path(questions_folder).exists():
        print(f"❌ Questions folder '{questions_folder}' does not exist!")
        return
    
    print("🧪🧪🧪 SPOJ TEST CASE GENERATOR 🧪🧪🧪")
    print("=" * 60)
    print(f"📁 Input folder: {questions_folder}")
    print(f"📁 Output folder: {output_folder} (timestamped)")
    print(f"💪 Workers: {max_workers}")
    print(f"🎯 Target: 20 test cases per question")
    print(f"🔄 Mode: CREATE NEW FOLDER (originals unchanged)")
    print("=" * 60)
    
    # Create generator and run
    generator = SPOJTestCaseGenerator(
        questions_folder=questions_folder,
        output_folder=output_folder,
        max_workers=max_workers
    )
    
    try:
        generator.generate_all_test_cases()
    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted by user")
        return
    
    print(f"\n🚀 Test case generation completed!")
    print(f"🔍 Check the generated files in: {generator.output_folder}")
    print(f"📝 Inspect a few files to verify test case quality before using")


if __name__ == "__main__":
    main()