#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV refinement script.
Split the LLM_rephrase column into separate rule and diagnosis columns.
"""

import pandas as pd
import re
import argparse
from datetime import datetime

class CSVRefiner:
    def __init__(self, input_csv_path):
        """
        Initialize the CSV refiner.

        Args:
            input_csv_path (str): input CSV file path
        """
        self.input_csv_path = input_csv_path
        self.df = None

    def load_data(self):
        """Load the CSV data."""
        print(f"Loading data: {self.input_csv_path}")
        self.df = pd.read_csv(self.input_csv_path)
        print(f"Loaded {len(self.df)} rows")

    def parse_llm_rephrase(self, llm_rephrase):
        """
        Parse the LLM_rephrase field and extract rule and diagnosis.

        Args:
            llm_rephrase (str): content of the LLM_rephrase field

        Returns:
            tuple: (rule, diagnosis)
        """
        if pd.isna(llm_rephrase) or llm_rephrase == '':
            return '', ''

        llm_rephrase = str(llm_rephrase).strip()

        # Match the "Rule: ... | Result: ..." format
        pattern = r'Rule:\s*(.*?)\s*\|\s*Result:\s*(.*)'
        match = re.search(pattern, llm_rephrase, re.DOTALL)

        if match:
            rule = match.group(1).strip()
            diagnosis = match.group(2).strip()
            return rule, diagnosis
        else:
            # Fall back if the standard format does not match
            print(f"Warning: unable to parse format: {llm_rephrase[:100]}...")
            return llm_rephrase, ''

    def refine_data(self):
        """Refine the data by splitting the LLM_rephrase column."""
        print("Processing the LLM_rephrase column...")

        # Check that the LLM_rephrase column exists
        if 'LLM_rephrase' not in self.df.columns:
            print("Error: LLM_rephrase column not found")
            return False

        # Build new rule and diagnosis columns
        rules = []
        diagnoses = []

        for idx, row in self.df.iterrows():
            llm_rephrase = row['LLM_rephrase']
            rule, diagnosis = self.parse_llm_rephrase(llm_rephrase)
            rules.append(rule)
            diagnoses.append(diagnosis)

            # Print progress every 100 rows
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(self.df)} rows")

        # Add the new columns
        self.df['rule'] = rules
        self.df['diagnosis'] = diagnoses

        # Parsing statistics
        valid_rules = sum(1 for rule in rules if rule != '')
        valid_diagnoses = sum(1 for diagnosis in diagnoses if diagnosis != '')

        print(f"Parsing complete:")
        print(f"  - Valid rules: {valid_rules}/{len(self.df)}")
        print(f"  - Valid diagnoses: {valid_diagnoses}/{len(self.df)}")

        return True

    def save_refined_data(self, output_path):
        """Save the refined data."""
        print(f"Saving refined data to: {output_path}")
        self.df.to_csv(output_path, index=False)
        print(f"Saved {len(self.df)} rows")

    def generate_report(self, output_path):
        """Generate a processing report."""
        report_path = output_path.replace('.csv', '_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CSV refinement report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Input file: {self.input_csv_path}\n")
            f.write(f"Output file: {output_path}\n")
            f.write(f"Total records: {len(self.df)}\n\n")

            # Validity statistics for rule and diagnosis
            valid_rules = sum(1 for rule in self.df['rule'] if rule != '')
            valid_diagnoses = sum(1 for diagnosis in self.df['diagnosis'] if diagnosis != '')

            f.write("Parsing statistics:\n")
            f.write(f"  - Valid rules: {valid_rules} ({valid_rules/len(self.df)*100:.1f}%)\n")
            f.write(f"  - Valid diagnoses: {valid_diagnoses} ({valid_diagnoses/len(self.df)*100:.1f}%)\n")
            f.write(f"  - Invalid rules: {len(self.df) - valid_rules}\n")
            f.write(f"  - Invalid diagnoses: {len(self.df) - valid_diagnoses}\n\n")

            # Show a few parsing examples
            f.write("Parsing examples:\n")
            for i in range(min(5, len(self.df))):
                row = self.df.iloc[i]
                f.write(f"Example {i+1}:\n")
                f.write(f"  Original LLM_rephrase: {row['LLM_rephrase'][:100]}...\n")
                f.write(f"  Parsed rule: {row['rule'][:100]}...\n")
                f.write(f"  Parsed diagnosis: {row['diagnosis']}\n")
                f.write("\n")

            # Show the diagnosis distribution
            f.write("Diagnosis distribution (top 10):\n")
            diagnosis_counts = self.df['diagnosis'].value_counts()
            for diagnosis, count in diagnosis_counts.head(10).items():
                if diagnosis != '':
                    f.write(f"  {diagnosis}: {count}\n")

        print(f"Report generated: {report_path}")

    def show_sample_results(self):
        """Show a few processed results."""
        print("\nProcessed result examples:")
        print("=" * 80)

        for i in range(min(3, len(self.df))):
            row = self.df.iloc[i]
            print(f"Example {i+1}:")
            print(f"  Original LLM_rephrase: {row['LLM_rephrase'][:100]}...")
            print(f"  Parsed rule: {row['rule'][:100]}...")
            print(f"  Parsed diagnosis: {row['diagnosis']}")
            print("-" * 80)

def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Refine CSV data by splitting the LLM_rephrase column')
    parser.add_argument('--input', '-i', default='filtered_final_data.csv',
                       help='Input CSV file path (default: filtered_final_data.csv)')
    parser.add_argument('--output', '-o', default='refined_data.csv',
                       help='Output CSV file path (default: refined_data.csv)')

    args = parser.parse_args()

    # Create the refiner
    refiner = CSVRefiner(args.input)

    try:
        # Load data
        refiner.load_data()

        # Refine data
        if refiner.refine_data():
            # Save results
            refiner.save_refined_data(args.output)

            # Generate report
            refiner.generate_report(args.output)

            # Show examples
            refiner.show_sample_results()

            print(f"\nDone.")
            print(f"Output file: {args.output}")
            print(f"Report file: {args.output.replace('.csv', '_report.txt')}")
        else:
            print("Processing failed")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
