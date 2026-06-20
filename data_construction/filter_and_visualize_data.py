#!/usr/bin/env python3
"""
Data filtering and visualization tool.

Features:
1. Filter out invalid rows whose LLM_rephrase is "Rule: NOT A RULE".
2. Generate an HTML page showing all valid records with their images and keys.
3. Support a second curation pass that excludes selected images.
4. Produce a new CSV file.
"""

import pandas as pd
import os
import shutil
from pathlib import Path
import json
from datetime import datetime

class DataFilterAndVisualizer:
    def __init__(self, csv_file_path, image_dir_path):
        """
        Initialize the data filter and visualizer.

        Args:
            csv_file_path (str): CSV file path
            image_dir_path (str): image directory path
        """
        self.csv_file_path = csv_file_path
        self.image_dir_path = image_dir_path
        self.df = None
        self.filtered_df = None
        self.excluded_images = set()

    def load_data(self):
        """Load the CSV data."""
        print(f"Loading data: {self.csv_file_path}")
        self.df = pd.read_csv(self.csv_file_path)
        print(f"Original rows: {len(self.df)}")

    def filter_invalid_data(self):
        """Filter out invalid rows (LLM_rephrase == 'Rule: NOT A RULE')."""
        print("Filtering out invalid rows...")
        self.filtered_df = self.df[self.df['LLM_rephrase'] != 'Rule: NOT A RULE'].copy()
        print(f"Rows after filtering: {len(self.filtered_df)}")
        print(f"Rows removed: {len(self.df) - len(self.filtered_df)}")

    def generate_image_gallery_html(self, output_html_path):
        """
        Generate an HTML gallery page.

        Args:
            output_html_path (str): output HTML file path
        """
        print("Generating the image gallery HTML page...")

        # Temporary directory for images
        os.makedirs("temp_images", exist_ok=True)

        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dermatology Image Curation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .stats {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .controls {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .image-card {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .image-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .image-card.selected {
            border: 3px solid #e74c3c;
            background-color: #fdf2f2;
        }
        .image-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .image-info {
            font-size: 12px;
            color: #666;
            margin-bottom: 10px;
        }
        .image-key {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
            word-break: break-all;
        }
        .label {
            color: #27ae60;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .text {
            color: #7f8c8d;
            font-size: 11px;
            margin-bottom: 5px;
        }
        .checkbox {
            margin-top: 10px;
        }
        .button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
            font-size: 14px;
        }
        .button:hover {
            background-color: #2980b9;
        }
        .button.danger {
            background-color: #e74c3c;
        }
        .button.danger:hover {
            background-color: #c0392b;
        }
        .button.success {
            background-color: #27ae60;
        }
        .button.success:hover {
            background-color: #229954;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Dermatology Image Curation</h1>
        <p>Select the images to exclude, then export the filtered data</p>
    </div>
    
    <div class="stats">
        <h3>Statistics</h3>
        <p>Total images: <span id="totalCount">0</span></p>
        <p>Selected: <span id="selectedCount">0</span></p>
        <p>Unselected: <span id="unselectedCount">0</span></p>
    </div>
    
    <div class="controls">
        <button class="button" onclick="selectAll()">Select all</button>
        <button class="button" onclick="deselectAll()">Deselect all</button>
        <button class="button success" onclick="exportSelection()">Export exclusion list</button>
        <button class="button danger" onclick="clearSelection()">Clear selection</button>
    </div>
    
    <div class="gallery" id="imageGallery">
"""

        # Build image cards
        for index, row in self.filtered_df.iterrows():
            image_key = row['image_key']
            label = row['label'] if pd.notna(row['label']) else 'No label'
            text = row['text'] if pd.notna(row['text']) else 'No text'
            llm_rephrase = row['LLM_rephrase'] if pd.notna(row['LLM_rephrase']) else 'No rephrase'

            # Build the image path
            image_path = os.path.join(self.image_dir_path, image_key)

            # Check that the image exists and copy it to the temp directory
            if os.path.exists(image_path):
                # Copy the image to the temp directory for display
                temp_image_path = f"temp_images/{image_key}"
                shutil.copy2(image_path, temp_image_path)

                # Encode the image as base64
                try:
                    import base64
                    with open(temp_image_path, 'rb') as image_file:
                        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
                    img_src = f"data:image/png;base64,{base64_data}"
                except Exception as e:
                    print(f"Warning: failed to encode image {image_key}: {e}")
                    # Fall back to the file path if base64 encoding fails
                    encoded_image_key = image_key.replace("'", "%27").replace(" ", "%20")
                    img_src = f"temp_images/{encoded_image_key}"
            else:
                print(f"Warning: image file not found: {image_path}")
                # Use a placeholder
                img_src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjhmOWZhIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzZjNzU3ZCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuaPkuS7tuWbvueJhzwvdGV4dD48L3N2Zz4="

            html_content += f"""
        <div class="image-card" data-image-key="{image_key}" data-row-index="{index}">
            <img src="{img_src}" alt="{image_key}" onerror="this.style.display='none'">
            <div class="image-info">
                <div class="image-key">{image_key}</div>
                <div class="label">Label: {label}</div>
                <div class="text">Text: {text[:100]}{'...' if len(text) > 100 else ''}</div>
                <div class="text">LLM rephrase: {llm_rephrase[:100]}{'...' if len(llm_rephrase) > 100 else ''}</div>
            </div>
            <div class="checkbox">
                <input type="checkbox" id="checkbox-{index}" data-image-key="{image_key}" data-index="{index}" onchange="toggleSelection(this)">
                <label for="checkbox-{index}">Select this image</label>
            </div>
        </div>
"""

        html_content += """
    </div>
    
    <script>
        let selectedImages = new Set();
        let selectedIndices = new Set();
        
        function updateStats() {
            const total = document.querySelectorAll('.image-card').length;
            const selected = selectedImages.size;
            const unselected = total - selected;
            
            document.getElementById('totalCount').textContent = total;
            document.getElementById('selectedCount').textContent = selected;
            document.getElementById('unselectedCount').textContent = unselected;
        }
        
        function toggleSelection(checkbox) {
            const imageKey = checkbox.getAttribute('data-image-key');
            const index = parseInt(checkbox.getAttribute('data-index'));
            
            if (checkbox.checked) {
                selectedImages.add(imageKey);
                selectedIndices.add(index);
                checkbox.closest('.image-card').classList.add('selected');
            } else {
                selectedImages.delete(imageKey);
                selectedIndices.delete(index);
                checkbox.closest('.image-card').classList.remove('selected');
            }
            updateStats();
        }
        
        function selectAll() {
            document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                if (!checkbox.checked) {
                    checkbox.checked = true;
                    const imageKey = checkbox.getAttribute('data-image-key');
                    const index = parseInt(checkbox.getAttribute('data-index'));
                    toggleSelection(checkbox);
                }
            });
        }
        
        function deselectAll() {
            document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                if (checkbox.checked) {
                    checkbox.checked = false;
                    const imageKey = checkbox.getAttribute('data-image-key');
                    const index = parseInt(checkbox.getAttribute('data-index'));
                    toggleSelection(checkbox);
                }
            });
        }
        
        function clearSelection() {
            selectedImages.clear();
            selectedIndices.clear();
            document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = false;
            });
            document.querySelectorAll('.image-card').forEach(card => {
                card.classList.remove('selected');
            });
            updateStats();
        }
        
        function exportSelection() {
            const data = {
                excluded_images: Array.from(selectedImages),
                excluded_indices: Array.from(selectedIndices),
                timestamp: new Date().toISOString()
            };
            
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'excluded_images.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            alert(`Exported an exclusion list of ${selectedImages.size} images`);
        }
        
        // Initialize statistics
        updateStats();
    </script>
</body>
</html>
"""

        # Write the HTML file
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML file generated: {output_html_path}")
        print(f"Contains {len(self.filtered_df)} images")

    def process_exclusion_list(self, exclusion_json_path, output_csv_path):
        """
        Apply the exclusion list and produce the final CSV file.

        Args:
            exclusion_json_path (str): exclusion list JSON file path
            output_csv_path (str): output CSV file path
        """
        print("Applying the exclusion list...")

        # Read the exclusion list
        with open(exclusion_json_path, 'r', encoding='utf-8') as f:
            exclusion_data = json.load(f)

        excluded_indices = set(exclusion_data.get('excluded_indices', []))

        # Build the final filtered dataframe
        final_df = self.filtered_df[~self.filtered_df.index.isin(excluded_indices)].copy()

        # Save the final CSV
        final_df.to_csv(output_csv_path, index=False)

        print(f"Final CSV generated: {output_csv_path}")
        print(f"Final rows: {len(final_df)}")
        print(f"Excluded rows: {len(excluded_indices)}")

        # Generate the statistics report
        self.generate_statistics_report(final_df, exclusion_data)

    def generate_statistics_report(self, final_df, exclusion_data):
        """Generate a statistics report."""
        report_path = "filtering_statistics_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Data filtering statistics report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Original rows: {len(self.df)}\n")
            f.write(f"Rows after removing invalid data: {len(self.filtered_df)}\n")
            f.write(f"Final rows kept: {len(final_df)}\n\n")
            f.write(f"Invalid rows removed: {len(self.df) - len(self.filtered_df)}\n")
            f.write(f"Manually excluded rows: {len(exclusion_data.get('excluded_indices', []))}\n\n")

            f.write("Excluded image list:\n")
            for img in exclusion_data.get('excluded_images', []):
                f.write(f"  - {img}\n")

            f.write(f"\nLabel statistics:\n")
            label_counts = final_df['label'].value_counts()
            for label, count in label_counts.head(10).items():
                f.write(f"  {label}: {count}\n")

        print(f"Statistics report generated: {report_path}")

def _resolve_paths():
    """Resolve input CSV / image dir from env (RUN_DIR), with a generic default."""
    run_dir = os.environ.get("RUN_DIR", "../data/outputs/run")
    csv_file_path = os.environ.get(
        "FILTER_INPUT_CSV", os.path.join(run_dir, "pdf_outputs.matched_with_llm.csv"))
    image_dir_path = os.environ.get(
        "FILTER_IMAGE_DIR", os.path.join(run_dir, "pdf_outputs.matched_image_paths_dir"))
    return csv_file_path, image_dir_path


def main():
    """Entry point: step 1 (generate the gallery)."""
    # Resolve paths (RUN_DIR points at the process_bbc_pdf.sh output directory)
    csv_file_path, image_dir_path = _resolve_paths()

    # Create the processor
    processor = DataFilterAndVisualizer(csv_file_path, image_dir_path)

    # Step 1: load and filter data
    processor.load_data()
    processor.filter_invalid_data()

    # Step 2: generate the image gallery HTML
    html_output_path = os.environ.get("FILTER_GALLERY_HTML", "image_gallery.html")
    processor.generate_image_gallery_html(html_output_path)

    print("\n" + "="*60)
    print("Step 1 complete.")
    print("1. Open image_gallery.html in a browser")
    print("2. Select the images to exclude")
    print("3. Click 'Export exclusion list'")
    print("4. Place the resulting excluded_images.json in the current directory")
    print("5. Then run:")
    print("   python filter_and_visualize_data.py --process-exclusion")
    print("="*60)

def process_exclusion():
    """Apply the exclusion list (step 2)."""
    csv_file_path, image_dir_path = _resolve_paths()
    exclusion_json_path = os.environ.get("EXCLUDED_JSON", "excluded_images.json")
    output_csv_path = os.environ.get("FILTER_OUTPUT_CSV", "filtered_final_data.csv")

    processor = DataFilterAndVisualizer(csv_file_path, image_dir_path)
    processor.load_data()
    processor.filter_invalid_data()
    processor.process_exclusion_list(exclusion_json_path, output_csv_path)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--process-exclusion":
        process_exclusion()
    else:
        main()
