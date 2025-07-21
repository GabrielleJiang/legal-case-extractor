# Court Case PDF Background Extractor

This project provides a script (`retrieve_case_backgrounds.py`) for batch extracting the background and synopsis sections from a folder of court case PDF files. The extracted information is saved into a structured CSV file for further analysis or downstream processing.

## Features
- Batch process all PDF files in a specified folder
- Automatically detect and extract the 'Background' and 'Synopsis' sections from each PDF
- Handles various heading formats and document structures
- Splits long background content into multiple columns for CSV compatibility
- Outputs a CSV file with case ID, filename, synopsis, background (chunked), and structure check flag

## Usage
1. Place your court case PDF files in a folder (default: `CaseAnalysis_Task/`).
2. Run the script:
   ```bash
   python retrieve_case_backgrounds.py
   ```
3. The results will be saved to `case_extraction_results_final_version.csv` by default.

## Input
- Folder containing PDF files (default: `CaseAnalysis_Task/`)

## Output
- CSV file (default: `case_extraction_results_final_version.csv`) with the following columns:
  - `CaseID`: Extracted from filename
  - `filename`: PDF filename
  - `Synopsis`: Extracted synopsis section
  - `Background_1`, `Background_2`, ...: Chunks of the background section
  - `StructureCheckFlag`: 1 if structure/heading not found, 0 otherwise

## Requirements
- Python 3.7+
- See `requirements.txt` for dependencies

## How it works
- The script reads each PDF, extracts all text, detects headings, and finds the relevant sections.
- If the background section is not found, a flag is set in the output.
- All results are saved in a CSV for easy review and further processing.

## License
MIT License 