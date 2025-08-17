import os
import re
import csv
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
from typing import Union, List, Optional, Tuple, Dict
import logging
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeadingInfo:
    """Class to store heading information"""
    def __init__(self, text: str, position: int, line_num: int, level: int, 
             is_bold: bool = False, font_size: float = 0.0, is_caps: bool = False,
             is_underline: bool = False, is_italic: bool = False):
        self.raw_text = text
        self.normalized_text = self._normalize_text_for_comparison(text)
        self.text = self.normalized_text

        self.position = position
        self.line_num = line_num
        self.level = level
        self.is_bold = is_bold
        self.font_size = font_size
        self.is_caps = is_caps
        self.is_underline = is_underline
        self.is_italic = is_italic
        self.formatting_signature = self._create_formatting_signature()
        self.numbering_pattern = self._extract_numbering_pattern()


    def _normalize_text_for_comparison(self, text: str) -> str:
        """
        Normalize text for heading comparison by removing leading star-number patterns.
        Args:
            text (str): The input text.
        Returns:
            str: Normalized text.
        """
        text = text.strip()
        text = re.sub(r'^\s*\*+\d+\s+', '', text)
        if re.match(r'^\s*\*\*[0-9]+', text):
            text = re.sub(r'^\s*\*\*[0-9]+\s*', '', text)
        elif re.match(r'^\s*\*[0-9]+', text):
            text = re.sub(r'^\s*\*[0-9]+\s*', '', text)
        
        return text.strip()
  
    def _create_formatting_signature(self) -> str:
        """
        Create a formatting signature string for heading comparison.
        Returns:
            str: Formatting signature.
        """
        signature_parts = []
        
        if self.is_bold:
            signature_parts.append("BOLD")
        if self.is_caps:
            signature_parts.append("CAPS")
        if self.is_underline:
            signature_parts.append("UNDERLINE")
        if self.is_italic:
            signature_parts.append("ITALIC")
        if self.font_size > 0:
            signature_parts.append(f"SIZE_{self.font_size:.1f}")
        
        if re.match(r'^\s*[IVX]+\.', self.text):
            signature_parts.append("ROMAN")
        elif re.match(r'^\s*[0-9]+\.', self.text):
            signature_parts.append("NUMBER")
        elif re.match(r'^\s*[A-Z]\.', self.text):
            signature_parts.append("LETTER")
        elif re.match(r'^\s*\*\*[0-9]+', self.text):
            signature_parts.append("DOUBLE_STAR_NUMBER")
        elif re.match(r'^\s*\*[0-9]+', self.text):
            signature_parts.append("STAR_NUMBER")
        elif re.match(r'^\s*\([0-9]+\)', self.text):
            signature_parts.append("PAREN_NUM")
        elif re.match(r'^\s*\([IVX]+\)', self.text):
            signature_parts.append("PAREN_ROMAN")
        elif re.match(r'^\s*\([A-Z]\)', self.text):
            signature_parts.append("PAREN_LETTER")
        elif re.match(r'^\s*\*[0-9]+', self.text):
            signature_parts.append("STAR_NUMBER")

        
        return "_".join(signature_parts) if signature_parts else "PLAIN"
    
    def _extract_numbering_pattern(self) -> Optional[Dict]:
        """
        Extract numbering pattern from heading text.
        Returns:
            Optional[Dict]: Numbering pattern info or None.
        """
        text = self.normalized_text.strip()
        
        # 检查单独的罗马数字（带或不带点号）
        match = re.match(r'^\s*([IVX]+)\.?\s*$', text)
        if match:
            roman_val = self._roman_to_int(match.group(1))
            return {"type": "roman_only", "value": roman_val, "text": match.group(1)}
        
        match = re.match(r'^\s*\*\*([0-9]+)', text)
        if match:
            return {"type": "double_star_number", "value": int(match.group(1))}
        
        match = re.match(r'^\s*\*([0-9]+)', text)
        if match:
            return {"type": "star_number", "value": int(match.group(1))}
        
        match = re.match(r'^\s*([0-9]+)\.', text)
        if match:
            return {"type": "number", "value": int(match.group(1))}
        
        match = re.match(r'^\s*([IVX]+)\.', text)
        if match:
            roman_val = self._roman_to_int(match.group(1))
            return {"type": "roman", "value": roman_val, "text": match.group(1)}
        
        match = re.match(r'^\s*([A-Z])\.', text)
        if match:
            return {"type": "letter", "value": ord(match.group(1)) - ord('A') + 1, "text": match.group(1)}
        
        return None

    def _roman_to_int(self, roman: str) -> int:
        """
        Convert a Roman numeral string to integer.
        Args:
            roman (str): Roman numeral string.
        Returns:
            int: Integer value.
        """
        roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        total = 0
        prev = 0
        for char in reversed(roman):
            val = roman_dict.get(char, 0)
            if val < prev:
                total -= val
            else:
                total += val
            prev = val
        return total

def extract_full_text_from_pdf(pdf_path: str) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Extract all text from a PDF file.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        Tuple[str, List[Tuple[int, str]]]: Full text and list of (page_num, text) tuples.
    """
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        page_info = []
        
        logger.info(f"  Extracting text from {len(reader.pages)} pages...")
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            page_info.append((page_num, text))
            full_text += f"--PAGE-{page_num}--\n{text}\n"
        
        logger.info(f"  Total text length: {len(full_text)} characters")
        return full_text, page_info
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return "", []

def extract_case_id_from_filename(filename: str) -> int:
    """
    Extract case ID from filename.
    Args:
        filename (str): Filename string.
    Returns:
        int: Case ID if found, else 0.
    """
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return 0

def detect_text_formatting(line: str) -> Tuple[bool, bool, float, bool, bool]:
    """
    Detect text formatting features in a line.
    Args:
        line (str): Input text line.
    Returns:
        Tuple[bool, bool, float, bool, bool]: (is_bold, is_caps, font_size, is_underline, is_italic)
    """
    line = line.strip()
    
    is_caps = line.isupper() and len(line.split()) > 1
    
    is_underline = '_' in line or '—' in line or '―' in line
    
    is_italic = False
    if len(line.split()) <= 8 and any(keyword in line.lower() for keyword in ['fact', 'background']):
        is_italic = True
    
    is_bold = False
    
    if is_caps and len(line.split()) <= 6:
        is_bold = True
    
    if re.match(r'^\s*([IVX]+\.|[0-9]+\.|[A-Z]\.|\\*\*?[0-9]+)', line):
        is_bold = True
    
    if len(line.split()) <= 8 and any(keyword in line.lower() for keyword in ['fact', 'background']):
        is_bold = True
    
    font_size = 12.0
    if is_caps or is_bold:
        font_size = 14.0
    
    return is_bold, is_caps, font_size, is_underline, is_italic

def analyze_heading_level(line: str) -> int:
    """
    Analyze heading level based on text pattern.
    Args:
        line (str): Input text line.
    Returns:
        int: Heading level (1, 2, or 3).
    """
    line = line.strip()
    
    if re.match(r'^\s*[IVX]+\s*$', line):
        return 1
    
    if re.match(r'^\s*[IVX]+\.\s+[A-Z]', line):
        return 1
    if re.match(r'^\s*[0-9]+\.\s+[A-Z]', line) and len(line.split()) <= 5:
        return 1
    if re.match(r'^\s*\*\*[0-9]+', line):
        return 1
    if re.match(r'^\s*\*[0-9]+', line):
        return 1
    if line.isupper() and len(line.split()) <= 4 and not line.endswith(':'):
        return 1
    
    if re.match(r'^\s*[A-Z]\.\s+[A-Z]', line):
        return 2
    if re.match(r'^\s*\([0-9]+\)\s+[A-Z]', line):
        return 2
    if re.match(r'^\s*\([IVX]+\)\s+[A-Z]', line):
        return 2
    
    if re.match(r'^\s*\([A-Z]\)\s+[A-Z]', line):
        return 3
    if line.endswith(':') and len(line.split()) <= 6:
        return 3
    
    return 2
    """
    Analyze heading level based on text pattern.
    Args:
        line (str): Input text line.
    Returns:
        int: Heading level (1, 2, or 3).
    """
    line = line.strip()
    
    if re.match(r'^\s*[IVX]+\.\s+[A-Z]', line):
        return 1
    if re.match(r'^\s*[0-9]+\.\s+[A-Z]', line) and len(line.split()) <= 5:
        return 1
    if re.match(r'^\s*\*\*[0-9]+', line):
        return 1
    if re.match(r'^\s*\*[0-9]+', line):
        return 1
    if line.isupper() and len(line.split()) <= 4 and not line.endswith(':'):
        return 1
    
    if re.match(r'^\s*[A-Z]\.\s+[A-Z]', line):
        return 2
    if re.match(r'^\s*\([0-9]+\)\s+[A-Z]', line):
        return 2
    if re.match(r'^\s*\([IVX]+\)\s+[A-Z]', line):
        return 2
    
    if re.match(r'^\s*\([A-Z]\)\s+[A-Z]', line):
        return 3
    if line.endswith(':') and len(line.split()) <= 6:
        return 3
    
    return 2

def is_heading(line: str) -> bool:
    """
    Determine if a line is a heading.
    Args:
        line (str): Input text line.
    Returns:
        bool: True if line is a heading, else False.
    """
    line = line.strip()
        
    if len(line) > 45 or len(line) < 1:  # 改为1以支持单个罗马数字
        return False
    
    # 检查是否为单独的罗马数字标题（带或不带点号）
    if re.match(r'^\s*[IVX]+\.?\s*$', line):
        return True
    
    numbered_patterns = [
        r'^\s*[IVX]+\.\s+[A-Z]',
        r'^\s*[0-9]+\.\s+[A-Z]',
        r'^\s*[A-Z]\.\s+[A-Z]',
        r'^\s*\*\*[0-9]+',
        r'^\s*\*[0-9]+',
        r'^\s*\([0-9]+\)\s+[A-Z]',
        r'^\s*\([IVX]+\)\s+[A-Z]',
        r'^\s*\([A-Z]\)\s+[A-Z]',
    ]
    
    for pattern in numbered_patterns:
        if re.match(pattern, line):
            return True
    
    words = line.split()
    if (line.isupper() and 
        len(words) <= 8 and 
        len(words) >= 1 and
        not any(word.isdigit() for word in words)):
        return True
    
    if line.endswith(':'):
        words = line[:-1].split()
        if (len(words) <= 6 and 
            (line[:-1].isupper() or 
             all(word[0].isupper() for word in words if word))):
            return True
    
    if any(keyword in line.lower() for keyword in ['fact', 'background']):
        words = line.split()
        if len(words) <= 8:
            return True
    
    if ('_' in line or '—' in line or '―' in line) and len(line.split()) <= 8:
        return True
    
    return False

def detect_table_of_contents(text: str, page_info: List[Tuple[int, str]]) -> Optional[int]:
    """
    Detect the table of contents page and return the cutoff line index.
    Args:
        text (str): Full text of the document.
        page_info (List[Tuple[int, str]]): List of (page_num, text) tuples.
    Returns:
        Optional[int]: Line index cutoff for table of contents, or None.
    """
    for page_num, page_text in page_info:
        if 2 <= page_num <= 4:
            dot_lines = re.findall(r'\.{5,}', page_text)
            if len(dot_lines) >= 3:
                logger.info(f"  Detected table of contents on page {page_num}")
                
                lines = text.split('\n')
                last_dot_line = -1
                
                for i, line in enumerate(lines):
                    if re.search(r'\.{5,}', line) and f'--PAGE-{page_num}--' in text[:text.find(line)] if line in text else False:
                        last_dot_line = i
                
                if last_dot_line > -1:
                    logger.info(f"  Table of contents cutoff at line {last_dot_line}")
                    return last_dot_line
    
    return None

def find_all_headings(text: str, page_info: List[Tuple[int, str]]) -> List[HeadingInfo]:
    """
    Find all headings in the document, excluding the table of contents.
    Args:
        text (str): Full text of the document.
        page_info (List[Tuple[int, str]]): List of (page_num, text) tuples.
    Returns:
        List[HeadingInfo]: List of HeadingInfo objects.
    """
    toc_cutoff = detect_table_of_contents(text, page_info)
    
    headings = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        if toc_cutoff is not None and i <= toc_cutoff:
            continue
            
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('--PAGE-'):
            continue
        
        if re.search(r'\.{5,}', line_stripped):
            continue
            
        if is_heading(line_stripped):
            is_bold, is_caps, font_size, is_underline, is_italic = detect_text_formatting(line_stripped)
            
            level = analyze_heading_level(line_stripped)
            
            position = len('\n'.join(lines[:i]))
            
            heading_info = HeadingInfo(
                text=line_stripped,
                position=position,
                line_num=i,
                level=level,
                is_bold=is_bold,
                font_size=font_size,
                is_caps=is_caps,
                is_underline=is_underline,
                is_italic=is_italic
            )
            
            headings.append(heading_info)
    
    return headings


def find_target_heading(headings: List[HeadingInfo]) -> Optional[HeadingInfo]:
    """
    Find the target heading containing 'fact' or 'background', or Roman numeral I.
    Args:
        headings (List[HeadingInfo]): List of HeadingInfo objects.
    Returns:
        Optional[HeadingInfo]: The matched HeadingInfo object or None.
    """
    target_keywords = ['fact', 'background']
    
    # 首先查找包含关键词的标题
    for heading in headings:
        search_text = heading.normalized_text.lower()
        original_text = heading.text.lower()
        
        logger.info(f"  Checking heading: '{heading.text}'")
        logger.info(f"  Normalized text: '{heading.normalized_text}'")
        logger.info(f"  Search text: '{search_text}'")
        
        for keyword in target_keywords:
            if keyword in search_text or keyword in original_text:
                logger.info(f"  Found matching heading: {heading.text}")
                logger.info(f"  Normalized text: {heading.normalized_text}")
                logger.info(f"  Matched keyword: {keyword}")
                logger.info(f"  Heading level: {heading.level}, Formatting: {heading.formatting_signature}")
                return heading
    
    # 如果没找到关键词标题，查找罗马数字I（带或不带点号）
    for heading in headings:
        heading_text = heading.text.strip()
        # 检查是否为单独的 "I" 或 "I."
        if re.match(r'^\s*I\.?\s*$', heading_text):
            logger.info(f"  Found Roman numeral I heading: '{heading.text}'")
            logger.info(f"  Raw text: '{heading.raw_text}'")
            return heading
    
    logger.info("  No matching heading found. All headings:")
    for i, heading in enumerate(headings):
        logger.info(f"    {i+1}. '{heading.text}' -> normalized: '{heading.normalized_text}' -> raw: '{heading.raw_text}'")
    
    return None

def find_next_same_level_heading(target_heading: HeadingInfo, headings: List[HeadingInfo]) -> Optional[HeadingInfo]:
    """
    Find the next heading at the same level as the target heading.
    Args:
        target_heading (HeadingInfo): The target HeadingInfo object.
        headings (List[HeadingInfo]): List of HeadingInfo objects.
    Returns:
        Optional[HeadingInfo]: The next same-level HeadingInfo object or None.
    """
    target_index = -1
    
    for i, heading in enumerate(headings):
        if heading.position == target_heading.position:
            target_index = i
            break
    
    if target_index == -1:
        return None
    
    logger.info(f"  Target heading pattern: {target_heading.numbering_pattern}")
    logger.info(f"  Target heading format: {target_heading.formatting_signature}")
    
    for i in range(target_index + 1, len(headings)):
        next_heading = headings[i]
        logger.info(f"  Checking heading: {next_heading.text}")
        logger.info(f"  Pattern: {next_heading.numbering_pattern}")
        logger.info(f"  Format: {next_heading.formatting_signature}")
        
        if target_heading.numbering_pattern:
            if next_heading.numbering_pattern and next_heading.numbering_pattern["type"] == target_heading.numbering_pattern["type"]:
                target_num = target_heading.numbering_pattern["value"]
                next_num = next_heading.numbering_pattern["value"]
                
                if next_num > target_num:
                    logger.info(f"  -> Found next numbered heading (same type, sequential)")
                    return next_heading
                else:
                    logger.info(f"  -> Not same level (number not sequential: {next_num} <= {target_num})")
            else:
                logger.info(f"  -> Not same level (different numbering pattern)")
        else:
            if next_heading.formatting_signature == target_heading.formatting_signature:
                if not next_heading.numbering_pattern:
                    logger.info(f"  -> Found next heading (same format, no numbering)")
                    return next_heading
                else:
                    logger.info(f"  -> Not same level (next has numbering but target doesn't)")
            else:
                logger.info(f"  -> Not same level (different format)")
    
    logger.info(f"  No same-level heading found after target")
    return None

def extract_content_between_headings(text: str, start_heading: HeadingInfo, 
                                   end_heading: Optional[HeadingInfo] = None) -> str:
    """
    Extract content between two headings.
    Args:
        text (str): Full text of the document.
        start_heading (HeadingInfo): Start HeadingInfo object.
        end_heading (Optional[HeadingInfo]): End HeadingInfo object or None.
    Returns:
        str: Extracted content string.
    """
    lines = text.split('\n')
    
    start_line = start_heading.line_num
    end_line = end_heading.line_num if end_heading else len(lines)
    
    content_lines = []
    for i in range(start_line + 1, end_line):
        line = lines[i].strip()
        
        if line and not line.startswith('--PAGE-') and not re.search(r'\.{5,}', line):
            content_lines.append(line)
    
    content = ' '.join(content_lines)
    
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    
    logger.info(f"  Extracted content length: {len(content)} characters")
    
    return content

def extract_synopsis_section(text: str) -> str:
   """
   Extract the Synopsis section from the document.
   Args:
       text (str): Full text of the document.
   Returns:
       str: Extracted synopsis or fallback content.
   """
   text = re.sub(r'[^\x00-\x7F]+', ' ', text)
   text = re.sub(r'-{3,}', '', text)
   
   synopsis_patterns = [
       r'(?i)Synopsis[:\s]+(.*?)(?=\n\s*(?:[IVX]+\.|[0-9]+\.|[A-Z]+\.|[A-Z][A-Z\s]*:|Holdings?:|Held:|\Z))',
       r'(?i)SYNOPSIS[:\s]+(.*?)(?=\n\s*(?:[IVX]+\.|[0-9]+\.|[A-Z]+\.|[A-Z][A-Z\s]*:|Holdings?:|Held:|\Z))',
   ]
   
   for pattern in synopsis_patterns:
       match = re.search(pattern, text, re.DOTALL)
       if match:
           synopsis = match.group(1).strip()
           synopsis = re.sub(r'\s+', ' ', synopsis)
           
           if synopsis.startswith("Background: "):
               synopsis = synopsis[12:].strip()
           elif synopsis.startswith("BACKGROUND: "):
               synopsis = synopsis[12:].strip()
               
           if len(synopsis) > 50:
               logger.info(f"  Found Synopsis section: {len(synopsis)} characters")
               return synopsis
   
   lines = text.split('\n')
   first_paragraph = []
   
   for line in lines:
       line = line.strip()
       if not line or line.startswith('--PAGE-') or re.search(r'\.{5,}', line):
           continue
       
       if (re.match(r'^\s*[IVX]+\.\s', line) or 
           re.match(r'^\s*[0-9]+\.\s', line) or
           (line.isupper() and len(line.split()) <= 5)):
           continue
       
       first_paragraph.append(line)
       
       if len(' '.join(first_paragraph)) > 300:
           break
   
   if first_paragraph:
       result = ' '.join(first_paragraph)
       
       if result.startswith("Background: "):
           result = result[12:].strip()
       elif result.startswith("BACKGROUND: "):
           result = result[12:].strip()
       
       logger.info(f"  Extracted opening content as synopsis: {len(result)} characters")
       return result
   
   return "Synopsis not found"

def extract_background_section(text: str, page_info: List[Tuple[int, str]]) -> Tuple[str, bool]:
    """
    Extract the Background section from the document, including Roman numeral sections I-III (or up to the last available).
    Args:
        text (str): Full text of the document.
        page_info (List[Tuple[int, str]]): List of (page_num, text) tuples.
    Returns:
        Tuple[str, bool]: (Extracted background content, True if structure not found)
    """
    headings = find_all_headings(text, page_info)
    
    if not headings:
        logger.warning("  No headings found in document")
        return "Background not found - no document structure", True
    
    logger.info(f"  Found {len(headings)} headings total")
    
    target_heading = find_target_heading(headings)
    
    if not target_heading:
        logger.warning("  No background, fact headings, or Roman numeral I found")
        return "Background not found - no matching headings", True
    
    # 检查是否为罗马数字I，如果是，提取I-III的内容（或到最后一个可用的罗马数字）
    if re.match(r'^\s*I\.?\s*$', target_heading.text.strip()):
        logger.info("  Processing Roman numeral sections I-III (or up to last available)")
        
        # 找到所有单独的罗马数字标题
        roman_headings = []
        for heading in headings:
            if re.match(r'^\s*[IVX]+\.?\s*$', heading.text.strip()):
                # 提取罗马数字部分（去掉可能的点号）
                roman_text = re.sub(r'\.', '', heading.text.strip())
                roman_value = target_heading._roman_to_int(roman_text)
                if 1 <= roman_value <= 10:  # 扩大范围以包含更多罗马数字
                    roman_headings.append((roman_value, heading))
        
        if not roman_headings:
            logger.warning("  No Roman numeral headings found")
            return "Roman numeral sections not found", True
        
        # 按数值排序
        roman_headings.sort(key=lambda x: x[0])
        logger.info(f"  Found Roman numeral headings: {[f'{num}:{heading.text}' for num, heading in roman_headings]}")
        
        # 确定要提取的范围：优先提取I-III，如果没有III就提取到最后一个可用的
        target_romans = []
        for roman_num, heading in roman_headings:
            if roman_num <= 3:  # I, II, III
                target_romans.append((roman_num, heading))
        
        # 如果没有找到III，但有其他罗马数字，就提取到最后一个可用的
        if not any(num == 3 for num, _ in target_romans) and roman_headings:
            # 找到最大的罗马数字（但不超过合理范围）
            max_roman = max(roman_headings, key=lambda x: x[0])
            if max_roman[0] > 3:
                # 重新定义目标范围：从I到找到的最大罗马数字
                target_romans = []
                for roman_num, heading in roman_headings:
                    if roman_num <= max_roman[0]:
                        target_romans.append((roman_num, heading))
        
        logger.info(f"  Target Roman sections to extract: {[f'{num}:{heading.text}' for num, heading in target_romans]}")
        
        if not target_romans:
            return "No valid Roman numeral sections found", True
        
        # 提取所有目标罗马数字section的内容
        content_parts = []
        for i, (roman_num, heading) in enumerate(target_romans):
            # 找下一个标题作为结束点
            next_heading = None
            
            # 首先尝试找下一个罗马数字标题
            if i + 1 < len(target_romans):
                next_heading = target_romans[i + 1][1]
            else:
                # 如果是最后一个目标罗马数字，找下一个非罗马数字的同级或更高级标题
                current_pos = heading.position
                for h in headings:
                    if h.position > current_pos:
                        # 检查是否为罗马数字标题
                        if not re.match(r'^\s*[IVX]+\.?\s*$', h.text.strip()):
                            # 检查是否为同级或更高级标题
                            if h.level <= heading.level:
                                next_heading = h
                                break
            
            section_content = extract_content_between_headings(text, heading, next_heading)
            if section_content:
                roman_text = re.sub(r'\.', '', heading.text.strip())  # 移除点号显示
                content_parts.append(f"Section {roman_text}: {section_content}")
                logger.info(f"  Extracted Section {roman_text}: {len(section_content)} characters")
            else:
                logger.warning(f"  Section {roman_num} is empty")
        
        if content_parts:
            combined_content = " ".join(content_parts)
            logger.info(f"  Total combined content: {len(combined_content)} characters")
            return combined_content, False
        else:
            return "Roman numeral sections found but empty", True
    
    else:
        # 原有逻辑：处理普通的背景或事实标题
        next_heading = find_next_same_level_heading(target_heading, headings)
        content = extract_content_between_headings(text, target_heading, next_heading)
        
        if content and len(content) > 20:
            return content, False
        else:
            logger.warning("  Background section found but content is too short or empty")
            return "Background section found but empty", True

def clean_text_for_csv(text: str) -> str:
    """
    Clean text for CSV output.
    Args:
        text (str): Input text.
    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""
    
    text = re.sub(r'--PAGE-\d+--', '', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.strip()
    
    return text

def split_into_chunks(text: str, chunk_size: int = 32000) -> List[str]:
    """
    Split text into chunks of a maximum character length.
    Args:
        text (str): Input text.
        chunk_size (int): Maximum chunk size.
    Returns:
        List[str]: List of text chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_pdf_folder(folder_path: str, output_csv: str):
   """
   Process all PDF files in a folder and extract background and synopsis sections.
   Args:
       folder_path (str): Path to the folder containing PDF files.
       output_csv (str): Output CSV file path.
   Returns:
       List[dict]: List of result dictionaries for each PDF file.
   """
   results = []
   processing_times = []
   pdf_files = list(Path(folder_path).glob("*.pdf"))
   pdf_files.sort()

   logger.info(f"Found {len(pdf_files)} PDF files to process...")

   for i, pdf_file in enumerate(pdf_files, 1):
       logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
       start_time = time.time()

       full_text, page_info = extract_full_text_from_pdf(pdf_file)

       if not full_text:
           logger.warning(f"No text extracted from {pdf_file.name}")
           results.append({
               'CaseID': extract_case_id_from_filename(pdf_file.name),
               'filename': pdf_file.name,
               'Synopsis': 'Text extraction failed',
               'Background_1': 'Text extraction failed',
               'StructureCheckFlag': 1
           })
           end_time = time.time()
           processing_time = end_time - start_time
           processing_times.append(processing_time)
           logger.info(f"  Processing time: {processing_time:.2f} seconds")
           continue

       case_id = extract_case_id_from_filename(pdf_file.name)
       synopsis = extract_synopsis_section(full_text)
       background, no_structure = extract_background_section(full_text, page_info)

       synopsis = clean_text_for_csv(synopsis)
       background = clean_text_for_csv(background)

       structure_flag = 1 if no_structure else 0

       result_row = {
           'CaseID': case_id,
           'filename': pdf_file.name,
           'Synopsis': synopsis,
           'StructureCheckFlag': structure_flag
       }

       background_chunks = split_into_chunks(background)

       for idx, chunk in enumerate(background_chunks):
           result_row[f'Background_{idx + 1}'] = chunk

       logger.info(f"  Results - CaseID: {case_id}, Structure Flag: {structure_flag}")
       logger.info(f"  Background parts: {len(background_chunks)}")

       results.append(result_row)
       
       end_time = time.time()
       processing_time = end_time - start_time
       processing_times.append(processing_time)
       logger.info(f"  Processing time: {processing_time:.2f} seconds")

   df = pd.DataFrame(results)
   initial_count = len(df)
   df = df.drop_duplicates(subset=['CaseID'], keep='last')
   final_count = len(df)
   
   if initial_count > final_count:
       logger.info(f"Removed {initial_count - final_count} duplicate CaseID entries")
   
   df.to_csv(output_csv, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL, escapechar='\\')
   logger.info(f"Results saved to {output_csv}")
   
   if processing_times:
       avg_time = sum(processing_times) / len(processing_times)
       total_time = sum(processing_times)
       logger.info(f"Time statistics:")
       logger.info(f"  Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
       logger.info(f"  Average time per file: {avg_time:.2f} seconds")
       logger.info(f"  Fastest file: {min(processing_times):.2f} seconds")
       logger.info(f"  Slowest file: {max(processing_times):.2f} seconds")

   return results

def main():
    folder_path = "CaseAnalysis_Task"
    
    if not Path(folder_path).exists():
        logger.error(f"Folder '{folder_path}' does not exist.")
        logger.error("Please modify the folder_path variable to point to your PDF folder.")
        return
    
    output_csv = "cases_New_Code_222.csv"
    results = process_pdf_folder(folder_path, output_csv)
    
    logger.info(f"Processing complete! Extracted information from {len(results)} files.")
    logger.info(f"Results saved to: {output_csv}")
    
    if results:
        print("\nSample results:")
        for result in results[:3]:
            print(f"\nCase {result['CaseID']}:")
            print(f"  Filename: {result['filename']}")
            synopsis_text = result.get('Synopsis', '')
            background_text = result.get('Background', '')
            print(f"  Synopsis: {synopsis_text[:100]}...")
            print(f"  Background: {background_text[:100]}...")
            print(f"  Structure Flag: {result['StructureCheckFlag']}")

if __name__ == "__main__":
    main()