import re
from typing import Optional

# Define license plate formats using regex-like placeholders
PLATE_FORMATS = {
    "US_STANDARD": "[A-Z]{3}\d{3}",  # Example: ABC123
    "EU_STANDARD": "[A-Z]{2}\d{4}[A-Z]{2}",  # Example: AB1234CD
    "ITALY_STANDARD": "[A-Z]{2}\d{2}[A-Z]{3}",  # Example: AB12CDE
    "INDIA_STANDARD": "[A-Z]{2}\d{2}[A-Z]{2}\d{4}",
    "ISRAEL_STANDARD1": "\d{8}",
    "ISRAEL_STANDARD2": "\d{7}", 
}

# Mapping of visually similar characters
SIMILAR_CHARACTERS = {
    "O": "0", "I": "1", "Z": "2", "S": "5", "B": "8",
    "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"
}

def regex_to_format(pattern: str) -> str:
    """Converts a regex pattern to a format string with 'L' for letters and 'D' for digits."""
    
    # Expand occurrences of {n} (e.g., [A-Z]{2} -> 'LL')
    pattern = re.sub(r"\[A-Z\]{(\d+)}", lambda m: "L" * int(m.group(1)), pattern)
    pattern = re.sub(r"\[0-9\]{(\d+)}", lambda m: "D" * int(m.group(1)), pattern)
    pattern = re.sub(r"\\d{(\d+)}", lambda m: "D" * int(m.group(1)), pattern)
    
    # Replace single letter/digit definitions
    pattern = pattern.replace("[A-Z]", "L").replace("[0-9]", "D").replace("\\d", "D")

    return re.sub(r"[^LD]", "", pattern)  # Remove extra symbols like dashes if needed

def validate_plate(plate_text: str) -> Optional[str]:
    """Validates if a plate matches a known format."""
    for pattern in PLATE_FORMATS.values():
        if re.fullmatch(pattern, plate_text):
            return plate_text
    return None

def correct_plate(plate_text: str) -> Optional[str]:
    """Attempts to correct and validate an invalid plate."""
    
    for pattern in PLATE_FORMATS.values():
        expected_format = regex_to_format(pattern)

        if len(plate_text) != len(expected_format):
            continue  # Skip if length does not match
        
        corrected_plate = list(plate_text)
        
        for i, char in enumerate(corrected_plate):
            expected_type = expected_format[i]
            
            # If a letter is in a digit position, try replacing it
            if expected_type == "D" and char in SIMILAR_CHARACTERS and char.isalpha():
                corrected_plate[i] = SIMILAR_CHARACTERS[char]
            
            # If a digit is in a letter position, try replacing it
            elif expected_type == "L" and char in SIMILAR_CHARACTERS and char.isdigit():
                corrected_plate[i] = SIMILAR_CHARACTERS[char]

        corrected_version = "".join(corrected_plate)
        if re.fullmatch(pattern, corrected_version):
            return corrected_version  # Return first valid correction
    
    return None  # No valid correction found

def process_plate(plate_text: str) -> Optional[str]:
    """Cleans, validates, or attempts to correct a license plate."""
    
    # Clean input: remove all non-alphanumeric characters
    cleaned_text = re.sub(r"[^A-Za-z0-9]", "", plate_text)

    valid_plate = validate_plate(cleaned_text)
    if valid_plate:
        return valid_plate
    
    return correct_plate(plate_text)
