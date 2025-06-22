import re
from typing import Optional, Dict

class PlateFormatService:
    def __init__(self):
        self.plate_formats = {
            "US_STANDARD": r"[A-Z]{3}\d{3}",  # Example: ABC123
            "EU_STANDARD": r"[A-Z]{2}\d{4}[A-Z]{2}",  # Example: AB1234CD
            "ITALY_STANDARD": r"[A-Z]{2}\d{2}[A-Z]{3}",  # Example: AB12CDE
            "INDIA_STANDARD": r"[A-Z]{2}\d{2}[A-Z]{2}\d{4}",
            "ISRAEL_STANDARD1": r"\d{8}",
            "ISRAEL_STANDARD2": r"\d{7}",
        }

        self.similar_characters = {
            "O": "0", "I": "1", "Z": "2", "S": "5", "B": "8",
            "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"
        }

    def get_formats(self) -> Dict[str, str]:
        """Return all current plate formats."""
        return self.plate_formats

    def add_format(self, name: str, pattern: str) -> bool:
        """Add a new plate format if valid and not duplicate."""
        if name in self.plate_formats:
            return False  # duplicate name

        try:
            re.compile(pattern)  # Validate regex syntax
        except re.error:
            return False

        self.plate_formats[name] = pattern
        return True

    def delete_format(self, name: str) -> bool:
        """Delete a plate format by name."""
        if name not in self.plate_formats:
            return False
        del self.plate_formats[name]
        return True

    def regex_to_format(self, pattern: str) -> str:
        """Convert a regex pattern to a simple format string with 'L' and 'D'."""
        # Expand occurrences of {n} (e.g., [A-Z]{2} -> 'LL')
        pattern = re.sub(r"\[A-Z\]{(\d+)}", lambda m: "L" * int(m.group(1)), pattern)
        pattern = re.sub(r"\[0-9\]{(\d+)}", lambda m: "D" * int(m.group(1)), pattern)
        pattern = re.sub(r"\\d{(\d+)}", lambda m: "D" * int(m.group(1)), pattern)

        # Replace single letter/digit definitions
        pattern = pattern.replace("[A-Z]", "L").replace("[0-9]", "D").replace("\\d", "D")

        # Remove any other characters except L and D
        return re.sub(r"[^LD]", "", pattern)

    def validate_plate(self, plate_text: str) -> Optional[str]:
        """Check if a plate matches any known format exactly."""
        for pattern in self.plate_formats.values():
            if re.fullmatch(pattern, plate_text):
                return plate_text
        return None

    def correct_plate(self, plate_text: str) -> Optional[str]:
        """
        Try to fix a plate by replacing visually similar characters
        and validate again against known formats.
        """
        for pattern in self.plate_formats.values():
            expected_format = self.regex_to_format(pattern)
            if len(plate_text) != len(expected_format):
                continue

            corrected_plate = list(plate_text)
            for i, char in enumerate(corrected_plate):
                expected_type = expected_format[i]

                if expected_type == "D" and char.isalpha() and char in self.similar_characters:
                    corrected_plate[i] = self.similar_characters[char]
                elif expected_type == "L" and char.isdigit() and char in self.similar_characters:
                    corrected_plate[i] = self.similar_characters[char]

            corrected_version = "".join(corrected_plate)
            if re.fullmatch(pattern, corrected_version):
                return corrected_version

        return None

    def process_plate(self, plate_text: str) -> Optional[str]:
        """
        Clean a plate (remove non-alphanumeric chars),
        then validate or try to correct it.
        """
        cleaned_text = re.sub(r"[^A-Za-z0-9]", "", plate_text)

        valid_plate = self.validate_plate(cleaned_text)
        if valid_plate:
            return valid_plate

        return self.correct_plate(cleaned_text)
