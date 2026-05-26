import re
from typing import Dict, Any, Optional

class EngineeringParser:
    def __init__(self):
        # Base patterns to handle various formats
        # Format 1: H10@300, T12@150, R8-200
        # Format 2: 2Y20
        # Format 3: Y16 TOP
        
        self.spacing_pattern = re.compile(r"([A-Z])(\d+)[@\-](\d+)", re.IGNORECASE)
        self.quantity_pattern = re.compile(r"(\d+)([A-Z])(\d+)", re.IGNORECASE)
        self.layer_pattern = re.compile(r"([A-Z])(\d+)\s+(TOP|BOTTOM|T|B)", re.IGNORECASE)

    def correct_ocr_mistakes(self, raw_text: str) -> str:
        """Fix common OCR errors in engineering drawings."""
        text = raw_text.upper().strip()
        
        # Replace common 'a' mistaken for '@' if between digits
        # e.g., H10a300 -> H10@300
        text = re.sub(r'([A-Z]\d+)[A|a](\d+)', r'\1@\2', text)
        
        # Replace 'l' or 'I' with '1' if inside numbers
        # e.g., Hl0 -> H10
        text = re.sub(r'([A-Z])[L|I|l](\d)', r'\g<1>1\2', text)
        
        # Fix spaces around @
        text = text.replace(" @", "@").replace("@ ", "@")
        
        return text

    def parse(self, raw_text: str) -> Dict[str, Any]:
        """
        Takes raw OCR text and attempts to extract engineering semantics.
        Returns a dictionary with parsed fields.
        """
        corrected_text = self.correct_ocr_mistakes(raw_text)
        
        result = {
            "raw": raw_text,
            "corrected": corrected_text,
            "parsed": False,
            "bar_type": None,
            "diameter": None,
            "spacing": None,
            "quantity": None,
            "layer": None
        }

        # Match Format 1: H10@300, R8-200
        match_spacing = self.spacing_pattern.match(corrected_text)
        if match_spacing:
            result["bar_type"] = match_spacing.group(1).upper()
            result["diameter"] = int(match_spacing.group(2))
            result["spacing"] = int(match_spacing.group(3))
            result["parsed"] = True
            return result

        # Match Format 2: 2Y20
        match_qty = self.quantity_pattern.match(corrected_text)
        if match_qty:
            result["quantity"] = int(match_qty.group(1))
            result["bar_type"] = match_qty.group(2).upper()
            result["diameter"] = int(match_qty.group(3))
            result["parsed"] = True
            return result

        # Match Format 3: Y16 TOP
        match_layer = self.layer_pattern.match(corrected_text)
        if match_layer:
            result["bar_type"] = match_layer.group(1).upper()
            result["diameter"] = int(match_layer.group(2))
            
            layer_raw = match_layer.group(3).upper()
            if layer_raw in ["T", "TOP"]:
                result["layer"] = "TOP"
            elif layer_raw in ["B", "BOTTOM"]:
                result["layer"] = "BOTTOM"
                
            result["parsed"] = True
            return result

        return result
