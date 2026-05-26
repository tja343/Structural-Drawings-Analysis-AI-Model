import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.parsing.regex_parser import EngineeringParser

def test_spacing_format():
    parser = EngineeringParser()
    res = parser.parse("H10@300")
    assert res["bar_type"] == "H"
    assert res["diameter"] == 10
    assert res["spacing"] == 300
    assert res["parsed"] == True

def test_quantity_format():
    parser = EngineeringParser()
    res = parser.parse("2Y20")
    assert res["quantity"] == 2
    assert res["bar_type"] == "Y"
    assert res["diameter"] == 20
    assert res["parsed"] == True

def test_layer_format():
    parser = EngineeringParser()
    res = parser.parse("Y16 TOP")
    assert res["bar_type"] == "Y"
    assert res["diameter"] == 16
    assert res["layer"] == "TOP"
    assert res["parsed"] == True

def test_ocr_correction():
    parser = EngineeringParser()
    # Testing OCR reading 'a' instead of '@'
    res = parser.parse("T12a150")
    assert res["corrected"] == "T12@150"
    assert res["bar_type"] == "T"
    assert res["diameter"] == 12
    assert res["spacing"] == 150
    assert res["parsed"] == True
    
    # Testing OCR reading 'l' instead of '1'
    res2 = parser.parse("Hl0@200")
    assert res2["corrected"] == "H10@200"
    assert res2["diameter"] == 10
    assert res2["spacing"] == 200

if __name__ == "__main__":
    test_spacing_format()
    test_quantity_format()
    test_layer_format()
    test_ocr_correction()
    print("All parser tests passed!")
