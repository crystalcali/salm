from salm.trace_jepa.evaluate import extract_answer, normalized_equal


def test_extract_answer_formats():
    assert extract_answer("Reason.\nFinal answer: 42.") == "42"
    assert extract_answer(r"Something \boxed{2} and \boxed{5}") == "5"
    assert extract_answer("The number 3 is somewhere") == ""


def test_numeric_and_text_exact_match():
    assert normalized_equal("0042", "42")
    assert normalized_equal("3.0", "3")
    assert normalized_equal("x + 1", "x+1")
    assert not normalized_equal("", "0")
    assert not normalized_equal("4", "5")
