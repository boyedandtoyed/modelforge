from modelforge.data import format_alpaca, _synthetic_samples

def test_format_alpaca():
    row = {"instruction": "Test", "input": "", "output": "Answer"}
    sample = format_alpaca(row)
    assert "Test" in sample.prompt
    assert sample.completion == "Answer"

def test_synthetic_samples():
    samples = _synthetic_samples(10)
    assert len(samples) == 10
    assert all(s.prompt for s in samples)
