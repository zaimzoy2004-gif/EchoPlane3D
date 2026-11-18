"""The manuscripts in paper/ must be regenerated from results/ — never edited by hand."""
import json
import re
import subprocess
import sys

import pytest

from backend.core import config

PAPER = config.PROJECT_ROOT / "paper"
SCRIPT = config.PROJECT_ROOT / "scripts" / "build_paper_numbers.py"
HAS_RESULTS = (config.RESULTS_DIR / "table2_camus_validation.csv").is_file() and (PAPER / "numbers.json").is_file()
pytestmark = pytest.mark.skipif(not HAS_RESULTS, reason="paper benchmark has not been run")


def test_rendered_papers_match_results():
    res = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, cwd=config.PROJECT_ROOT)
    assert res.returncode == 0, res.stdout + res.stderr


def test_templates_have_no_hand_typed_result_numbers():
    """Percentages and Dice-style decimals in the results sections must come from placeholders."""
    for tpl in (PAPER / "templates").glob("*.md"):
        text = tpl.read_text()
        for section in re.findall(r"<!-- numbers-from-results -->(.*?)<!-- /numbers-from-results -->", text, re.S):
            literal = re.findall(r"(?<![\w{.])\d+\.\d+(?![\w}])", re.sub(r"\{\{\w+\}\}", "", section))
            assert literal == [], f"{tpl.name}: hand-typed numbers {literal[:5]}"


def test_every_placeholder_resolves():
    nums = json.loads((PAPER / "numbers.json").read_text())
    for tpl in (PAPER / "templates").glob("*.md"):
        missing = set(re.findall(r"\{\{(\w+)\}\}", tpl.read_text())) - set(nums)
        assert not missing, f"{tpl.name}: {sorted(missing)}"
