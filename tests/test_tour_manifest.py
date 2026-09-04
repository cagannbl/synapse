"""
Tests for Tour of Synapse Interactive Manifest and Code Snippets
"""

import os
import json
import pytest
from synapse.parser import parse_source


def test_tour_json_structure_and_schema():
    """Validates playground/tour.json format, step count, and fields."""
    tour_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "playground", "tour.json")
    )
    assert os.path.isfile(tour_file), f"Missing {tour_file}"

    with open(tour_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "title" in data
    assert "steps" in data
    steps = data["steps"]
    assert len(steps) == 6, f"Expected 6 tour steps, got {len(steps)}"

    for i, step in enumerate(steps, start=1):
        assert step["id"] == i
        assert "title" in step and len(step["title"]) > 0
        assert "description" in step and len(step["description"]) > 0
        assert "code" in step and len(step["code"]) > 0


def test_tour_code_snippets_parse_cleanly():
    """All 6 Tour of Synapse code snippets must parse cleanly without syntax errors."""
    tour_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "playground", "tour.json")
    )
    with open(tour_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for step in data["steps"]:
        code = step["code"]
        # Must parse without throwing LexerError or ParseError
        ast = parse_source(code, filename=f"tour_step_{step['id']}.syn", use_cache=False)
        assert ast is not None
        assert hasattr(ast, "statements") or hasattr(ast, "body")
