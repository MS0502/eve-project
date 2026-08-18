import json

import pytest

from natural_lang import NaturalLanguage
from eve_main_abc import EVE_TierAB


class FakeSA:
    def __init__(self):
        self.neighbors = {}
        self.activations = {}
        self.weights = {}

    def discover_category(
        self,
        category,
        context_categories=None,
        initial_strength=0.0,
        link_strength=0.0,
    ):
        self.neighbors.setdefault(category, set())
        self.activations.setdefault(category, initial_strength)

    def learn_pair(self, a, b, strength=0.3):
        self.neighbors.setdefault(a, set()).add(b)
        self.neighbors.setdefault(b, set()).add(a)
        self.weights[tuple(sorted((a, b)))] = strength


class Dummy:
    pass


def make_nl():
    return NaturalLanguage(FakeSA(), Dummy(), Dummy())


def test_no_source_means_empty_belief_state():
    nl = make_nl()

    result = nl.load_beliefs()

    assert result["beliefs_loaded"] == 0
    assert nl.beliefs == {}
    assert nl.belief_index == {}


def test_missing_file_means_empty_belief_state(tmp_path):
    nl = make_nl()

    result = nl.load_beliefs(
        path=str(tmp_path / "does-not-exist.json")
    )

    assert result["beliefs_loaded"] == 0
    assert nl.beliefs == {}


def test_corrupt_json_fails_closed(tmp_path):
    path = tmp_path / "beliefs.json"
    path.write_text("{ broken json", encoding="utf-8")

    nl = make_nl()

    with pytest.raises(json.JSONDecodeError):
        nl.load_beliefs(path=str(path))


def test_explicit_beliefs_still_load_normally():
    beliefs = {
        "B1": {
            "belief_id": "B1",
            "statement": "EVE 존재한다",
            "triple": {
                "subject": "EVE",
                "predicate_text": "존재한다",
                "object_": "존재",
                "is_negation": False,
                "original": "EVE 존재한다",
            },
            "confidence": 0.9,
            "is_innate": True,
        }
    }

    nl = make_nl()

    result = nl.load_beliefs(beliefs_dict=beliefs)

    assert result["beliefs_loaded"] == 1
    assert "B1" in nl.beliefs
    assert not hasattr(nl, "_generate_mock_beliefs")


def test_innate_protection_preserves_subject_and_object_semantics():
    beliefs = {
        "B1": {
            "belief_id": "B1",
            "statement": "EVE 존재한다",
            "triple": {
                "subject": "EVE",
                "predicate_text": "존재한다",
                "object_": "존재",
                "is_negation": False,
                "original": "EVE 존재한다",
            },
            "confidence": 0.9,
            "is_innate": True,
        }
    }

    class FakeNL:
        def __init__(self):
            self.beliefs = {}

        def load_beliefs(
            self,
            path=None,
            beliefs_dict=None,
            max_beliefs=None,
        ):
            self.beliefs = dict(beliefs_dict or {})
            return {
                "beliefs_loaded": len(self.beliefs),
                "categories_added": 0,
                "connections_made": 0,
            }

    class FakeSAForEVE:
        def __init__(self):
            self.weights = {}

    eve = EVE_TierAB.__new__(EVE_TierAB)
    eve.nl = FakeNL()
    eve.sa = FakeSAForEVE()
    eve.protected_categories = set()

    eve.learn_beliefs(beliefs_dict=beliefs)

    assert "EVE" in eve.protected_categories
    assert "존재" in eve.protected_categories
    assert "존재한다" not in eve.protected_categories
