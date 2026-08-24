"""
Tests for .concept/ontology_metrics.json consistency

Stale-metrics drift required manual correction in two consecutive cycles
(ab7fa53: stale Q3 denominator; 7a26f1d: stale decisions_md_lines + Q5_note).
These tests re-derive every machine-checkable field from the actual .concept/
files and the tests/ tree, so a concept edit that forgets to refresh
ontology_metrics.json fails the suite instead of surfacing in eval.
"""

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONCEPT_DIR = REPO_ROOT / ".concept"

pytestmark = pytest.mark.skipif(
    not CONCEPT_DIR.is_dir(), reason=".concept/ not present in this checkout"
)


def _load_yaml(name):
    with open(CONCEPT_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_metrics():
    with open(CONCEPT_DIR / "ontology_metrics.json", encoding="utf-8") as f:
        return json.load(f)


def _terms():
    return _load_yaml("ontology.yml")["canonical_terms"]


def _claims():
    """Return (total, unmapped) claim counts from claims.ndjson."""
    total = 0
    unmapped = 0
    with open(CONCEPT_DIR / "claims.ndjson", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            json.loads(line)  # raises on malformed claim records
            total += 1
            if "__UNMAPPED__" in line:
                unmapped += 1
    return total, unmapped


def _strict_test_files():
    """Strict test-file count: test_*.py / *_test.py under tests/."""
    return [
        p
        for p in (REPO_ROOT / "tests").rglob("*.py")
        if (p.name.startswith("test_") or p.name.endswith("_test.py"))
        and p.name not in ("conftest.py", "__init__.py")
    ]


def _app_sources():
    return "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in (REPO_ROOT / "app").rglob("*.py")
    )


class TestTopLevelCounts:
    """Recorded counts must match the .concept/ files they describe"""

    def test_counts_match_concept_files(self):
        metrics = _load_metrics()
        terms = _terms()
        layers = {"core": 0, "domain": 0, "aux": 0}
        for term in terms.values():
            layers[term.get("layer")] += 1

        assert metrics["total_terms"] == len(terms)
        assert metrics["by_layer"] == layers
        assert metrics["draft_terms"] == sum(
            1 for t in terms.values() if t.get("status") == "draft"
        )
        assert metrics["invariants"] == len(_load_yaml("invariants.yml")["invariants"])
        assert metrics["mappings"] == len(_load_yaml("mappings.yml")["mappings"])
        assert metrics["ambiguities"] == len(
            _load_yaml("ambiguities.yml")["ambiguities"]
        )
        assert metrics["conflicts"] == len(_load_yaml("conflicts.yml")["conflicts"])
        assert metrics["term_queue_pending"] == len(
            _load_yaml("term_queue.yml")["queue"]
        )

    def test_decisions_md_lines_matches_file(self):
        metrics = _load_metrics()
        with open(CONCEPT_DIR / "decisions.md", encoding="utf-8") as f:
            actual = len(f.read().splitlines())
        assert metrics["decisions_md_lines"] == actual

    def test_test_files_matches_strict_glob(self):
        metrics = _load_metrics()
        assert metrics["test_files"] == len(_strict_test_files())


class TestQualityGateDerivations:
    """Q-metric percentages must derive from the counts above"""

    def test_q2_q4_q5_q6_q8_derive_from_state(self):
        metrics = _load_metrics()
        m = metrics["metrics"]
        terms = _terms()
        mappings = _load_yaml("mappings.yml")["mappings"]
        claims_total, claims_unmapped = _claims()

        mapped = sum(1 for v in mappings.values() if v.get("code_symbols"))
        evidenced = sum(1 for v in terms.values() if v.get("evidence"))
        draft = sum(1 for v in terms.values() if v.get("status") == "draft")
        unmapped_rate = (
            round(claims_unmapped / claims_total * 100, 1) if claims_total else 0.0
        )

        assert m["Q2_mapping_rate_pct"] == round(mapped / len(terms) * 100, 1)
        assert m["Q4_draft_rate_pct"] == round(draft / len(terms) * 100, 1)
        assert m["Q5_unmapped_claims_rate_pct"] == unmapped_rate
        assert m["Q6_terms_count"] == len(terms) == metrics["total_terms"]
        assert m["Q8_evidence_rate_pct"] == round(evidenced / len(terms) * 100, 1)

    def test_q3_ratio_derives_from_counts(self):
        metrics = _load_metrics()
        expected = round(
            metrics["invariants"] / metrics["test_files"] * 100, 1
        )
        assert metrics["metrics"]["Q3_invariant_to_testfile_ratio_pct"] == expected

    def test_q1_class_resolution_derives_from_app_sources(self):
        metrics = _load_metrics()
        sources = _app_sources()
        terms = _terms()
        resolved = sum(1 for name in terms if f"class {name}" in sources)
        assert (
            metrics["metrics"]["Q1_ontology_coverage_class_resolution_pct"]
            == round(resolved / len(terms) * 100, 1)
        )

    def test_q7_charter_complete(self):
        metrics = _load_metrics()
        charter = _load_yaml("charter.yml")
        complete = bool(
            charter.get("product_goal", {}).get("north_star")
            and charter.get("milestones")
        )
        assert metrics["metrics"]["Q7_charter_complete"] is complete


class TestMappingCoverage:
    """Every canonical term must have a mappings.yml entry"""

    def test_mappings_keys_match_terms(self):
        mappings = _load_yaml("mappings.yml")["mappings"]
        assert set(mappings) == set(_terms())
