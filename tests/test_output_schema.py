import pytest
from burner.output import DomainResult

def test_output_schema_has_no_character_fields():
    """DomainResult must not contain any prohibited field names."""
    prohibited = {"lazy", "disciplined", "personality", "diagnosis", "trait"}
    fields = set(DomainResult.model_fields.keys())
    assert fields.isdisjoint(prohibited)
