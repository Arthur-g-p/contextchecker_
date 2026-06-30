"""Unit tests for shared utils — exact triplet deduplication (language-independent)."""

import pytest

from contextchecker.utils import (
    canonicalize_triplets,
    deduplicate_triplets,
    find_duplicate_triplets,
    triplet_key,
)


def canon(s, p, o, **meta):
    return {"subject": s, "predicate": p, "object": o, **meta}


def legacy(s, p, o, **meta):
    return {"triplet": [s, p, o], **meta}


class TestCanonicalizeTriplets:
    def test_canonicalize_legacy_triplet_array(self):
        """Legacy {"triplet": [s, p, o]} becomes {"subject": s, "predicate": p, "object": o}."""
        kg = [legacy("France", "has capital", "Paris")]
        canonicalize_triplets(kg)
        assert kg[0]["subject"] == "France"
        assert kg[0]["predicate"] == "has capital"
        assert kg[0]["object"] == "Paris"
        assert "triplet" not in kg[0]

    def test_canonicalize_preserves_canonical_triplet(self):
        """Already-canonical dicts are left untouched."""
        kg = [canon("France", "has capital", "Paris")]
        original = kg[0].copy()
        canonicalize_triplets(kg)
        assert kg[0] == original

    def test_canonicalize_preserves_human_label(self):
        """human_label survives normalization."""
        kg = [legacy("A", "B", "C", human_label="Entailment")]
        canonicalize_triplets(kg)
        assert kg[0]["human_label"] == "Entailment"

    def test_canonicalize_preserves_extra_fields(self):
        """Extra custom fields like aliases survive normalization."""
        kg = [{"triplet": ["X", "is", "Y"], "alias": "value"}]
        canonicalize_triplets(kg)
        assert kg[0]["subject"] == "X"
        assert kg[0]["alias"] == "value"

    def test_canonicalize_mixed_list(self):
        """Mix of legacy and canonical triplets in the same list are handled correctly."""
        kg = [
            legacy("A", "B", "C", human_label="Entailment"),
            canon("X", "Y", "Z", human_label="Contradiction")
        ]
        canonicalize_triplets(kg)
        assert kg[0] == {"subject": "A", "predicate": "B", "object": "C", "human_label": "Entailment"}
        assert kg[1] == {"subject": "X", "predicate": "Y", "object": "Z", "human_label": "Contradiction"}

    def test_canonicalize_empty_list(self):
        """Empty list — no crash."""
        kg = []
        canonicalize_triplets(kg)
        assert kg == []

    def test_canonicalize_triplet_key_with_subject_already_present(self):
        """If both 'triplet' and 'subject' keys exist, do not overwrite 'subject'."""
        kg = [{"triplet": ["X", "Y", "Z"], "subject": "existing"}]
        canonicalize_triplets(kg)
        assert kg[0]["subject"] == "existing"
        assert kg[0]["triplet"] == ["X", "Y", "Z"]

    def test_canonicalize_mutation_in_place(self):
        """Normalization mutates the original list items in-place."""
        kg = [legacy("A", "B", "C")]
        original_item = kg[0]
        canonicalize_triplets(kg)
        assert kg[0] is original_item

    def test_short_legacy_is_padded(self):
        """Short arrays are padded safely without crashing."""
        kg = [{"triplet": ["a"]}]
        canonicalize_triplets(kg)
        assert kg[0] == {"subject": "a", "predicate": "", "object": ""}



class TestTripletKey:
    def test_whitespace_trimmed(self):
        assert triplet_key(canon(" a ", "b", "c ")) == triplet_key(canon("a", "b", "c"))

    def test_case_sensitive_by_default(self):
        assert triplet_key(canon("A", "b", "c")) != triplet_key(canon("a", "b", "c"))

    def test_case_insensitive_flag(self):
        assert triplet_key(canon("A", "B", "C"), case_insensitive=True) == \
               triplet_key(canon("a", "b", "c"), case_insensitive=True)

    def test_non_canonical_dict_raises(self):
        # Triplets are canonicalized at ingestion; a non-canonical dict reaching
        # this function is a contract violation, not a fallback.
        with pytest.raises(KeyError):
            triplet_key({"triplet": ["a", "b", "c"]})


class TestDeduplicate:
    def test_exact_dup_removed(self):
        out = deduplicate_triplets([canon("a", "b", "c"), canon("a", "b", "c")])
        assert len(out) == 1

    def test_distinct_triplets_kept(self):
        out = deduplicate_triplets([canon("a", "b", "c"), canon("a", "b", "d")])
        assert len(out) == 2

    def test_first_occurrence_and_metadata_preserved(self):
        out = deduplicate_triplets([
            canon("a", "b", "c", human_label="Entailment", id=1),
            canon("a", "b", "c", human_label="Neutral", id=2),
        ])
        assert len(out) == 1
        assert out[0]["human_label"] == "Entailment"
        assert out[0]["id"] == 1

    def test_order_preserved(self):
        out = deduplicate_triplets([
            canon("x", "p", "1"), canon("y", "p", "2"),
            canon("x", "p", "1"), canon("z", "p", "3"),
        ])
        assert [t["subject"] for t in out] == ["x", "y", "z"]

    def test_input_not_mutated(self):
        items = [canon("a", "b", "c"), canon("a", "b", "c")]
        deduplicate_triplets(items)
        assert len(items) == 2

    def test_whitespace_variants_are_dups(self):
        out = deduplicate_triplets([canon("a", "b", "c"), canon(" a", "b", "c ")])
        assert len(out) == 1

    def test_case_sensitive_by_default(self):
        out = deduplicate_triplets([canon("Turkish", "is", "x"), canon("turkish", "is", "x")])
        assert len(out) == 2

    def test_case_insensitive_merges_variants(self):
        out = deduplicate_triplets(
            [canon("Turkish", "is", "x"), canon("turkish", "is", "x")],
            case_insensitive=True,
        )
        assert len(out) == 1

    def test_empty(self):
        assert deduplicate_triplets([]) == []

    def test_no_duplicates_unchanged(self):
        items = [canon("a", "b", "c"), canon("d", "e", "f")]
        assert deduplicate_triplets(items) == items

    def test_language_independent_non_latin(self):
        # Cyrillic + CJK — pure string handling, no language logic.
        out = deduplicate_triplets([
            canon("Москва", "столица", "Россия"),
            canon("Москва", "столица", "Россия"),
            canon("東京", "は", "日本"),
        ])
        assert len(out) == 2

    def test_german_casefold_when_insensitive(self):
        # 'ß'.casefold() == 'ss' — Unicode-correct folding, language-independent.
        out = deduplicate_triplets(
            [canon("STRASSE", "ist", "x"), canon("straße", "ist", "x")],
            case_insensitive=True,
        )
        assert len(out) == 1


class TestFindDuplicates:
    def test_returns_the_dropped_occurrences(self):
        dups = find_duplicate_triplets([canon("a", "b", "c"), canon("a", "b", "c")])
        assert len(dups) == 1
        assert dups[0] == canon("a", "b", "c")

    def test_none_when_all_unique(self):
        assert find_duplicate_triplets([canon("a", "b", "c"), canon("d", "e", "f")]) == []

    def test_counts_each_repeat(self):
        dups = find_duplicate_triplets([
            canon("a", "b", "c"), canon("a", "b", "c"), canon("a", "b", "c"),
        ])
        assert len(dups) == 2  # second and third occurrence

    def test_complements_deduplicate(self):
        items = [canon("a", "b", "c"), canon("a", "b", "c"), canon("d", "e", "f")]
        assert len(deduplicate_triplets(items)) + len(find_duplicate_triplets(items)) == len(items)

    def test_empty(self):
        assert find_duplicate_triplets([]) == []
