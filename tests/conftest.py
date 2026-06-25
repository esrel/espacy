"""
fixtures

Symbols: a, b, c, d
Lexicon: aaa, bbb, ccc, ddd
Labels : A, B, C, D
"""

import pytest

from ematcher.lookup import gaz2tok


@pytest.fixture
def lexicon() -> list[str]:
    """
    lexicon
    :return: tokens of the lexicon
    :rtype: list[str]
    """
    return ['aaa', 'bbb', 'ccc', 'ddd']


@pytest.fixture(name="rules")
def gazetteer() -> dict[str, list[str]]:
    """
    gazetteer (matcher rules)
    :return: gazetteer
    :rtype: dict[str, list[str]]
    """
    return {
        "A": ["aaa", "aaa bbb aaa", "aaa ccc aaa", "aaa ddd aaa"],
        "B": ["bbb", "bbb aaa bbb", "bbb ccc bbb", "bbb ddd bbb"],
        "C": ["ccc", "ccc aaa ccc", "ccc bbb ccc", "ccc ddd ccc"],
        "D": ["ddd", "ddd aaa ddd", "ddd bbb ddd", "ddd ccc ddd"]
    }


@pytest.fixture(name="tests")
def matcher_tests():
    """ matcher test cases """
    return [
        "aaa", "bbb", "ccc", "ddd",
        "aaa bbb aaa", "aaa ccc aaa", "aaa ddd aaa",
        "bbb aaa bbb", "bbb ccc bbb", "bbb ddd bbb",
        "ccc aaa ccc", "ccc bbb ccc", "ccc ddd ccc",
        "ddd aaa ddd", "ddd bbb ddd", "ddd ccc ddd",
        "aaa bbb ccc ddd"
    ]


@pytest.fixture(name="parts_matches")
def matcher_parts_matches():
    """ exact matches """
    return [
        # "aaa", "bbb", "ccc", "ddd",
        [("A", 0, 1)], [("B", 0, 1)], [("C", 0, 1)], [("D", 0, 1)],
        # "aaa bbb aaa", "aaa ccc aaa", "aaa ddd aaa",
        [("A", 0, 3), ("A", 0, 1), ("B", 1, 2), ("A", 2, 3)],
        [("A", 0, 3), ("A", 0, 1), ("C", 1, 2), ("A", 2, 3)],
        [("A", 0, 3), ("A", 0, 1), ("D", 1, 2), ("A", 2, 3)],
        # "bbb aaa bbb", "bbb ccc bbb", "bbb ddd bbb",
        [("B", 0, 3), ("B", 0, 1), ("A", 1, 2), ("B", 2, 3)],
        [("B", 0, 3), ("B", 0, 1), ("C", 1, 2), ("B", 2, 3)],
        [("B", 0, 3), ("B", 0, 1), ("D", 1, 2), ("B", 2, 3)],
        # "ccc aaa ccc", "ccc bbb ccc", "ccc ddd ccc",
        [("C", 0, 3), ("C", 0, 1), ("A", 1, 2), ("C", 2, 3)],
        [("C", 0, 3), ("C", 0, 1), ("B", 1, 2), ("C", 2, 3)],
        [("C", 0, 3), ("C", 0, 1), ("D", 1, 2), ("C", 2, 3)],
        # "ddd aaa ddd", "ddd bbb ddd", "ddd ccc ddd",
        [("D", 0, 3), ("D", 0, 1), ("A", 1, 2), ("D", 2, 3)],
        [("D", 0, 3), ("D", 0, 1), ("B", 1, 2), ("D", 2, 3)],
        [("D", 0, 3), ("D", 0, 1), ("C", 1, 2), ("D", 2, 3)],
        # "aaa bbb ccc ddd"
        [("A", 0, 1), ("B", 1, 2), ("C", 2, 3), ("D", 3, 4)]
    ]


@pytest.fixture(name="exact_matches")
def matcher_exact_matches():
    """ parts matches """
    return [
        # "aaa", "bbb", "ccc", "ddd",
        [("A", 0, 1)], [("B", 0, 1)], [("C", 0, 1)], [("D", 0, 1)],
        # "aaa bbb aaa", "aaa ccc aaa", "aaa ddd aaa",
        [("A", 0, 3)], [("A", 0, 3)], [("A", 0, 3)],
        # "bbb aaa bbb", "bbb ccc bbb", "bbb ddd bbb",
        [("B", 0, 3)], [("B", 0, 3)], [("B", 0, 3)],
        # "ccc aaa ccc", "ccc bbb ccc", "ccc ddd ccc",
        [("C", 0, 3)], [("C", 0, 3)], [("C", 0, 3)],
        # "ddd aaa ddd", "ddd bbb ddd", "ddd ccc ddd",
        [("D", 0, 3)], [("D", 0, 3)], [("D", 0, 3)],
        # "aaa bbb ccc ddd"
        []
    ]


@pytest.fixture
def token_rules(rules) -> dict[str, list]:
    """
    spacy token patterns
    :param rules: trie rules
    :type rules: dict[str, list]
    :return: rules
    :rtype: dict[str, list]
    """
    return gaz2tok(rules)
