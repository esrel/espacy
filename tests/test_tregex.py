""" SpacyMatcher tests """

import pytest

from espacy.tregex import SpacyPhraseMatcher, SpacyTokenMatcher, TokenRegExMatcher
from espacy.tregex import load_gazetteers, isa_optional, clean_pattern, get_target_span
from espacy.tregex import build_token_context, apply_token_context
from espacy.tregex import get_output_spans
from eengine.reader import dump_list
from ematcher.consolidator import contain


def test_spacy_phrase_matcher(rules: dict[str, list[str]],
                              tests: list[str],
                              parts_matches: list[list[tuple[str, int, int]]]
                              ) -> None:
    """
    test matches
    :param rules: rules
    :type rules: dict[str, list[str]]
    :param tests: test data
    :type tests: list[str]
    :param parts_matches: results
    :type parts_matches: list[list[tuple[str, int, int]]]
    """
    matcher = SpacyPhraseMatcher(rules)
    for seq, res in zip(tests, parts_matches):
        matches = matcher(seq)
        assert set(matches) == set(res)


def test_spacy_token_matcher(token_rules: dict[str, list[list[dict]]],
                             tests: list[str],
                             parts_matches: list[list[tuple[str, int, int]]]
                             ) -> None:
    """
    test matches
    :param token_rules: rules
    :type token_rules: dict[str, list[str]]
    :param tests: test data
    :type tests: list[str]
    :param parts_matches: results
    :type parts_matches: list[list[tuple[str, int, int]]]
    """
    matcher = SpacyTokenMatcher(token_rules)
    for seq, res in zip(tests, parts_matches):
        matches = matcher(seq)
        # per-label consolidation:
        # does not return several overlapping entities of the same type
        assert set(matches) == set(contain(res, labels=True))


def test_token_regex_matcher(token_rules: dict[str, list[list[dict]]],
                             tests: list[str],
                             parts_matches: list[list[tuple[str, int, int]]]
                             ) -> None:
    """
    test matches
    :param token_rules: rules
    :type token_rules: dict[str, list[str]]
    :param tests: test data
    :type tests: list[str]
    :param parts_matches: results
    :type parts_matches: list[list[tuple[str, int, int]]]
    """
    # test compatibility: no custom extension
    matcher = TokenRegExMatcher(token_rules)
    for seq, res in zip(tests, parts_matches):
        matches = matcher(seq)
        assert set(matches) == set(contain(res, labels=True))

    # test custom token attributes
    extra_rules = {"F": [[{"_": {"E": True}, "OP": "+"}]]}
    extra_match = [("E", 0, 1), ("E", 1, 2)]
    extra_ruler = TokenRegExMatcher(extra_rules)

    # no other matches, but 'F'
    for seq in tests:
        hyp = extra_ruler(seq, matches=extra_match)
        assert set(hyp) == set([('F', 0, 1)] if len(seq.split()) == 1 else [('F', 0, 2)])

    # adjacent match tests
    hyp = extra_ruler("a e b i", matches=[("E", 0, 1), ("E", 1, 2), ("E", 3, 4)])
    assert hyp == [('F', 0, 2), ('F', 3, 4)]


def test_load_gazetteers() -> None:
    """ test load_gazetteers """
    path = '/tmp/a.gaz'
    gazs = ["aaa", "bbb", "ccc", "ddd"]
    data = {
        "A": [[{"LOWER": {"IN": path}, "OP": "?"}]],
        "B": [[{"LOWER": {"FUZZY": {"NOT_IN": path}}}, {"LOWER": {"IN": ["xxx"]}}]]
    }
    refs = {
        "A": [[{"LOWER": {"IN": gazs}, "OP": "?"}]],
        "B": [[{"LOWER": {"FUZZY": {"NOT_IN": gazs}}}, {"LOWER": {"IN": ["xxx"]}}]]
    }

    dump_list(gazs, path)

    assert load_gazetteers(data) == refs


# token context tests
@pytest.mark.parametrize("pattern, result", [
    ([{}, {}], False),
    ([{"OP": "?"}, {"OP": "*"}], True),
    ([{"OP": "*"}, {"OP": "+"}], False),
])
def test_isa_optional(pattern: list[dict], result: bool) -> None:
    """
    test isa_optional
    :param pattern: pattern
    :type pattern: list[dict]
    :param result: result
    :type result: bool
    """
    assert isa_optional(pattern) == result


@pytest.mark.parametrize("pattern, cleaned", [
    ([{"TEXT": "aaa"}, {"TEXT": "bbb"}], [{"TEXT": "aaa"}, {"TEXT": "bbb"}]),
    ([{"TEXT": "aaa", "BOS": True}, {"TEXT": "bbb", "EOS": True}],
     [{"TEXT": "aaa"}, {"TEXT": "bbb"}]),
])
def test_clean_pattern(pattern: list[dict], cleaned: list[dict]) -> None:
    """
    test clean pattern
    :param pattern: input pattern
    :type pattern: list[dict]
    :param cleaned: clean pattern
    :type cleaned: list[dict]
    """
    assert clean_pattern(pattern) == cleaned


@pytest.mark.parametrize("pattern, span", [
    ([{"TEXT": "aaa"}, {"TEXT": "bbb"}, {"TEXT": "ccc"}, {"TEXT": "ddd"}], (0, 4)),
    ([{"TEXT": "aaa"}, {"TEXT": "bbb", "BOS": True}, {"TEXT": "ccc"}, {"TEXT": "ddd"}], (1, 4)),
    ([{"TEXT": "aaa"}, {"TEXT": "bbb"}, {"TEXT": "ccc", "EOS": True}, {"TEXT": "ddd"}], (0, 3)),
    ([{"TEXT": "aaa"}, {"TEXT": "bbb", "BOS": True},
      {"TEXT": "ccc", "EOS": True}, {"TEXT": "ddd"}], (1, 3)),
    ([{"TEXT": "aaa"}, {"TEXT": "bbb", "BOS": True, "EOS": True},
      {"TEXT": "ccc"}, {"TEXT": "ddd"}], (1, 2)),
    # errors
    ([{"TEXT": "aaa"}, {"TEXT": "bbb", "BOS": True},
      {"TEXT": "ccc", "BOS": True}, {"TEXT": "ddd"}], (0, 0)),
    ([{"TEXT": "aaa"}, {"TEXT": "bbb", "EOS": True},
      {"TEXT": "ccc", "EOS": True}, {"TEXT": "ddd"}], (0, 0)),
    ([{"TEXT": "aaa"}, {"TEXT": "bbb", "EOS": True},
      {"TEXT": "ccc", "BOS": True}, {"TEXT": "ddd"}], (0, 0)),
])
def test_get_target_span(pattern: list[dict], span: tuple[int, int]) -> None:
    """
    test get_target_span
    :param pattern: pattern
    :type pattern: list[dict]
    :param span: span
    :type span: tuple[int, int]
    """
    if span == (0, 0):
        with pytest.raises(ValueError):
            get_target_span(pattern)
    else:
        assert get_target_span(pattern) == span


@pytest.fixture(name="con_pat")
def context_patterns() -> list[list[dict]]:
    """ returns context patterns """
    return [
        # no marking
        [{"TEXT": "aaa"}, {"TEXT": "bbb"}, {"TEXT": "ccc"}, {"TEXT": "ddd"}],
        # 1 OUT
        [{"TEXT": "aaa"},
         {"TEXT": "bbb", "OUT": "R.B"},
         {"TEXT": "ccc"}, {"TEXT": "ddd"}],
        [{"TEXT": "aaa"},
         {"TEXT": "bbb", "OUT": "R.B", "BOS": True, "EOS": True},
         {"TEXT": "ccc"}, {"TEXT": "ddd"}],
        # BOS & NOT EOS
        [{"TEXT": "aaa"},
         {"TEXT": "bbb", "OUT": "R.B", "BOS": True},
         {"TEXT": "ccc"}, {"TEXT": "ddd"}],
        # EOS & NOT BOS
        [{"TEXT": "aaa"},
         {"TEXT": "bbb", "OUT": "R.B", "EOS": True},
         {"TEXT": "ccc"}, {"TEXT": "ddd"}],
        # 2 OUT
        [{"TEXT": "aaa"}, {"TEXT": "bbb", "OUT": "R.B"},
         {"TEXT": "ccc"}, {"TEXT": "ddd", "OUT": "R.D"}],
    ]


@pytest.fixture(name="con_ref")
def context_reference_spans() -> list[list[tuple[str, int, int]]]:
    """
    :return: reference spans
    :rtype: list[list[tuple[str, int, int]]]
    """
    return [
        [],
        [('R.B', 1, 2)], [('R.B', 1, 2)],
        [('R.B', 1, 4)], [('R.B', 0, 2)],
        [('R.B', 1, 2), ('R.D', 3, 4)]
    ]


def test_get_output_spans(con_pat: list[list], con_ref: list[list]) -> None:
    """
    test get output spans
    :param con_pat: patterns
    :type con_pat: list[list]
    :param con_ref: reference spans
    :type con_ref: list[list]
    """
    for pattern, reference in zip(con_pat, con_ref):
        assert reference == get_output_spans(pattern)


def test_build_token_context(token_rules: dict[str, list[list[dict]]]) -> None:
    """
    test build token context
    :param token_rules: rules
    :type token_rules: dict[str, list[str]]
    """
    patterns, contexts = build_token_context(token_rules)

    assert patterns == token_rules
    assert contexts == {'mapping': {}, 'context': {}, 'options': {}}

    # new custom pattern
    pattern = [{"TEXT": "aaa"},
               {"TEXT": "bbb", "BOS": True},
               {"TEXT": "ccc", "EOS": True},
               {"TEXT": "ddd"}]
    context_pats = {
        'subs.0': [[{'TEXT': 'aaa'}, {'TEXT': 'bbb'}, {'TEXT': 'ccc'}, {'TEXT': 'ddd'}]],
        'subs.0.0.prefix': [[{'TEXT': 'aaa'}]],
        'subs.0.0.target': [[{'TEXT': 'bbb'}, {'TEXT': 'ccc'}]],
        'subs.0.0.suffix': [[{'TEXT': 'ddd'}]]
    }
    context_cons = {"mapping": {"subs.0.0": "subs", "subs.0": "subs"},
                    "context": {'subs.0': {'subs.0.0': (
                        'subs.0.0.prefix', 'subs.0.0.target', 'subs.0.0.suffix')}},
                    "options": {'subs.0.0.prefix': False,
                                'subs.0.0.target': False,
                                'subs.0.0.suffix': False}}

    rules = token_rules | {"subs": [pattern]}

    patterns, contexts = build_token_context(rules)

    assert patterns == {**token_rules, **context_pats}

    assert contexts == context_cons


def test_apply_token_context() -> None:
    """ test apply_token_context """
    contexts = {"mapping": {"subs.0": "subs"},
                "context": {'subs.0': {'subs.0': ('subs.0.prefix',
                                                  'subs.0.target',
                                                  'subs.0.suffix')}},
                "options": {'subs.0.prefix': False,
                            'subs.0.target': False,
                            'subs.0.suffix': False}}

    matches = [("subs.0", 0, 4),
               ('subs.0.prefix', 0, 1), ('subs.0.target', 1, 3), ('subs.0.suffix', 3, 4)]

    results = apply_token_context(matches, **contexts)

    assert results == [("subs", 1, 3)]


def test_get_spacy_ents(rules: dict[str, list[str]]) -> None:
    """
    test get_spacy_ents
    :param rules: rules
    :type rules: dict[str, list[str]]
    """
    example = "100% 2024-03-31"
    matcher = SpacyPhraseMatcher(rules, ents=True)
    matches = matcher(example)
    assert matches == [("PERCENT", 0, 2), ("DATE", 2, 7)]
