""" test eSpaCy parser """


from espacy.parser import SpacyParser


def test_spacy_parser_singleton():
    """ test eSpaCy parser """
    par_a = SpacyParser()
    par_b = SpacyParser()
    assert par_a is par_b


def test_spacy_parser_output():
    """ test eSpaCy parser output """
    par = SpacyParser()
    txt = "aaa, bbb & ccc."
    ref = "aaa , bbb & ccc ."
    assert par.test_doc(par(txt)) is True
    assert par(txt, words=True) == ref
