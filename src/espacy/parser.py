""" spaCy parser wrapper """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


from eengine import Component
from eengine.singleton import Singleton

import spacy
from spacy.attrs import ORTH, NORM
from spacy.tokens.doc import Doc
from spacy import Language


class SpacyParser(Singleton, Component):
    """ spaCy parser wrapper """

    agnostic = True
    singular = True

    implicit = {
        "model": "en_core_web_md",
        "vocab": None,
    }

    defaults = {
        "words": False
    }

    initials = {
        "logging": None
    }

    def __init__(self,
                 data: str | dict | list | None = None,
                 *,
                 name: str | None = None,
                 config: dict | None = None,
                 params: dict | None = None,
                 setups: dict | None = None,
                 **kwargs
                 ) -> None:

        self.model = None
        self.vocab = None

        super().__init__(data, name=name, config=config, params=params, setups=setups, **kwargs)

    def make(self,
             data: dict | list[str] | None = None,
             model: str = "en_core_web_md",
             vocab: str | list[str] | dict[str, str] | None = None,
             **kwargs
             ) -> None:

        model = spacy.load(model)

        if vocab:
            vocab = self.read(vocab)
            add_vocab(model, vocab)

        self.model = model
        self.vocab = model.vocab

    def proc(self, data: str, words: bool = False, **kwargs) -> str | Doc | None :
        doc = self.model(data) if isinstance(self.model, Language) else None
        txt = " ".join([tok.text for tok in doc]) if doc and words else None
        return txt or doc

    # exposed spacy methods
    def make_doc(self, data: str) -> Doc:
        return self.model.make_doc(data)

    @staticmethod
    def test_doc(data) -> bool:
        return isinstance(data, Doc)


def add_vocab(model: spacy.Language, vocab: str | list[str] | dict[str, str]) -> None:
    """
    add additional vocabulary
    :param model: spacy model
    :type model: spacy.Language
    :param vocab: vocabulary (as {token: norm})
    :type vocab: str | dict[str, str]
    """
    vocab = vocab.split() if isinstance(vocab, str) else vocab
    vocab = dict(zip(vocab, vocab)) if isinstance(vocab, list) else vocab

    for token, value in (vocab or {}).items():
        model.tokenizer.add_special_case(token, [{ORTH: token, NORM: value}])
