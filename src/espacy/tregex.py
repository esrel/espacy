""" Token RegEx Matcher """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


from collections import defaultdict
from itertools import groupby, product
from typing import Callable

from spacy.tokens import Token, Doc
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

from eengine.reader import read_list
from ematcher.matcher import BaseMatcher
from ematcher.consolidator import overlap
from espacy.parser import SpacyParser


# pylint: disable=no-member


PTS = tuple[str | None, str, str | None]


class SpacyMatcher(BaseMatcher):
    """ spacy matcher's base class """

    def __init__(self, data: str | dict[str, list], **kwargs) -> None:
        self.spacy = None
        self.vocab = None
        self.inner = None
        super().__init__(data, **kwargs)

    def make(self,
             data: dict[str, list],
             lang: str = "en_core_web_md",
             ents: bool = False,
             **kwargs) -> None:
        self.spacy = SpacyParser(config={"model": lang})
        self.vocab = self.spacy.vocab
        self.model = self.init_spacy_matcher(data, **kwargs)
        self.inner = ents

    def prep(self, data: str | Doc, **kwargs) -> Doc:
        return data if isinstance(data, Doc) else self.spacy(data)

    def find(self, data: Doc) -> list[tuple[str, int, int]]:
        matches = [(str(self.vocab.strings[y]), b, e) for y, b, e in self.model(data)]
        matches = (matches + get_spacy_ents(data)) if self.inner else matches
        return matches

    def init_spacy_matcher(self, data: dict[str, list], **kwargs) -> Callable:
        """
        build spacy matcher
        :param data: rules data
        :type data: dict[str, list]
        :return: matcher model
        :rtype: callable | object
        """
        raise NotImplementedError


class SpacyPhraseMatcher(SpacyMatcher):
    """ spacy PhraseMatcher wrapper """

    def init_spacy_matcher(self,
                           data: dict[str, list],
                           attr: str = "ORTH",
                           **kwargs
                           ) -> Callable:

        matcher = PhraseMatcher(self.vocab, attr=attr, validate=True)

        for label, patterns in data.items():
            matcher.add(label, [(self.spacy.make_doc(pattern) if attr == "ORTH"
                                 else self.spacy(pattern)) for pattern in patterns])
        return matcher


class SpacyTokenMatcher(SpacyMatcher):
    """ spacy Matcher wrapper """

    def init_spacy_matcher(self, data: dict[str, list[list[dict]]], **kwargs) -> Callable:

        matcher = Matcher(self.vocab, validate=True)

        for label, patterns in data.items():
            matcher.add(label, patterns, greedy="LONGEST")

        return matcher


class TokenRegExMatcher(SpacyMatcher):
    """ spacy Matcher with custom token attributes from matches """

    def __init__(self, data: str | dict[str, list], **kwargs) -> None:
        self.attrs = None
        self.context = None
        super().__init__(data, **kwargs)

    def prep(self,
             data: str,
             matches: list[tuple[str, int, int]] | None = None,
             **kwargs
             ) -> Doc:
        data = data if isinstance(data, Doc) else self.spacy(data)
        data = add_match_attrs(data, [(y, b, e) for y, b, e in (matches or []) if y in self.attrs])
        return data

    def find(self, data: Doc) -> list[tuple[str, int, int]]:
        matches = [(str(self.vocab.strings[y]), b, e) for y, b, e in self.model(data)]
        matches = apply_token_context(matches, **self.context)
        matches = (matches + get_spacy_ents(data)) if self.inner else matches
        return matches

    def init_spacy_matcher(self, data: dict[str, list[list]], **kwargs) -> Callable:
        # custom attributes from patterns
        self.attrs = get_match_attrs(data) or []
        set_match_attrs(self.attrs)

        matcher = Matcher(self.vocab, validate=True)

        # load token gazetteers
        data = load_gazetteers(data)

        # make token context
        data, context = build_token_context(data)

        self.context = context

        for label, patterns in data.items():
            matcher.add(label, patterns, greedy="LONGEST")

        return matcher


# match as a token attribute
def get_match_attrs(data: dict[str, list[list[dict]]]) -> list[str]:
    """
    get match attribute string IDs
    :param data: pattern data
    :type data: dict[str, list[list[dict]]]
    :return: custom attributes
    :rtype: list[str]
    """
    attrs = set()
    for _, patterns in data.items():
        for pattern in patterns:
            pattern_attrs = {attr for token in pattern for attr in token.get("_", {}).keys()}
            attrs = attrs.union(pattern_attrs)
    return sorted(list(attrs))


def set_match_attrs(data: list[str]) -> None:
    """
    set match attributes as token attributes
    :param data: list of attributes
    :type data: list[str]
    """
    for attr in data:
        if not Token.has_extension(attr):
            Token.set_extension(attr, default=False)


def add_match_attrs(data: Doc, matches: list[tuple[str, int, int]]) -> Doc:
    """
    add provided matches as token attributes
    :param data: spacy Doc object
    :type data: Doc
    :param matches: matches to be used as token attrs
    :type matches: list[tuple[str, int, int]]
    :return: updated Doc object
    :rtype: Doc
    """
    for label, bos, eos in matches:
        for token in data:
            if bos <= token.i < eos:
                setattr(token._, label, True)
    return data


# token from gazetteer
def load_gazetteers(data: dict[str, list]) -> dict[str, list]:
    """
    load gazetteers into token regex pattern files
    :param data: token regex patterns
    :type data: dict[str, list]
    :return: updated rules
    :rtype: dict[str, list]
    """
    return {label: [[load_token_gaz(token) for token in pattern] for pattern in patterns]
            for label, patterns in data.items()}


def load_token_gaz(data: dict, keys: list[str] | None = None) -> dict:
    """
    load gaz of a token object
    :param data: token
    :type data: dict
    :param keys: keys to load for, defaults to None
    :type keys: list[str], optional
    :return: update token
    :rtype: dict
    """
    keys = keys or ["IN", "NOT_IN"]
    return {k: (load_token_gaz(v, keys=keys) if isinstance(v, dict) else
                sorted(list(set(read_list(v)))) if (isinstance(v, str) and k in keys) else v)
            for k, v in data.items()} if isinstance(data, dict) else data


# local token context
def isa_optional(pattern: list[dict]) -> bool:
    """
    check if pattern is optional
    :param pattern: pattern
    :type pattern: list[dict]
    :return: truth value
    :rtype: bool
    """
    return all(token.get("OP") in ["?", "*"] for token in pattern)


def clean_pattern(pattern: list[dict]) -> list[dict]:
    """
    remove custom keys
    :param pattern: pattern
    :type pattern: list[dict]
    :return: pattern
    :rtype: list[dict]
    """
    keys = ["OUT", "BOS", "EOS"]
    return [{k: v for k, v in token.items() if k not in keys} for token in pattern]


def get_target_span(pattern: list[dict]) -> tuple[int, int]:
    """
    get target span of a pattern with token context
    :param pattern: token regex pattern
    :type pattern: list[dict]
    :return: target span
    :rtype: tuple[int, int]
    """
    bos = [i for i, token in enumerate(pattern) if "BOS" in token]
    eos = [i + 1 for i, token in enumerate(pattern) if "EOS" in token]

    if len(bos) > 1 or len(eos) > 1:
        raise ValueError(f"invalid span: {pattern}")

    bos = min(bos, default=0)
    eos = max(eos, default=len(pattern))

    if bos >= eos:
        raise ValueError(f"invalid span: ({bos}, {eos})")

    return bos, eos


# OUT attribute for additional spans on pattern
def get_output_spans(pattern: list[dict]) -> list[tuple[str, int, int]]:
    """
    get spans with labels to be added to the output
    :param pattern: token regex pattern
    :type pattern: list[dict]
    :return: labeled spans
    :rtype: list[tuple[str, int, int]]
    """
    keys = ["OUT", "BOS", "EOS"]
    marks = [(token.get("OUT"), token.get("BOS"), token.get("EOS"), i)
             for i, token in enumerate(pattern) if any(x in token for x in keys)]

    spans = sorted(list(
        {(y,
          (i if (b or (y and not e)) else
           max({x[3] for x in marks if (x[0] == y and x[3] and x[3] < i)},
               default=0)
           if e else None),
          (i + 1 if (e or (y and not b)) else
           min({x[3] + 1 for x in marks if (x[0] == y and x[2] and x[2] > i)},
               default=len(pattern))
           if b else None))
         for y, b, e, i in marks if y}), key=lambda x: (x[1], -x[2]))

    return spans


def build_parts_context(span: tuple[str, int, int],
                        pattern: list[dict]
                        ) -> tuple[dict, dict, dict]:
    """
    build context for a span w.r.t. pattern
    :param span: a span as (label, bos, eos)
    :type span: tuple[str, int, int]
    :param pattern: token regex pattern
    :type pattern: list[dict]
    :return: context
    :rtype: tuple[dict, dict, dict]
    """
    label, bos, eos = span
    # create sub-patterns & labels
    noms = [f"{label}.{e}" for e in ["prefix", "target", "suffix"]]
    subs = [pattern[:bos], pattern[bos:eos], pattern[eos:]]

    # add sub-patterns to patterns to be used for updates
    updates = {y: [p] for y, p in zip(noms, subs) if p}
    # add label-sub-pattern mapping
    context = {label: tuple((y if p else None) for y, p in zip(noms, subs))}
    # add optionality mapping
    options = {y: isa_optional(p) for y, p in zip(noms, subs) if p}

    return updates, context, options


def build_spans_context(spans: list[tuple[str, int, int]],
                        index: int,
                        pattern: list[dict]
                        ) -> tuple[dict, dict, dict, dict]:
    """
    build context for a set of spans w.r.t. pattern
    :param spans: labeled spans
    :type spans: list[tuple[str, int, int]]
    :param index: pattern index for unique name
    :type index: int
    :param pattern: token regex pattern
    :type pattern: list[dict]
    :return: context
    :rtype: tuple[dict, dict, dict]
    """
    updates = {}
    context = {}
    options = {}
    mapping = {}
    for i, (lbl, bos, eos) in enumerate(spans):
        nom = f"{lbl}.{index}.{i}"
        upd, con, opt = build_parts_context((nom, bos, eos), pattern)
        mapping.update({nom: lbl})
        updates.update(upd)
        context.update(con)
        options.update(opt)

    return updates, context, options, mapping


def build_token_context(data: dict[str, list]) -> tuple[dict, dict]:
    """
    build token context from data (i.e. rules)
    :param data: rule patterns
    :type data: dict[str, list]
    :return: patterns, mapping, optionality
    :rtype: tuple[dict, dict]
    """
    mapping: dict = {}  # mapping from unique to original labels
    context: dict = {}  # mapping from unique to token context labels
    options: dict = {}  # mapping from token context labels to optionality flags
    updates = defaultdict(list)  # clean patterns

    for label, patterns in data.items():
        for i, pattern in enumerate(patterns):

            # clean pattern
            pat = clean_pattern(pattern)

            spans = build_context_spans(pattern, label)

            if spans:
                # create additional patterns & context
                token_context = build_spans_context(spans, i, pat)

                context[f"{label}.{i}"] = token_context[1]
                options.update(token_context[2])
                mapping.update(token_context[3])
                updates.update(token_context[0])

                # add whole pattern with new label
                updates[f"{label}.{i}"].append(pat)
                mapping[f"{label}.{i}"] = label

            else:
                updates[label].append(pat)

    return updates, {"mapping": mapping, "context": context, "options": options}


def build_context_spans(pattern: list[dict], label: str) -> list[tuple[str, int, int]]:
    """
    build context spans
    :param pattern: token regex pattern
    :type pattern: list[dict]
    :param label: pattern label
    :type label: str
    :return: spans
    :rtype: list[tuple[str, int, int]]
    """
    # OUT & BOS & EOS
    spans = get_output_spans(pattern)

    if spans:
        return spans

    # BOS & EOS
    bos, eos = get_target_span(pattern)

    if (bos, eos) != (0, len(pattern)):
        return [(label, bos, eos)]

    return []


def apply_token_context(matches: list[tuple[str, int, int]],
                        mapping: dict[str, str] | None = None,
                        context: dict[str, dict[str, PTS]] | None = None,
                        options: dict[str, bool] | None = None
                        ) -> list[tuple[str, int, int]]:
    """
    apply token context
    :param matches: matches
    :type matches: list[tuple[str, int, int]]
    :param mapping: mapping from unique to output labels
    :type mapping: dict[str, str]
    :param context: mapping to contexts as {label: {span_label: (prefix, target, suffix)}}
    :type context: dict[str, dict[str, tuple[str | None, str, str | None]]]
    :param options: token context optional flag
    :type options: dict[str, bool]
    :return: matches
    :rtype: list[tuple[str, int, int]]
    """
    if not options or not context:
        return matches

    no_context = [(y, b, e) for y, b, e in matches if (y not in context and y not in options)]

    if len(matches) == len(no_context):
        return matches

    in_context = clean_token_context([m for m in matches if m not in no_context],
                                     mapping, context, options)

    return no_context + in_context


def clean_token_context(matches: list[tuple[str, int, int]],
                        mapping: dict[str, str] | None = None,
                        context: dict[str, dict[str, PTS]] | None = None,
                        options: dict[str, bool] | None = None
                        ) -> list[tuple[str, int, int]]:
    """
    clean token context: remove context spans
    :param matches: matches
    :type matches: list[tuple[str, int, int]]
    :param mapping: mapping from unique to output labels
    :type mapping: dict[str, str]
    :param context: mapping to contexts as {label: {span_label: (prefix, target, suffix)}}
    :type context: dict[str, dict[str, tuple[str | None, str, str | None]]]
    :param options: token context optional flag
    :type options: dict[str, bool]
    :return: matches
    :rtype: list[tuple[str, int, int]]
    """
    if not options or not context:
        return matches

    mapping = mapping or {}

    wholes = {k: {(b, e) for y, b, e in matches if y == k} for k in context}

    result = []
    for label, spans in wholes.items():
        # add whole spans, if label in context
        output = {mapping.get(y) for y in context.get(label, {}).keys()}
        result.extend(([(mapping.get(label), b, e) for b, e in spans]
                       if mapping.get(label) not in output else []))

        for bos, eos in spans:
            frame = []
            for name, parts in context.get(label, {}).items():
                subs = clean_parts_context((bos, eos), matches, parts, options)
                subs = [(mapping.get(name), b, e) for b, e in subs]
                frame.extend(subs)

            frame = clean_parts_frame(frame)
            result.extend(frame)

    return result


def clean_parts_context(indices: tuple[int, int],
                        matches: list[tuple[str, int, int]],
                        context: PTS | None = None,
                        options: dict[str, bool] | None = None
                        ) -> list[tuple[int, int]]:
    """
    clean match context parts
    :param indices: bos & eos indices
    :type indices: tuple[int, int]
    :param matches: matches
    :type matches: list[tuple[str, int, int]]
    :param context: mapping to contexts as {label: {span_label: (prefix, target, suffix)}}
    :type context: dict[str, dict[str, tuple[str | None, str, str | None]]]
    :param options: token context optional flag
    :type options: dict[str, bool]
    :return: match spans
    :rtype: list[tuple[int, int]]
    """
    options = options or {}
    bos, eos = indices
    prefix, target, suffix = context

    target_set = {(b, e) for y, b, e in matches
                  if y == target and bos <= b < eos and bos < e <= eos}

    if not target_set:
        return []

    if prefix:
        prefix_set = {(b, e) for y, b, e in matches if y == prefix}
        prefix_set = prefix_set.union({(bos, bos)}) if options.get(prefix) else prefix_set
        target_set = {(b, e) for b, e in target_set if (bos, b) in prefix_set}
    else:
        target_set = {(b, e) for b, e in target_set if b == bos}

    if suffix:
        suffix_set = {(b, e) for y, b, e in matches if y == suffix}
        suffix_set = suffix_set.union({(eos, eos)}) if options.get(suffix) else suffix_set
        target_set = {(b, e) for b, e in target_set if (e, eos) in suffix_set}
    else:
        target_set = {(b, e) for b, e in target_set if e == eos}

    return sorted(list(target_set))


def clean_parts_frame(matches: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """
    group & clean part matches (check)
    :param matches: part matches
    :type matches: list[tuple[str, int, int]]
    :return: part matches
    :rtype: list[tuple[str, int, int]]
    """
    # group matches w.r.t. labels
    groups = groupby(sorted(matches), lambda x: x[0])
    groups = [sorted(list(group)) for y, group in groups]
    frames = [comb for comb in product(*groups)
              if not overlap(list(comb)) and len(groups) == len(comb)]
    return [e for x in frames for e in x]


def get_spacy_ents(data: Doc,
                   labels: list[str] | None = None,
                   mapper: dict[str, str] | None = None
                   ) -> list[tuple[str, int, int]]:
    """
    get spacy entities
    :param data: spacy document
    :type data: doc
    :param labels: labels to get; defaults to None
    :type labels: list[str], optional
    :param mapper: label mapping; defaults to None
    :type mapper: dict[str, str], optional
    :return: matches
    :rtype: list[tuple[str, int, int]]
    """
    matches = [(ent.label_, ent.start, ent.end) for ent in data.ents]
    matches = matches if not labels else [(y, b, e) for y, b, e in matches if y in (labels or [])]
    matches = matches if not mapper else [(mapper.get(y, y), b, e) for y, b, e in matches]
    return matches
