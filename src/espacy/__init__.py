""" eSpaCy """

from espacy.parser import SpacyParser
from espacy.tregex import SpacyPhraseMatcher, SpacyTokenMatcher, TokenRegExMatcher

__all__ = ["SpacyParser", "SpacyPhraseMatcher", "SpacyTokenMatcher", "TokenRegExMatcher"]
