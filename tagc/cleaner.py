import json
import logging
import re
from functools import partial
from operator import methodcaller
from typing import List, NamedTuple, Optional, Tuple

from toolz.functoolz import pipe

logging.basicConfig()
logger = logging.getLogger("cleaner")

_FIELDS = [
    "PARTICLES:",
    "MEGAKARYOCYTES",
    "EXTRINSIC",
    "ERYTHROPOIESIS",
    "GRANULOPOIESIS",
    "RETICULUM",
    "HEMOSIDERIN",
    "CELLULARITY",
    "COMMENT:",
    "INTERPRETATION",
    "CONCLUSIONS",
]

MAX_KEY_LEN = 59
MIN_KEY_LEN = 5
MIN_VALUE_LEN = 3
ENTRY_LEN = 2


class Entry(NamedTuple):
    key: str
    value: str


class Cleaner(object):
    def __init__(self):
        self._legit = 0
        self._warning = 0
        self._warning_collector = set()

    @property
    def not_used_texts(self):
        return self._warning_collector

    def parse_aspirate_result(self, aspirate_result_text: str) -> dict:
        return pipe(aspirate_result_text, self._clean_text, self._parse_text)

    def _clean_text(self, aspirate_result_text: str) -> str:
        return pipe(
            aspirate_result_text,
            _remove_signature,
            _remove_inline_space,
            _remove_end_space,
            _remove_reporting_system,
            _replace_add,
        )

    def _parse_text(self, aspirate_result_cleaned_text: str) -> dict:
        return pipe(aspirate_result_cleaned_text, self._entry_divide, self._form_dict)

    def _entry_divide(self, text: str) -> List[str]:
        for term in _FIELDS:
            text = text.replace(term, "\n" + term, 1)
        text = _special_field(text)
        return text.split("\n")

    def _form_dict(self, entry_texts: List[str]) -> dict:
        return {
            entry.key: entry.value
            for entry in map(self._extract_entry, entry_texts)
            if entry is not None
        }

    def _extract_entry(self, entry_text: str) -> Optional[Entry]:
        sudo_entry = tuple(map(str.strip, entry_text.split(":", 1)))
        entry = None
        if Cleaner._is_legit_entry(sudo_entry):
            entry = self._form_entry(sudo_entry)
        if Cleaner._deserve_warning(entry, sudo_entry):
            self._process_warning_case(entry_text)
        return entry

    @staticmethod
    def _is_legit_entry(sudo_entry: Tuple[str, ...]) -> bool:
        return (
            len(sudo_entry) == ENTRY_LEN
            and _nonword_in_attr(sudo_entry[0]) is None
            and sudo_entry[1] != ""
            and MIN_KEY_LEN < len(sudo_entry[0]) < MAX_KEY_LEN
            and MIN_VALUE_LEN < len(sudo_entry[1])
        )

    def _form_entry(self, sudo_entry: tuple):
        self._legit += 1
        return Entry(sudo_entry[0], sudo_entry[1])

    @staticmethod
    def _deserve_warning(entry, sudo_entry: tuple):
        return entry is None and len(sudo_entry) != 1 and sudo_entry[1] != ""

    def _process_warning_case(self, entry_text: str):
        self._warning += 1
        self._warning_collector.add(entry_text)

    def parse_as_json(self, aspirate_result_text: str, out_path: str):
        with open(out_path, "w") as json_:
            json.dump(self.parse_aspirate_result(aspirate_result_text), json_)
        self.job_report()

    def job_report(self):
        warn_rate = self._warning / (self._legit + self._warning)
        print(
            f"Parser: {self._legit} legit fields,{self._warning}"
            f"({warn_rate : .3f}) warning"
        )


_remove_inline_space = partial(re.compile(r"\s+").sub, " ")
_remove_end_space = partial(re.compile(r"\s$").sub, "")
_nonword_in_attr = re.compile(r"\d+|\.|@+|TO|OF").search
_remove_signature = partial(re.compile(r"REPORTING PHYSICIAN.*|D[rR][.]? .*").sub, "")
_replace_add = methodcaller("replace", "&", "AND")
_remove_reporting_system = partial(re.compile(r"\(REPORTING.*\)").sub, "")
_special_field = partial(re.compile(r"PERIPHERAL (?!BLOOD[\s.])").sub, "\nPERIPHERAL ")


def has_content(case: dict):
    return len(case) != 0
