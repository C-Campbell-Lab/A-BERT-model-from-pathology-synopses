from collections import Counter
from typing import List


def show_replica(cases: List[dict]):
    tmp = Counter("".join(case.values()) for case in cases)
    for k, v in tmp.items():
        if v > 1:
            print(k)


def tag_patch(tag: str):

    if tag == "plasma" or tag == "plasma cell disorder":
        return "plasma cell neoplasm"
    return tag


def label_to_tags(label: str):

    tmp = list(
        map(
            lambda x: tag_patch(
                x.lower()
                .strip()
                .replace(".", "")
                .replace("syndrome no", "syndrome")
                .replace("inadquate", "inadequate")
            ),
            label.split(";"),
        )
    )
    return tmp


def get_unlabelled(all_cases: List[dict], other_cases: List[dict]):
    def not_same_content(case: dict) -> bool:
        return "".join(case.values()) not in used_cases

    used_cases = {"".join(case.values()) for case in other_cases}
    assert "" not in used_cases, "has empty case in other cases"
    unlabelled_cases = list(filter(not_same_content, all_cases))

    return unlabelled_cases
