"""
Run every question in eval/questions.json through search() and score it.

A question passes if its expected procedure is in the top results.
Questions with "expected": null have no answer in the docs: for those,
check the distance. It should be clearly higher than the passing ones,
because that gap is how Shahd will later know to say "I don't know".

Run from the server/ folder:
    uv run python -m eval.run_eval
"""

import json

from retrieval import search

K = 3


def main():
    with open("eval/questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    passed = answerable = 0
    for item in questions:
        results = search(item["q"], k=K)
        found = [r["metadata"].get("procedure_number") for r in results]
        top = results[0]

        if item["expected"] is None:
            status = "NO-ANSWER"
        else:
            answerable += 1
            ok = item["expected"] in found
            passed += ok
            status = "PASS" if ok else "MISS"

        print(f"[{status}] {item['q']}")
        print(f"    expected: {item['expected']}   top {K}: {found}   best distance: {top['distance']}")

    print(f"\n{passed}/{answerable} answerable questions found the right procedure in the top {K}")


if __name__ == "__main__":
    main()