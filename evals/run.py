"""Run a tiny behavioral evaluation against a running API."""

import json
import os
import sys
import unicodedata
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

CASES_PATH = Path(__file__).with_name("cases.json")
API_URL = os.getenv("EVAL_API_URL", "http://localhost:8000").rstrip("/")


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def ask(question: str) -> dict:
    body = json.dumps({"question": question}).encode()
    request = Request(
        f"{API_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - URL is operator-controlled
        return json.load(response)


def case_passes(case: dict, response: dict) -> bool:
    answer = normalize(response["answer"])
    exact = case.get("exact_answer")
    terms_match = all(normalize(term) in answer for term in case.get("expected_terms", []))
    answer_matches = normalize(exact) == answer if exact else terms_match
    sources_match = bool(response["sources"]) is case["expects_sources"]
    return answer_matches and sources_match


def main() -> int:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    passed = 0
    for case in cases:
        response = ask(case["question"])
        success = case_passes(case, response)
        passed += success
        print(f"{'PASS' if success else 'FAIL'}  {case['question']}")
        if not success:
            print(f"      {response}")
    print(f"\n{passed}/{len(cases)} cases passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except URLError as error:
        print(f"Could not reach {API_URL}: {error}", file=sys.stderr)
        sys.exit(2)
