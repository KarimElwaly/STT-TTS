"""Print the language IDs and voice-design vocabulary OmniVoice accepts.

`instruct` is NOT free-form prose: it must be a comma-separated list drawn
from a fixed vocabulary, or generation raises ValueError.
"""

from __future__ import annotations

import sys

from omnivoice.models.omnivoice import _INSTRUCT_MUTUALLY_EXCLUSIVE, _INSTRUCT_VALID_EN
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID


def main() -> None:
    print(f"languages supported: {len(LANG_IDS)}")
    query = sys.argv[1].lower() if len(sys.argv) > 1 else "arabic"
    matches = {name: lid for name, lid in LANG_NAME_TO_ID.items() if query in name}
    print(f"language names matching {query!r}:")
    for name, lid in sorted(matches.items()):
        print(f"  {name:32} -> {lid}")
    print(f"\n'ar' is a valid id: {'ar' in LANG_IDS}")
    print(f"'en' is a valid id: {'en' in LANG_IDS}")

    print("\nvalid instruct items (English):")
    for item in sorted(_INSTRUCT_VALID_EN):
        print(f"  {item}")
    print("\nmutually exclusive groups:")
    for group in _INSTRUCT_MUTUALLY_EXCLUSIVE:
        print(f"  {sorted(group)}")


if __name__ == "__main__":
    main()
