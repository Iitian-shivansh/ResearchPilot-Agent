"""Optional diagnostic: list the Groq models available to GROQ_API_KEY.

Makes a live network call. Run manually: python scripts/check_models.py
"""

import os
import sys

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("FAILED: GROQ_API_KEY is not set.")
        return 1

    import groq

    try:
        models = groq.Groq().models.list()
    except groq.APIError as error:
        print(f"FAILED: Could not list Groq models ({type(error).__name__}).")
        return 1
    for model_id in sorted(model.id for model in models.data):
        print(model_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
