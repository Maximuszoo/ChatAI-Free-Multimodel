#!/usr/bin/env python3
"""CAFM — ChatAI-Free-Multimodel entry point."""

from cafm.cli import main_loop


def main() -> None:
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[Interrupted] Goodbye!")


if __name__ == "__main__":
    main()
