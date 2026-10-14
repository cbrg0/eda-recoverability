"""Allow ``python -m edarecover`` as well as the installed entry point."""

from edarecover.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
