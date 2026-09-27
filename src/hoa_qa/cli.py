"""Command-line entry point; Q&A commands will follow in the QA core unit."""

import argparse
from importlib.metadata import version


def main() -> None:
    parser = argparse.ArgumentParser(description="Lakewood Creek HOA Q&A")
    parser.add_argument("--version", action="version", version=version("hoa-qa"))
    parser.parse_args()
