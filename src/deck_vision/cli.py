from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assets import AssetStore
from .errors import DeckVisionError
from .recognize import recognize_deck


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "recognize":
            output = recognize_deck(args.image, endpoint=args.endpoint, cache_dir=args.cache_dir)
            payload = output.to_dict()
            if args.json:
                print(json.dumps(payload, ensure_ascii=False))
            else:
                print(payload["code"])
            return 0
        if args.command == "assets":
            store = AssetStore(endpoint=args.endpoint, cache_dir=args.cache_dir)
            if args.assets_command == "refresh":
                cards = store.refresh()
                print(json.dumps({"ok": True, "cards": len(cards), **store.info()}, ensure_ascii=False))
                return 0
            if args.assets_command == "info":
                print(json.dumps(store.info(), ensure_ascii=False))
                return 0
    except DeckVisionError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return 2
    parser.print_help(sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deck-vision")
    parser.add_argument("--endpoint", default=None, help="Override ASSETS_API_ENDPOINT.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override the asset cache directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recognize = subparsers.add_parser("recognize", help="Recognize a deck image.")
    recognize.add_argument("image", type=Path)
    recognize.add_argument("--json", action="store_true", help="Print the full JSON output.")

    assets = subparsers.add_parser("assets", help="Manage cached card assets.")
    asset_subparsers = assets.add_subparsers(dest="assets_command", required=True)
    asset_subparsers.add_parser("refresh", help="Fetch metadata, card faces, and template fingerprints.")
    asset_subparsers.add_parser("info", help="Print cache status.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
