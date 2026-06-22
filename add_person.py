# watchlist/add_person.py
# Simple CLI to upload a wanted person's photo to the watchlist
#
# Usage:
#   python watchlist/add_person.py
#
# Or directly:
#   python watchlist/add_person.py --photo path/to/photo.jpg --name "John Doe" --case "CASE-2024-001"

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_watchlist import FaceWatchlist


def main():
    parser = argparse.ArgumentParser(description="Add a wanted person to the watchlist")
    parser.add_argument("--photo", type=str, help="Path to the person's photo")
    parser.add_argument("--name",  type=str, help="Person's name or alias")
    parser.add_argument("--case",  type=str, default="", help="Police case ID (optional)")
    args = parser.parse_args()

    watchlist = FaceWatchlist()

    # Interactive mode if no args given
    if not args.photo or not args.name:
        print("\n=== Add Wanted Person to Watchlist ===\n")
        args.photo = input("Photo path (drag & drop file here): ").strip().strip("'\"")
        args.name  = input("Person's name or alias            : ").strip()
        args.case  = input("Case ID (press Enter to skip)     : ").strip()

    if not os.path.exists(args.photo):
        print(f"[ERROR] File not found: {args.photo}")
        sys.exit(1)

    print(f"\nProcessing photo: {args.photo}")
    result = watchlist.add_person(args.photo, args.name, args.case)

    if result["success"]:
        print(f"[OK] {result['message']}")
        print(f"\nWatchlist now has {len(watchlist.watchlist)} person(s):")
        for p in watchlist.list_watchlist():
            print(f"  - {p['name']}  (Case: {p['case_id'] or 'N/A'})  Added: {p['added_at']}")
    else:
        print(f"[FAILED] {result['message']}")


if __name__ == "__main__":
    main()
