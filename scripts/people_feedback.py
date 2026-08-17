"""Apply manual person-group corrections to a saved people index."""

import argparse
import json

from memora.clustering.people import apply_feedback, load_people_index, save_people_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply person merge/remove feedback")
    parser.add_argument("--people-path", default="data/people.json")
    parser.add_argument(
        "--merge",
        nargs=2,
        type=int,
        action="append",
        metavar=("PERSON_A", "PERSON_B"),
        help="Merge two person IDs; repeat for multiple merges",
    )
    parser.add_argument(
        "--remove",
        nargs=2,
        action="append",
        metavar=("PERSON_ID", "PHOTO_ID"),
        help="Remove a photo from a person group; repeat as needed",
    )
    args = parser.parse_args()
    if not args.merge and not args.remove:
        parser.error("provide --merge and/or --remove")

    removals = [
        {"person_id": int(person_id), "photo_id": photo_id}
        for person_id, photo_id in (args.remove or [])
    ]
    people = load_people_index(args.people_path)
    people = apply_feedback(people, merges=args.merge, removed_photos=removals)
    save_people_index(people, args.people_path)
    print(
        json.dumps(
            {
                "group_count": len(people.groups),
                "groups": [
                    {
                        "person_id": group.id,
                        "photo_count": len(group.photo_ids),
                        "removed_photo_ids": group.removed_photo_ids,
                    }
                    for group in people.groups
                ],
                "people_path": args.people_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
