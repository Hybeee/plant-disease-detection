from __future__ import annotations

from pathlib import Path


def remap_label_ids(labels_dir: Path, from_id: str = "16", to_id: str = "15") -> tuple[int, int]:
    files_changed = 0
    labels_changed = 0

    for label_file in labels_dir.rglob("*.txt"):
        lines = label_file.read_text(encoding="utf-8").splitlines()
        file_changed = False
        updated_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                updated_lines.append(line)
                continue

            parts = stripped.split()
            if parts and parts[0] == from_id:
                parts[0] = to_id
                labels_changed += 1
                file_changed = True

            updated_lines.append(" ".join(parts))

        if file_changed:
            label_file.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
            files_changed += 1

    return files_changed, labels_changed


def main() -> None:
    dataset_dir = Path("dataset")

    total_files_changed = 0
    total_labels_changed = 0

    for split in ("train", "valid", "test"):
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            continue

        files_changed, labels_changed = remap_label_ids(labels_dir)
        total_files_changed += files_changed
        total_labels_changed += labels_changed
        print(
            f"{split}: changed {labels_changed} labels in {files_changed} files"
        )

    print(
        f"Total changed labels: {total_labels_changed}, files: {total_files_changed}"
    )


if __name__ == "__main__":
    main()
