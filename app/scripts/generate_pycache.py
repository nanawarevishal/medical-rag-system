"""Generate Python bytecode cache files (__pycache__) for the app package."""

from pathlib import Path
import compileall


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    target_dir = project_root / "app"
    success = compileall.compile_dir(
        str(target_dir),
        maxlevels=20,
        quiet=1,
        force=False,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

