from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    app_path = repo_root / "swaasthya_mitra" / "app.py"
    root_requirements = repo_root / "requirements.txt"

    missing = [
        str(path.relative_to(repo_root)).replace("\\", "/")
        for path in [app_path, root_requirements]
        if not path.exists()
    ]
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("Installing dependencies from root requirements.txt...")
    run([sys.executable, "-m", "pip", "install", "-r", str(root_requirements)], cwd=repo_root)

    print("Running import smoke check...")
    run(
        [
            sys.executable,
            "-c",
            "import streamlit; from dotenv import load_dotenv; print('Dependency import check: OK')",
        ],
        cwd=repo_root,
    )

    print("\nPre-deploy check passed.")
    print("You can now push and redeploy on Streamlit Cloud.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"\nPre-deploy check failed with exit code {exc.returncode}.")
        raise SystemExit(exc.returncode)
