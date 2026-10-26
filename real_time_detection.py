import subprocess
import sys
from pathlib import Path


VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    sys.exit(subprocess.call([str(VENV_PYTHON), __file__, *sys.argv[1:]]))

from scripts.real_time_detection import run_real_time_detection


if __name__ == "__main__":
    run_real_time_detection()
