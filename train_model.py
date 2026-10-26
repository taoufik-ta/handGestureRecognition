import subprocess
import sys
from pathlib import Path


VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    sys.exit(subprocess.call([str(VENV_PYTHON), __file__, *sys.argv[1:]]))

from scripts.train_model import train_model


if __name__ == "__main__":
    train_model()
