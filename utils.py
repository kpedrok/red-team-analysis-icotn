# utils/set_root.py
import os
from pathlib import Path

def set_project_root():
    os.chdir(Path().resolve().parent)
    print(f"Working directory set to: {os.getcwd()}")
