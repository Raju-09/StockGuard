"""
scripts/execute_notebooks.py — Run all notebooks and save pre-rendered outputs.
"""

from pathlib import Path
import nbformat
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"

notebook_files = [
    "01_data_collection.ipynb",
    "02_eda.ipynb",
    "03_baseline.ipynb",
    "04_model_comparison.ipynb",
]

for nb_name in notebook_files:
    nb_path = NOTEBOOKS_DIR / nb_name
    print(f"Executing {nb_name}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"  ✓ {nb_name} pre-rendered and saved.")

print("All notebooks successfully executed with outputs!")
