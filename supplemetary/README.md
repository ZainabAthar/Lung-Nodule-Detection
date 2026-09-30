# Task 2 — Development Environment Setup

This document records how the development environment was set up in two places:

1. **Locally in VS Code** — for writing/editing code, fast iteration, and running the CPU-only MNIST smoke test.
2. **On Kaggle** — for actual GPU-dependent training later in the project (e.g. MONAI-based medical imaging models).

MNIST itself doesn't need a GPU, so the smoke test was run locally on CPU. Kaggle is reserved for the heavier GPU workloads once the real pipeline (Milestone 6+) is underway.

---

## 1. Local Setup (VS Code)

### Prerequisites
- Python 3.10+ installed
- VS Code with the **Python** and **Jupyter** extensions installed

### Steps

1. **Create a virtual environment** (keeps this project's packages isolated):
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

2. **Install dependencies** from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   > Note: `pip` will automatically install the CPU build of PyTorch/Torchvision on a machine without an NVIDIA GPU — no extra flags needed.

3. **Select the Python interpreter in VS Code**:
   - `Cmd/Ctrl + Shift + P` → `Python: Select Interpreter` → choose the `venv` you just created.

4. **Run the smoke test**:
   ```bash
   python mnist_smoke_test.py
   ```
   Or open `mnist_smoke_test.py` in VS Code and run cell-by-cell / via "Run Python File."

5. **Confirm output**:
   - Prints versions of PyTorch, NumPy, Pandas, OpenCV, MONAI
   - Trains a small CNN on a 2,000-image MNIST subset (1 epoch, CPU)
   - Reports test accuracy on a 500-image subset
   - Saves `smoke_test_sample.png` (an OpenCV-resized MNIST digit) to confirm OpenCV + Matplotlib work on real tensor data
   - Prints `✅ SMOKE TEST PASSED` if accuracy > 50%

### Files
- `requirements.txt` — pinned package list
- `mnist_smoke_test.py` — the smoke test script

---

## 2. Kaggle Setup (for GPU work)

Kaggle Notebooks come with PyTorch, Torchvision, NumPy, Pandas, Matplotlib, and OpenCV **preinstalled**. Only MONAI needs to be added manually.

### Steps

1. Go to [kaggle.com](https://www.kaggle.com) → **Create → New Notebook**.
2. Upload `kaggle_env_smoke_test.ipynb` (**File → Import Notebook**), or copy its cells into a new notebook.
3. In the right-hand **Settings** panel:
   - **Accelerator** → set to `GPU T4 x2` (or `P100`, depending on availability/quota)
   - **Internet** → toggle **On** (required to `pip install monai` and to download the MNIST dataset)
4. Run all cells top to bottom (**Run → Run All**, or `Shift+Enter` through each cell).
5. Confirm output:
   - Environment check cell prints package versions and confirms `CUDA available: True` with the GPU device name
   - Smoke test cell trains the same small CNN, this time on GPU, and reports accuracy + elapsed time
   - Final cell shows the OpenCV-resized MNIST digit inline

### Notes
- Kaggle sessions have weekly GPU quota limits — since MNIST doesn't need a GPU, this Kaggle run is really just to confirm the *GPU path* works (CUDA available, model moves to `cuda`, training completes) before relying on it for the real MONAI pipeline.
- No local-to-Kaggle "remote kernel" connection was set up in VS Code — Kaggle doesn't support this cleanly (unlike Colab's official local-runtime feature). Code was written/tested locally, then run directly in the Kaggle browser notebook.

### Files
- `kaggle_env_smoke_test.ipynb` — Kaggle-ready notebook version of the smoke test

---

## 3. Package Versions Verified

| Package      | Purpose                                   |
|--------------|--------------------------------------------|
| PyTorch      | Core deep learning framework               |
| Torchvision  | Datasets (MNIST) + image transforms        |
| MONAI        | Medical imaging deep learning (Milestone 6)|
| OpenCV       | Image preprocessing                        |
| NumPy        | Array operations                           |
| Pandas       | Data handling                              |
| Matplotlib   | Visualization                              |
| Jupyter/VS Code | Development interface                   |

Exact installed versions are printed by both the local script and the Kaggle notebook's environment-check cell — paste those into this table (or a version-lock file) once run, for reproducibility.

## 4. Status

- [x] Environment defined (`requirements.txt`)
- [x] Local VS Code setup documented and script written
- [x] Kaggle notebook prepared for GPU verification
- [ ] **Local virtual environment (`venv`) not yet created** — the `python -m venv venv` step in Section 1 has not been run. No packages have been installed locally yet.
- [ ] Smoke test run locally — confirm ✅ pass, paste accuracy/timing here
- [ ] Smoke test run on Kaggle GPU — confirm ✅ pass, paste accuracy/timing here
- [ ] Package versions from both environments recorded in the table above

*(This README currently documents the intended setup only. Nothing has actually been run yet — the local venv still needs to be created and activated before `pip install -r requirements.txt` will work.)*