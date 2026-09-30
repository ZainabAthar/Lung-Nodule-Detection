# Lung Nodule Malignancy Classification on LIDC-IDRI

This project is part of the research work on **Agentic AI–Based Medical Image Classification for Early Cancer Detection Using CT Scans**. It classifies lung nodules in CT scans from the [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) dataset as **benign or malignant**. It compares CNN and Vision Transformer baselines, then adds uncertainty quantification and hyper-parameter tuning on top of them.

The long-term goal is a multi-agent pipeline with perception, classification, reasoning, self-correction and explanation agents, aimed at reducing false negatives in early cancer detection. The notebooks here build and validate the classification and uncertainty parts of that pipeline.

---

## Pipeline at a glance

| Stage | What happens |
|---|---|
| **Labels** | Annotations for the same nodule from up to four radiologists are matched (6 mm tolerance). Nodules seen by at least 3 readers are kept. The median malignancy score is binarised, and score-3 (indeterminate) nodules are dropped. |
| **Patches** | A 48 mm physical patch is cut around each nodule and resized to 224×224. 3 axial slices are stacked as channels, with an HU window of [-1000, 400]. |
| **Splits** | Patient-level 5-fold cross-validation, so no patient appears in both train and test. 15% of each training fold is held out for validation and early stopping. |
| **Models** | ResNet-18 and ViT-tiny, both ImageNet-pretrained. |
| **Evaluation** | Pooled out-of-fold AUROC, AUPRC, accuracy, sensitivity, specificity, sensitivity at 90/95/99% specificity, Brier score and ECE, with patient-clustered bootstrap CIs (1000 resamples). |
| **Fingerprint** | Every notebook prints a hash of the labels, patient IDs, protocol constants and fold assignment. Two runs can only be compared (DeLong / McNemar) if their fingerprints match. |

Dataset used: **322 nodules from 187 patients**, of which 141 are benign and 181 malignant.

---

## Repository structure

```
Tasks/
├── Task 01/     Literature review of LIDC-IDRI work (PDF)
├── Task 04/     Baseline CNN: ResNet-18, patient-level 5-fold CV
├── Task 04-01/  Task 04 rebuilt for methodological review (audits + through-plane slice fix)
├── Task 05/     ViT-tiny on the same protocol
├── Task 05-01/  ViT-tiny rebuilt on the Task 04-01 pipeline, for a paired comparison
├── Task 06/     MC Dropout uncertainty quantification (ResNet-18 and ViT-tiny)
├── Task 10/     Frozen pretrained backbone + custom head tuned with Optuna (nested CV)
└── medical-ai-imaging.ipynb   Running notebook (environment setup, Tasks 02–06)

supplemetary/
├── README.md             Environment setup notes (local VS Code + Kaggle)
├── requirements.txt      Python dependencies
├── mnist_smoke_test.py   CPU smoke test for the environment
└── ...                   Earlier notebook versions and project notes
```

Each task folder has its own notebook. Most folders also include a write-up (`.docx` or `.pdf`) explaining the method.

---

## Results so far

Pooled out-of-fold results from the original **Task 04** and **Task 05** runs (threshold 0.5):

| Model | AUROC (95% CI) | AUPRC | Accuracy | Sensitivity | Specificity | F1 | ECE |
|---|---|---|---|---|---|---|---|
| ResNet-18 | **0.901** (0.867–0.934) | 0.923 | 0.839 | 0.851 | 0.823 | 0.856 | 0.102 |
| ViT-tiny | 0.603 (0.516–0.677) | 0.649 | 0.522 | 0.448 | 0.617 | 0.513 | 0.156 |

Full metrics are in [Tasks/Task 04/resnet18_summary.json](Tasks/Task%2004/resnet18_summary.json) and [Tasks/Task 05/vit_tiny_summary.json](Tasks/Task%2005/vit_tiny_summary.json).

> **Note:** these numbers come before the Task 04-01 review. That review found a real bug: the 3-slice input was not consistently centred on the nodule in the through-plane direction. Tasks 04-01 and 05-01 fix this and replace these results. Their result bundles are not in this repo.

---

## Running the notebooks

Training runs on **Kaggle** with a GPU (T4 or P100).

1. Create a new Kaggle notebook and import the task's `.ipynb` file.
2. Under **Settings**, set **Accelerator** to GPU and turn **Internet** on. Internet is needed to install MONAI and download LIDC-IDRI.
3. Attach any inputs the notebook asks for. For example, Tasks 05-01, 06 and 10 expect the nodule cache `.npz` built by Task 04-01. The first markdown cell of each notebook lists what it needs.
4. Choose **Run All**.

To run locally (for editing and the CPU smoke test only):

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r supplemetary/requirements.txt
python supplemetary/mnist_smoke_test.py
```

---

## Files not included

Large files are excluded by `.gitignore` and are kept locally only:

- `*.npz`: nodule patch caches (`lidc_nodules_cache*.npz`, about 25 MB each)
- `*.pt` / `*.pth`: trained model weights (for example ViT-tiny fold checkpoints)
- `*.zip`: bundled task outputs

To reproduce them, run the notebooks. Task 04-01 builds the cache from raw LIDC-IDRI, and the training notebooks save the checkpoints.

---

## Acknowledgements

- **Lead researcher:** Dr. Abdul Sattar
- **Supervisor:** Dr. Muhammad Atif Chattha (Dean IT, University of Lahore)
- **Data:** LIDC-IDRI, The Cancer Imaging Archive (TCIA). Armato III, S. G., et al. *Medical Physics* 38(2), 2011.
