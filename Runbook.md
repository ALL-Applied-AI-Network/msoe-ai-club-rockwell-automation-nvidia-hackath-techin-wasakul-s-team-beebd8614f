# PourSightAI — Project Runbook

**Project:** PourSightAI  
**Purpose:** Detect likely liquid carryover and contamination in vials using the repository's advisory inspection logic.  
**Environment:** BREV / Ubuntu, Python, project repository checkout, and optional model environment  
**Status:** Development and validation

---

## 1. Project Overview

PourSightAI is a read-only inspection and advisory system designed to identify likely cross-contamination between liquid vials. The software analyzes vial appearance, recipe context, and recent production history without sending commands to the physical line.

This repository is organized around the following key pieces:

- `src/line_controller.py` — primary inspection and reporting logic
- `tests/test_line_controller.py` — expected behavior checks for mock runtime flow
- `twin_data/` — digital-twin generation and sample dataset utilities
- `models/` — model weight notes and Hugging Face references
- `requirements.txt` — Python dependencies for the project

The system should observe, assess, and report results without controlling equipment.

## 2. Repository Layout

Expected repository root:

```text
/workspace
├── README.md
├── requirements.txt
├── src/
│   ├── line_controller.py
│   ├── test_system.py
│   └── train_cosmos_lora.py
├── tests/
│   ├── test_line_controller.py
│   └── mock_production_line_demo.py
├── twin_data/
│   ├── README.md
│   └── ...
├── models/
│   └── README.md
└── ...
```

If the repository is checked out somewhere else, replace `/workspace` below with that path.

## 3. Environment Setup

Use the Python interpreter available in the environment where the repository is running. Do not assume a conda or venv name is present.

Verify Python:

```bash
python --version
```

Install project dependencies if needed:

```bash
cd /workspace
python -m pip install -r requirements.txt
```

If the project needs to import from `src`, either run from the repo root or add it to `PYTHONPATH`:

```bash
cd /workspace
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## 4. Validation Commands

Run the project’s unit tests from the repository root:

```bash
cd /workspace
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m pytest -q tests/test_line_controller.py
```

Expected outcome: the mock controller tests pass and the API contract remains stable.

Quick smoke check without the full model stack:

```bash
cd /workspace
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python src/line_controller.py --mock --predictive
```

Important: the tests import `line_controller` from `src/`, not from the repo root directly.

## 5. Dataset and Twin Data

This repository includes the digital-twin/data-generation pipeline under `twin_data/`.

Useful commands:

```bash
cd /workspace
ls twin_data
```

The project documentation explains that the digital twin is used to generate synthetic vial data and labels. The real trained model weights are expected to live on Hugging Face rather than in GitHub.

Important notes:

- Do not commit large datasets or model weights to GitHub.
- Keep generated images/labels local unless intentionally uploaded.
- Use the data folders and scripts in `twin_data/` for testing and validation.

## 6. Model / Hugging Face Guidance

The repo notes that trained weights are stored on Hugging Face, not in the repository itself.

Relevant references:

- `models/README.md`
- `src/line_controller.py` constants such as `HF_REPO`, `HF_BASE_ID`, and `HF_ADAPTER_SUBFOLDER`

The project expects the model repo and adapter folder to be resolved at runtime. The code is designed to fail fast if a required adapter or config is missing rather than silently falling back to an untrained model.

## 7. Production-Line Demonstration Constraints

This project is advisory-only. It must remain read-only with respect to manufacturing equipment.

The inspection system should:

1. Receive vial images or image sequences.
2. Use recipe and batch context.
3. Compare current appearance to expected color and recent history.
4. Produce an operator-facing advisory report.
5. Never send control commands to production equipment.

The report layer is intended for operator review and dashboards, not machine actuation.

## 8. Operational Principles

- The inspection system is not a PLC or machine controller.
- If a result is ambiguous, report uncertainty instead of making a definitive claim.
- The software may recommend operational action, but it does not directly stop or modify the line.
- Model and adapter changes require validation before being trusted in active demonstrations.

## 9. Troubleshooting

| Problem | Recommended action |
|---|---|
| `ModuleNotFoundError: No module named 'line_controller'` | Run from the repo root and set `PYTHONPATH=$PWD/src` |
| pytest import fails | Use `python -m pytest -q tests/test_line_controller.py` from `/workspace` |
| A function returns the wrong tuple shape | Confirm it returns `(status, report)` as expected by tests |
| Model fails to load | Check adapter path, model repo, and local environment variables |
| Missing dataset files | Validate the `twin_data/` folder and any generated output paths |
| CUDA or ML tooling missing | Install project dependencies and validate the active Python environment |

## 10. GitHub Workflow

Before making significant changes:

```bash
git status
git pull
```

After a working change:

```bash
git add .
git commit -m "Describe the changes"
git push
```

To inspect previous commits:

```bash
git log --oneline
```

To safely reverse a pushed commit:

```bash
git revert COMMIT_ID
git push
```

Avoid `git reset --hard` on shared branches because it can discard work and rewrite local history.

## 11. Pre-Demonstration Checklist

- [ ] Repository is checked out in the expected workspace.
- [ ] Python environment is active and dependencies are installed.
- [ ] `PYTHONPATH` includes `src/` when needed.
- [ ] Unit tests pass.
- [ ] Model or adapter is available if running the real inference path.
- [ ] Dataset is validated and paths are correct.
- [ ] The inspection report is advisory-only.
- [ ] No control commands are sent to the production line.
- [ ] Results can be displayed to the team without modifying equipment.

## 12. Final Notes

The project is best treated as an AI-assisted inspection and reporting pipeline. It is not a machine-control layer, and it should remain so during the hackathon and final demonstration.

If anything in this runbook conflicts with the repository itself, the repository files (`src/`, `tests/`, `twin_data/`, and `models/`) are the source of truth.
