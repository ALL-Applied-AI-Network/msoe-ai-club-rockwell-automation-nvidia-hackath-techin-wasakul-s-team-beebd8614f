# PourSightAI — Project Runbook

**Project:** PourSightAI  
**Purpose:** Detect potential liquid cross-contamination in vials on the CSI production line.  
**Environment:** NVIDIA BREV, Ubuntu 24.04, NVIDIA Isaac Sim / Omniverse, Cosmos 3 Nano Reasoner  
**Status:** Development and integration

---

## 1. Project Overview

PourSightAI is an independent, read-only inspection system designed to identify potential cross-contamination between liquid vials.

The system compares a target vial's observed color against:
- Its expected recipe color.
- Other vials in the same production sequence.
- Relevant production and recipe information.

The system should observe and report results without controlling or modifying the physical production line.

## 2. System Environment

| Component | Configuration |
|---|---|
| Operating environment | BREV Ubuntu 24.04.3 |
| Project directory | `/workspace/twin_data` |
| Digital twin | `/workspace/csi_digital_twin_v1` |
| Cosmos environment | `/workspace/cosmos3-env` |
| Python interpreter | `/workspace/cosmos3-env/bin/python` |
| GPU | NVIDIA RTX PRO 6000 Blackwell |
| AI model | Cosmos 3 Nano Reasoner |
| Hugging Face account | `PourSightAI` |
| Dataset repository | `PourSightAI/Training_Data` |

Always use the explicit Python interpreter above when running Cosmos-related scripts.

## 3. Environment Startup

Open the BREV environment through VS Code or its terminal.

Activate the Cosmos environment:

```bash
source /workspace/cosmos3-env/bin/activate
```

Verify Python:

```bash
/workspace/cosmos3-env/bin/python --version
```

Verify GPU availability:

```bash
/workspace/cosmos3-env/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected result:
- CUDA availability: `True`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition

## 4. Dataset Management

### Dataset location

```text
/workspace/twin_data/
```

The project contains generated vial images, labels, metadata, and supporting scripts.

One established dataset:

```text
/workspace/twin_data/full_dataset_5000/
├── images/
└── labels.csv
```

Note: The folder name does not indicate the exact image count. The dataset previously validated contained 4,803 images.

### Dataset validation

Check the image count:

```bash
find /workspace/twin_data/full_dataset_5000/images \
-type f | wc -l
```

Inspect CSV headers:

```bash
head -n 5 /workspace/twin_data/full_dataset_5000/labels.csv
```

Check that image and label counts match before training.

### Dataset labels

The dataset includes information such as:

- Recipe ID
- Previous recipe ID
- Vial index
- Contamination label
- Expected and observed color
- RGB measurements
- Carryover information
- Fill-level information

Contamination labels:
- `0` — Clean
- `1` — Contaminated

## 5. Hugging Face Dataset Upload

The existing dataset repository is:

`PourSightAI/Training_Data`

To upload a dataset directory:

```bash
HF_XET_HIGH_PERFORMANCE=1 \
/workspace/cosmos3-env/bin/hf upload \
PourSightAI/Training_Data \
/workspace/twin_data/full_dataset_5000 \
full_dataset_5000 \
--type dataset \
--commit-message "Upload vial dataset"
```

Verify the upload through the Hugging Face repository website.

Do not create another repository unless the project team explicitly decides to do so.

## 6. Cosmos 3 Nano

### Model information

- Model: Cosmos 3 Nano Reasoner
- Repository: `PourSightAI/CosmosReason3Hackathon`
- Model directory: `cosmos3-nano/`

The model has previously been loaded successfully using the Transformers-based workflow.

### Authentication check

```bash
/workspace/cosmos3-env/bin/hf auth whoami
```

### Important considerations

Cosmos should assist with visual reasoning, not serve as the only contamination detector.

The inspection pipeline should combine:
1. Image acquisition.
2. Vial detection.
3. Color measurement.
4. Comparison against expected and neighboring vial colors.
5. AI-assisted interpretation.
6. Inspection reporting.

Any model changes or LoRA training should be validated against held-out data before being used in the physical demonstration.

## 7. Digital Twin

Project directory:

```text
/workspace/csi_digital_twin_v1
```

The digital twin is used to generate simulated production data and vial images.

Use generated data to:
- Test contamination scenarios.
- Validate image-processing logic.
- Generate training examples.
- Evaluate model performance.

The digital twin is a development and testing resource. It is not a substitute for validating the system against real CSI production-line images.

## 8. Live CSI Demonstration

### Objective

Demonstrate PourSightAI detecting potential cross-contamination on the physical CSI production line.

### Expected operating workflow

1. CSI production line operates normally.
2. A camera captures images of the vials.
3. PourSightAI receives the camera images on its inspection computer.
4. The program identifies the vials.
5. The program compares observed colors against expected recipe colors and surrounding vials.
6. Cosmos 3 Nano assists with visual interpretation.
7. PourSightAI reports the inspection result.

### Hardware and access requirements

These items still need confirmation:

- [ ] Identify the existing CSI camera make and model.
- [ ] Confirm how the inspection computer receives camera images.
- [ ] Confirm camera feed permissions and access.
- [ ] Determine camera resolution and frame rate.
- [ ] Identify how recipe and production-sequence information will be obtained.
- [ ] Confirm the physical computer that will run PourSightAI.
- [ ] Test image acquisition without interfering with production.

**Important:** The inspection system must remain read-only. It must not send commands to production equipment.

## 9. Troubleshooting

| Problem | Recommended action |
|---|---|
| Wrong Python interpreter | Use `/workspace/cosmos3-env/bin/python` explicitly. |
| CUDA unavailable | Check GPU visibility and PyTorch installation. |
| Hugging Face authentication failure | Run `hf auth whoami` and verify login. |
| Dataset image count mismatch | Compare image filenames against CSV entries. |
| Cosmos model fails to load | Check model repository, checkpoint path, and environment. |
| Python code produces Bash errors | Save Python code in a `.py` file and execute it with Python. |
| CSI camera unavailable | Verify camera permissions, connection, and image-feed method with CSI staff. |

## 10. GitHub Version Control

Before making significant changes:

```bash
git status
git pull
```

After completing a working change:

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

- [ ] BREV environment accessible.
- [ ] Correct Python environment activated.
- [ ] GPU available.
- [ ] Cosmos 3 Nano loads successfully.
- [ ] Dataset validated.
- [ ] Image acquisition tested.
- [ ] Camera access confirmed.
- [ ] Vial detection tested.
- [ ] Expected recipe information available.
- [ ] Cross-vial comparison tested.
- [ ] Clean and contaminated examples tested.
- [ ] Results can be displayed to the team.
- [ ] Inspection system does not control production equipment.

## 12. Operational Principle

PourSightAI is an independent inspection system. Its purpose is to observe, analyze, and report potential contamination while allowing the CSI production line to operate normally.

When confidence is insufficient, report an uncertain result rather than declaring a vial contaminated or clean without sufficient evidence.
