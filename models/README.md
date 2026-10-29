# Models

Trained weights are **not stored in this repo**. They live on Hugging Face:

**https://huggingface.co/PourSightAI/CosmosReason3Hackathon**

Each training run is uploaded to its own folder, `adapters/<run-name>/`. The file that holds what the model learned is `adapter_model.safetensors` (a LoRA adapter on top of Cosmos Reason).

## How the pieces fit together

```
Isaac Sim job (digital twin)  ->  results folder: images + labels.csv
                                          |
                                          v
              train_cosmos_lora.py  (reads those files, trains a LoRA adapter)
                                          |
                                          v
                  Hugging Face repo: adapters/<run-name>/
```

The training script does not talk to Isaac Sim. It only reads the files that the Isaac Sim job leaves behind. Train on **digital-twin data only**. The real camera data is for testing.

## Base model (on the Spark)

Cosmos Reason is already downloaded on the team Spark at `/workspace/models/cosmos-reason` (source: `nvidia/Cosmos-Reason1-7B`). Do not download it again.

## Step 1: Start the training container

On the Spark, inside tmux:

```bash
docker run --rm -it --gpus all -v /workspace:/workspace nvcr.io/nvidia/pytorch:25.01-py3 bash
pip install transformers==4.51.3 qwen-vl-utils peft accelerate pillow
```

The container deletes itself on exit, so the `pip install` has to be repeated each time. Your files in `/workspace` are kept.

## Step 2: Set the token (inside the container)

```bash
export HF_TOKEN=hf_yourTokenHere
```

Use your real token in place of the placeholder, and type it in the terminal only. It lasts for that terminal session, so repeat it if you reopen the container.

## Step 3: Check the labels file

The labels file (`labels.csv`) needs these columns, or whatever columns the team agrees the data needs:

| Column | Meaning |
|---|---|
| `image` | image file name, relative to the folder you pass as `--img-root` |
| `recipe_id` | recipe this vial was dosed with (5 to 14) |
| `prev_recipe_id` | recipe that ran just before it (blank if none) |
| `vial_in_batch` | 1, 2, or 3 |
| `contaminated` | 1 if the image shows carryover discoloration, otherwise 0 |
| `observed_color` | optional, for example `green` |

Check that the script reads your file correctly. This loads no model and uses no GPU, and it takes seconds:

```bash
python /workspace/train_cosmos_lora.py \
  --labels /workspace/results/job-abc/labels.csv \
  --img-root /workspace/results/job-abc \
  --dry-run
```

It prints one contaminated and one clean example prompt and answer, and ends with `Dry run OK: N rows read`. Read the examples and make sure they make sense. If it fails with a `KeyError`, the column names do not match, so rename the columns in the file or tell the script owner.

The dry run does **not** check that the image files exist. Look at a few yourself:

```bash
ls /workspace/results/job-abc | head
```

Also open 10 or so images and confirm the labels match what you see, before starting a long run. If every row says `contaminated` is 0, the model will learn to say "matches" every time.

## Step 4: Train and upload

Replace `job-abc` with the real job folder and `run-01` with your run name:

```bash
python /workspace/train_cosmos_lora.py \
  --labels /workspace/results/job-abc/labels.csv \
  --img-root /workspace/results/job-abc \
  --holdout 7 \
  --out /workspace/models/run-01 \
  --push
```

What the options do:

| Option | Meaning |
|---|---|
| `--labels` | the labels file |
| `--img-root` | folder that the `image` column is relative to |
| `--holdout 7` | leave every Recipe 7 vial out of training (comma-separated for more, for example `7,9`) |
| `--out` | where the adapter is saved on the Spark (use a new name each run) |
| `--push` | after training, fix the base-model id and upload to Hugging Face |
| `--epochs`, `--lr`, `--accum` | optional training settings (defaults: 2, 1e-4, 8) |

With `--push`, the script saves the adapter, replaces the local base-model path with `nvidia/Cosmos-Reason1-7B` in the adapter files, and uploads the folder to `adapters/<run-name>` in the Hugging Face repo. If the token is missing or the upload fails, the adapter is still saved on the Spark under `--out`.

## Step 5: Check the result

1. The terminal prints `uploaded to PourSightAI/CosmosReason3Hackathon in folder adapters/run-01`.
2. Open the repo's **Files and versions** tab and look for `adapters/run-01/adapter_model.safetensors`.
3. Write the run down in the table below.

## Run log

| Run name | Date | Job / data used | Holdout recipe(s) | Notes |
|---|---|---|---|---|
| | | | | |


## Everything is deleted when the event ends

The Spark workspace is wiped after the event. Adapters are safe on Hugging Face and code is safe in GitHub. The generated images and labels are only on the Spark, so copy them off or upload them somewhere if you want to keep them.
