# Review notes: `train_cosmos_lora.py` (Lydia)

## ✅ Works with our data as-is
- **Columns match.** `labels.csv` from `render_plan.py` has her 6 columns exactly (`image, recipe_id, prev_recipe_id, vial_in_batch, contaminated, observed_color`). The extra columns are ignored by her `csv.DictReader`.
- **Paths.** Images are written as `images/<name>.png`, so run with `--img-root <render output folder>`.
- **Recipe table is identical** to ours (tested).
- **`build_example()` accepts every row** we generate (tested on 135 rows): contaminated rows get the carryover answer, clean rows get "Yes".
- **`prev_recipe_id`** = the recipe before the whole batch (same for vials 1-3), `0` for the first recipe of a run, which is what her prompt expects.
- **Holding out by recipe** (`--holdout`) matches the Dataset 2 page's "leave one recipe out" advice.

## ⚠️ Worth fixing (yourrrr call)
1. **Fill defects are labeled "Yes, matches."** Her prompt/answer only covers color, so an underfilled vial is trained as correct (we no longer make overfills). That teaches the model to ignore fill level, and the challenge asks for color **and** fill problems.
   - Quick fix, no code change: make the plan with `--defect-rate 0`.
   - Better: add a fill sentence to `build_example()` using our extra columns `expected_total_ml`, `actual_total_ml`, `fill_defect`.
2. **Not every first vial is contaminated.** Her `--smoke` data marks every vial 1 as contaminated. On the real line, dark→dark changes (Recipe 8→9, real vial 13) show no visible carryover. Our labels set `contaminated=1` only when the color visibly changes. That's fine for a smoke test, just FYI.
3. **Not reproducible.** `random.shuffle` is never seeded, so two runs train in a different order. Suggest `random.seed(0)` (or a `--seed` argument).

## ❓ Questions to confirm
4. **Which Cosmos model?** The default base is `nvidia/Cosmos-Reason1-7B`, loaded with `Qwen2_5_VLForConditionalGeneration`, but the Hugging Face repo is called `CosmosReason3Hackathon`. If the model in `/workspace/models/cosmos-reason` is Cosmos Reason 3, the model class/processor may need to change.
5. **Testing on real data.** `heldout.json` only holds twin rows. For a real test, use `data/real_test_labels.csv`: the 30 real labeled vials (video 5) in her exact label format. Run it from `csi-vial-fill-dataset/` with `--img-root` pointing there; image paths are `key_frames_sharp/...`.

## ℹ️ Small
- Answer masking finds the prompt length with a separate processor call. That's fine if the prompt tokens are an exact prefix of the full text (normally true). A one-time check would be to decode the unmasked tokens of one example and make sure it is only the answer.
