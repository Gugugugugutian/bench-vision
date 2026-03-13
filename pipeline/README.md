# Bench-Vision Pipeline (00~05)

## Steps

1. `step1_prepare_data.py`: Prepare dataset JSONL files into `00_data/`.
2. `step2_generate_response.py`: Generate model responses into `01_response/<model_name>/`.
3. `step3_evaluate_str.py`: String-matching evaluation into `02_str_evaluation/<model_name>/`.
4. `step4_evaluate_llm.py`: vLLM-based LLM evaluation into `03_llm_evaluation/<model_name>/`.
5. `step5_score.py`: Merge two evaluations and output final scores into `05_score/`, with optional conflict calibration cache in `04_calibration/`.

## Quick Usage

```bash
# 1) prepare data (copy existing root jsonl)
python pipeline/step1_prepare_data.py --mode copy --dataset all

# 2) generate response (submit rjob)
bash rjob_step2.sh all Llama-3.2-11B-Vision-Instruct

# 3) string evaluation
python pipeline/step3_evaluate_str.py \
  --model_name Llama-3.2-11B-Vision-Instruct \
  --dataset all

# 4) llm evaluation via vllm (submit rjob)
bash rjob_step4.sh all Llama-3.2-11B-Vision-Instruct ../models/gpt-oss-20b

# 5a) final score, OR mode
python pipeline/step5_score.py \
  --model_name Llama-3.2-11B-Vision-Instruct \
  --mode or \
  --dataset all

# 5b) final score, calibrated mode
export OPENAI_API_KEY=...
python pipeline/step5_score.py \
  --model_name Llama-3.2-11B-Vision-Instruct \
  --mode calibrated \
  --verifier_model gpt-4.1-mini \
  --dataset all
```

## Notes

- Calibrated mode only calls external API for conflicts where string and llm scores differ.
- Calibration cache is stored per tested model in `04_calibration/<model_name>.csv`.
- Cache key uses `dataset + qid`, so repeated runs can reuse verified results.
- In this repo environment, `step2` and `step4` should be submitted via `rjob`.
