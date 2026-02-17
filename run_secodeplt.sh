cd AIgiSE

uv run --python ../.venv/bin/python -m src.aigise.evaluations.secodeplt.vul_detection run_debug \
    --agent-id aaa \
    --max_llm_calls 75 \
    --log_level INFO \
    --start_idx 1 \
    --end_idx 2 \
    --model_name="gemini-3-pro-preview" \
    --output_dir ./evals/secodeplt/test \
    --skip_poc \
    --max_workers 1

# uv run --python ../.venv/bin/python -m src.aigise.evaluations.secodeplt.vul_detection run_debug \
#     --agent-id aaa \
#     --max_llm_calls 75 \
#     --log_level INFO \
#     --start_idx 1 \
#     --end_idx 2 \
#     --model_name="Qwen/Qwen3-8B" \
#     --output_dir ./evals/secodeplt/test \
#     --skip_poc \
#     --max_workers 1