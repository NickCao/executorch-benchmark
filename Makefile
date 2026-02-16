.PHONY: all serve bench

all: models/qwen3-0.6b/model.pte models/qwen3-0.6b/tokenizer.json

models/qwen3-0.6b/model.pte:
	./.venv/bin/optimum-cli export executorch -m Qwen/Qwen3-0.6B -o models/qwen3-0.6b \
		--task text-to-text --recipe xnnpack --device cpu --dtype bfloat16 --max_seq_len 2048

models/qwen3-0.6b/tokenizer.json:
	./.venv/bin/hf download --local-dir models/qwen3-0.6b Qwen/Qwen3-0.6B tokenizer.json

serve:
	./.venv/bin/fastapi dev src/executorch_benchmark

serve-vllm:
	./.venv/bin/vllm serve Qwen/Qwen3-0.6B

bench:
	./.venv/bin/vllm bench serve --max-concurrency=1 --random-input-len=512 --num-prompts=10
