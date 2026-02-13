all: models/gemma-3-4b-it/model.pte models/gemma-3-4b-it/tokenizer.json

models/gemma-3-4b-it/model.pte:
	./.venv/bin/optimum-cli export executorch -m google/gemma-3-4b-it -o models/gemma-3-4b-it \
		--task image-text-to-text --recipe cuda --device cuda --dtype bfloat16

models/gemma-3-4b-it/tokenizer.json:
	./.venv/bin/hf download --local-dir models/gemma-3-4b-it google/gemma-3-4b-it tokenizer.json
