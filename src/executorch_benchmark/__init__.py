import executorch.extension.pybindings._portable_lib  # noqa: F401
from asyncio import to_thread, Queue, QueueShutDown, create_task, Semaphore
from executorch.extension.llm.runner import MultimodalRunner, GenerationConfig
from transformers import AutoProcessor
from http import HTTPStatus
from fastapi import FastAPI, Depends, Request
from fastapi.responses import Response, JSONResponse, StreamingResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from vllm.entrypoints.utils import with_cancellation
from vllm.entrypoints.openai.engine.protocol import (
    ModelCard,
    ModelList,
    ModelPermission,
    ErrorResponse,
)


processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

runner = MultimodalRunner(
    model_path="models/gemma-3-4b-it/model.pte",
    data_path="models/gemma-3-4b-it/aoti_cuda_blob.ptd",
    tokenizer_path="models/gemma-3-4b-it/tokenizer.json",
)

sem = Semaphore(1)


app = FastAPI()


@app.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models(raw_request: Request):
    return JSONResponse(
        content=ModelList(
            data=[
                ModelCard(
                    id="google/gemma-3-4b-it",
                    max_model_len=2**16,
                    root="google/gemma-3-4b-it",
                    permission=[ModelPermission()],
                )
            ]
        ).model_dump()
    )


@app.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def create_completion(request: CompletionRequest, raw_request: Request):
    async def dummy():
        await sem.acquire()
        inputs_hf = processor(
            images=None,
            text=request.prompt,
            return_tensors="pt",
        )

        config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=0.7,
        )

        queue = Queue(1024)

        def generate():
            runner.generate_hf(
                inputs=inputs_hf,
                config=config,
                token_callback=queue.put_nowait,
            )
            queue.shutdown()
            runner.reset()

        task = create_task(to_thread(generate))

        while True:
            try:
                text = await queue.get()
                choice_data = CompletionResponseStreamChoice(index=0, text=text)
                chunk = CompletionStreamResponse(
                    model="google/gemma-3-4b-it",
                    choices=[choice_data],
                )
                response_json = chunk.model_dump_json(exclude_unset=False)
                yield f"data: {response_json}\n\n"
            except QueueShutDown:
                sem.release()
                break

        await task

    return StreamingResponse(content=dummy(), media_type="text/event-stream")
