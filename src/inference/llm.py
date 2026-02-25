from llama_cpp import Llama
from config.settings import (
    LLM_MODEL_PATH,
    LLM_CONTEXT_SIZE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_THREADS,
)


class LLMEngine:
    def __init__(self):
        self.model = None

    def load(self):
        self.model = Llama(
            model_path=str(LLM_MODEL_PATH),
            n_ctx=LLM_CONTEXT_SIZE,
            n_threads=LLM_THREADS,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        stop: list[str] | None = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )
        return response["choices"][0]["text"].strip()

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        stop: list[str] | None = None,
    ):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for token in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            stream=True,
        ):
            yield token["choices"][0]["text"]
