import re
import time

from llama_cpp import Llama
from config.settings import (
    LLM_MODEL_PATH,
    LLM_CONTEXT_SIZE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_THREADS,
    LLM_BATCH_SIZE,
)
from src.logger import get_logger

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
logger = get_logger(__name__)


class LLMEngine:
    def __init__(self):
        self.model = None
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    def get_token_usage(self) -> dict:
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def load(self):
        self.model = Llama(
            model_path=str(LLM_MODEL_PATH),
            n_ctx=LLM_CONTEXT_SIZE,
            n_threads=LLM_THREADS,
            n_batch=LLM_BATCH_SIZE,
            verbose=False,
        )

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3 <think>...</think> blocks from output."""
        return _THINK_RE.sub("", text).strip()

    @staticmethod
    def _wrap_chatml(prompt: str) -> str:
        """Wrap prompt in Qwen3 ChatML format with thinking disabled."""
        return (
            "<|im_start|>system\n"
            "/no_think\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        stop: list[str] | None = None,
        max_retries: int = 2,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        qwen_stop = ["<|im_end|>", "<|endoftext|>"]
        all_stop = list(dict.fromkeys((stop or []) + qwen_stop))

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.model(
                    self._wrap_chatml(prompt),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=all_stop,
                )
                usage = response.get("usage", {})
                self._prompt_tokens += usage.get("prompt_tokens", 0)
                self._completion_tokens += usage.get("completion_tokens", 0)
                logger.info(
                    "llm_generate",
                    extra={
                        "attempt": attempt,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens_session": self.total_tokens,
                    },
                )
                text = self._strip_thinking(response["choices"][0]["text"])
                if text:
                    return text
                # Empty output — retry
                logger.warning("llm_empty_response", extra={"attempt": attempt})
            except Exception as exc:
                last_exc = exc
                logger.warning("llm_generate_error", extra={"attempt": attempt, "error": str(exc)})

            if attempt < max_retries:
                time.sleep(1)

        if last_exc:
            raise last_exc
        return ""  # all retries yielded empty text

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

        qwen_stop = ["<|im_end|>", "<|endoftext|>"]
        all_stop = list(dict.fromkeys((stop or []) + qwen_stop))

        buffer = ""
        in_think = False
        for token in self.model(
            self._wrap_chatml(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=all_stop,
            stream=True,
        ):
            buffer += token["choices"][0]["text"]
            if not in_think and "<think>" in buffer:
                in_think = True
                before = buffer[: buffer.index("<think>")]
                if before:
                    yield before
                buffer = buffer[buffer.index("<think>"):]
            if in_think:
                if "</think>" in buffer:
                    buffer = buffer[buffer.index("</think>") + len("</think>"):]
                    in_think = False
                else:
                    continue
            if not in_think and buffer:
                yield buffer
                buffer = ""
        if buffer and not in_think:
            yield buffer
