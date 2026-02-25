from llama_cpp import Llama
from config.settings import EMBED_MODEL_PATH


class Embedder:
    def __init__(self):
        self.model = None

    def load(self):
        self.model = Llama(
            model_path=str(EMBED_MODEL_PATH),
            embedding=True,
            verbose=False,
        )

    def embed(self, text: str) -> list[float]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        embedding = self.model.embed(text)
        return embedding if isinstance(embedding, list) else list(embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
