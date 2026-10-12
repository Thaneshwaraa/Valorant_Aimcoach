"""Central configuration.

Every tunable knob in the system lives here and is sourced from `.env`.
Nothing else in the codebase reads os.environ directly, which keeps the
"what can I change?" surface in exactly one file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM ---
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "llama3.2:3b-instruct-q4_K_M"
    llm_temperature: float = 0.25
    llm_max_tokens: int = 400
    # Informational: context length is fixed when Ollama loads the model and
    # cannot be set per request over the OpenAI-compatible endpoint.
    llm_num_ctx: int = 4096
    llm_timeout_s: float = 120.0

    # --- Database ---
    database_url: str = "sqlite:///./data/app.db"

    # --- RAG ---
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "valorant_kb"
    kb_dir: str = "./data/kb"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    retrieval_top_k: int = 8
    retrieval_candidates: int = 24
    hybrid_alpha: float = 0.55

    # --- Domain gate ---
    domain_gate_enabled: bool = True
    domain_gate_min_score: float = 0.28

    # --- Speech ---
    stt_model: str = "small.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    piper_exe: str = "./data/voices/piper/piper.exe"
    piper_voice: str = "./data/voices/en_US-ryan-high.onnx"

    # --- Server ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    def abs_path(self, relative: str) -> Path:
        """Resolve a configured path against the project root, not the CWD.

        Without this, launching uvicorn from another directory would silently
        create a second, empty database and vector store.
        """
        p = Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url.startswith("sqlite:///./"):
            return f"sqlite:///{self.abs_path(self.database_url[10:])}"
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
