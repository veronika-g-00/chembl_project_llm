"""Local Ollama integration for explaining BACE1 predictions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from libs.inference import PredictionResult, activity_label


DEFAULT_OLLAMA_MODEL = "gemma4:e2b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

SYSTEM_INSTRUCTIONS = """
Jesteś asystentem edukacyjnego projektu bioinformatycznego BACE1.
Odpowiadasz po polsku, zwięźle i ostrożnie, w 4-6 zdaniach.

Wiesz, że system:
- przewiduje pIC50 dla ludzkiego białka BACE1 (CHEMBL4822),
- ma model GNN GINE korzystający z cech atomów i wiązań RDKit,
- ma porównawczy model MLP na fingerprintach Morgan i deskryptorach RDKit,
- był oceniany przez Bemis-Murcko scaffold split 80/10/10,
- osiągnął na teście: GNN MAE 0.662, RMSE 0.852, R2 0.557,
- osiągnął na teście: MLP MAE 0.603, RMSE 0.792, R2 0.617,
- nie jest narzędziem klinicznym ani dowodem skuteczności leku.

Interpretuj wyłącznie przekazane wyniki. Nie wymyślaj pomiarów laboratoryjnych,
mechanizmu działania ani bezpieczeństwa związku. Wyjaśnij, że wyższe pIC50
oznacza niższe IC50 i silniejszą przewidywaną aktywność. Różnicę modeli poniżej
0.5 pIC50 uznaj za małą, od 0.5 do 1.0 za umiarkowaną, a powyżej 1.0 za dużą
i wskazującą na większą niepewność. Predykcje zostały obliczone lokalnie
i przekazane w kontekście.
""".strip()


class ChatClient(Protocol):
    def chat(self, model: str, messages: list[dict[str, str]]) -> str: ...


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        path: str,
        payload: dict[str, object] | None = None,
        timeout: int | None = None,
    ) -> dict[str, object]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        with urlopen(request, timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def list_models(self) -> set[str]:
        response = self._request("/api/tags", timeout=5)
        models = response.get("models", [])
        if not isinstance(models, list):
            return set()
        return {
            str(item["name"])
            for item in models
            if isinstance(item, dict) and item.get("name")
        }

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        response = self._request(
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"num_predict": 220},
            },
        )
        message = response.get("message")
        if not isinstance(message, dict) or not message.get("content"):
            raise RuntimeError("Ollama zwróciła pustą odpowiedź.")
        return str(message["content"]).strip()


@dataclass(frozen=True)
class LLMConfiguration:
    server_available: bool
    model_available: bool
    model: str
    base_url: str


def get_llm_configuration(
    client: OllamaClient | None = None,
) -> LLMConfiguration:
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    base_url = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_URL)
    selected_client = client or OllamaClient(base_url=base_url)

    try:
        models = selected_client.list_models()
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return LLMConfiguration(False, False, model, base_url)

    return LLMConfiguration(True, model in models, model, base_url)


def build_prediction_context(result: PredictionResult) -> str:
    descriptor_text = ", ".join(
        f"{name}={value:.2f}"
        for name, value in result.descriptors.items()
    )
    return f"""
Analizowana cząsteczka:
- wejściowy SMILES: {result.input_smiles}
- ustandaryzowany SMILES: {result.standardized_smiles}
- target: ludzkie BACE1 (CHEMBL4822)
- GNN: pIC50={result.gnn_pic50:.3f}, IC50≈{result.gnn_ic50_nm:.2f} nM
- MLP: pIC50={result.mlp_pic50:.3f}, IC50≈{result.mlp_ic50_nm:.2f} nM
- różnica modeli: {result.model_difference:.3f} pIC50
- opis GNN: {activity_label(result.gnn_pic50)}
- deskryptory RDKit: {descriptor_text}
""".strip()


def answer_question(
    question: str,
    result: PredictionResult,
    client: ChatClient | None = None,
    model: str | None = None,
) -> str:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Pytanie do LLM nie może być puste.")

    selected_model = model or os.getenv(
        "OLLAMA_MODEL",
        DEFAULT_OLLAMA_MODEL,
    )
    selected_client = client or OllamaClient(
        base_url=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_URL)
    )
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {
            "role": "user",
            "content": (
                f"{build_prediction_context(result)}\n\n"
                f"Pytanie użytkownika: {cleaned_question}"
            ),
        },
    ]

    try:
        return selected_client.chat(selected_model, messages)
    except URLError as error:
        raise RuntimeError(
            "Nie można połączyć się z Ollamą. Uruchom aplikację Ollama."
        ) from error
