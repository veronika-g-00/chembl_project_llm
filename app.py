"""Streamlit interface for BACE1 prediction and local LLM explanation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from libs.inference import BACE1Predictor, PredictionResult, activity_label
from libs.llm_assistant import answer_question, get_llm_configuration


EXAMPLE_SMILES = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"


st.set_page_config(
    page_title="BACE1 Predictor",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource
def load_predictor() -> BACE1Predictor:
    return BACE1Predictor()


def show_prediction(result: PredictionResult) -> None:
    image_column, result_column = st.columns([1, 1.25])

    with image_column:
        st.subheader("Struktura 2D")
        st.image(
            result.image,
            caption="Wizualizacja wygenerowana przez RDKit",
            width="stretch",
        )
        st.caption(f"Standaryzowany SMILES: `{result.standardized_smiles}`")

    with result_column:
        st.subheader("Predykcja aktywności wobec BACE1")
        gnn_column, mlp_column = st.columns(2)
        gnn_column.metric("GNN pIC50", f"{result.gnn_pic50:.3f}")
        gnn_column.metric("GNN IC50", f"{result.gnn_ic50_nm:.2f} nM")
        mlp_column.metric("MLP pIC50", f"{result.mlp_pic50:.3f}")
        mlp_column.metric("MLP IC50", f"{result.mlp_ic50_nm:.2f} nM")

        st.info(
            f"Interpretacja GNN: **{activity_label(result.gnn_pic50)}**. "
            "Wyższe pIC50 oznacza niższe IC50 i silniejszą przewidywaną "
            "aktywność."
        )
        if result.model_difference >= 1.0:
            st.warning(
                "Modele różnią się o co najmniej 1 jednostkę pIC50. "
                "Predykcję należy traktować jako bardziej niepewną."
            )
        else:
            st.caption(
                f"Różnica GNN-MLP: {result.model_difference:.3f} pIC50."
            )

    descriptor_names = {
        "mw": "Masa cząsteczkowa",
        "logp": "LogP",
        "tpsa": "TPSA",
        "hbd": "Donory H",
        "hba": "Akceptory H",
        "rotatable_bonds": "Wiązania rotowalne",
        "heavy_atoms": "Ciężkie atomy",
    }
    descriptor_table = pd.DataFrame(
        {
            "Deskryptor": [
                descriptor_names[name] for name in result.descriptors
            ],
            "Wartość": [
                round(value, 3) for value in result.descriptors.values()
            ],
        }
    )
    with st.expander("Deskryptory RDKit"):
        st.dataframe(descriptor_table, hide_index=True, width="stretch")


st.title("BACE1 Predictor: GNN + RDKit + LLM")
st.write(
    "System przewiduje `pIC50` cząsteczki wobec ludzkiego białka "
    "**BACE1 (CHEMBL4822)**. Główny model to GNN operujący na atomach i "
    "wiązaniach; MLP służy jako model porównawczy."
)

with st.sidebar:
    st.header("Informacje o modelu")
    st.markdown(
        """
        - Dane: ChEMBL 36, Homo sapiens
        - Podział: Bemis-Murcko scaffold split
        - GNN test: MAE 0,662; R² 0,557
        - MLP test: MAE 0,603; R² 0,617
        - LLM: lokalna Gemma 4 przez Ollama
        """
    )

smiles = st.text_input(
    "SMILES cząsteczki",
    value=EXAMPLE_SMILES,
    help="Wprowadź poprawny zapis SMILES.",
)

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.button("Przewidź aktywność", type="primary"):
    try:
        with st.spinner("Featuryzacja RDKit i predykcja modeli..."):
            st.session_state.prediction = load_predictor().predict(smiles)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        st.session_state.prediction = None
        st.error(str(error))

result = st.session_state.prediction
if result is not None:
    show_prediction(result)

    st.divider()
    st.subheader("Asystent LLM")
    configuration = get_llm_configuration()
    if not configuration.server_available:
        st.warning(
            "Ollama nie jest uruchomiona. Otwórz aplikację Ollama i odśwież "
            "stronę."
        )
    elif not configuration.model_available:
        st.warning(
            f"Brak modelu `{configuration.model}`. Pobierz go poleceniem: "
            f"`ollama pull {configuration.model}`."
        )
    else:
        st.success(
            f"Lokalny LLM jest gotowy. Model: `{configuration.model}`."
        )

    question = st.text_area(
        "Pytanie do asystenta",
        value=(
            "Zinterpretuj wynik, porównaj GNN z MLP i wyjaśnij najważniejsze "
            "ograniczenia tej predykcji."
        ),
    )
    if st.button("Zapytaj LLM"):
        if not (
            configuration.server_available
            and configuration.model_available
        ):
            st.error("Uruchom Ollamę i upewnij się, że model jest pobrany.")
        else:
            try:
                with st.spinner(
                    "Lokalny model przygotowuje odpowiedź. Pierwsze "
                    "uruchomienie może potrwać dłużej..."
                ):
                    response = answer_question(question, result)
                st.markdown(response)
            except Exception as error:
                st.error(f"Nie udało się uzyskać odpowiedzi LLM: {error}")
