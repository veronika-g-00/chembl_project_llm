# Przewidywanie aktywności inhibitorów BACE1

Projekt służy do przewidywania aktywności związków chemicznych wobec
ludzkiego białka BACE1 (`CHEMBL4822`). Na podstawie zapisu SMILES modele
wyznaczają przewidywane `pIC50`.

W projekcie wykorzystano dane ChEMBL 36, bibliotekę RDKit, modele ExtraTrees,
MLP i GNN oraz interfejs Streamlit. Aplikacja pokazuje strukturę 2D cząsteczki
i przekazuje wyniki do lokalnego modelu językowego.

## Wymagania

- Python 3.11,
- Ollama,
- około 8 GB wolnego miejsca na model `gemma4:e2b`.

Baza `chembl_36.db` nie jest potrzebna do uruchomienia gotowej aplikacji.
Jest wymagana tylko do ponownego wykonania ekstrakcji i treningu.

## Instalacja

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull gemma4:e2b
```

## Uruchomienie aplikacji

Uruchom Ollamę, a następnie w katalogu projektu wykonaj:

```powershell
py -3.11 -m streamlit run app.py
```

Aplikacja otworzy się pod adresem `http://localhost:8501`.
Pierwsza odpowiedź LLM może pojawić się po około 1-2 minutach.

## Ponowne wykonanie pipeline'u

Po umieszczeniu bazy `chembl_36.db` w głównym katalogu:

```powershell
py -3.11 scripts\extract_bace1.py
py -3.11 scripts\run_eda.py
py -3.11 scripts\train_baseline.py
py -3.11 scripts\train_mlp.py
py -3.11 scripts\train_gnn.py
```

## Wyniki

Wyniki na zbiorze testowym utworzonym metodą scaffold split:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| ExtraTrees | 0,587 | 0,777 | 0,632 |
| MLP | 0,603 | 0,792 | 0,617 |
| GNN | 0,662 | 0,852 | 0,557 |

Opis danych, przygotowania modeli i wyników znajduje się w
[`docs/DOKUMENTACJA.md`](docs/DOKUMENTACJA.md).

## Testy

```powershell
py -3.11 -m unittest discover -s tests -v
```
