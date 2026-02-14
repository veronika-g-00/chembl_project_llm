# ChEMBL - ML project

Projekt przetwarza dane o związkach chemicznych z bazy ChEMBL w celu przygotowania ich do analizy z wykorzystaniem uczenia maszynowego.

---

## Cel projektu

Celem projektu jest przygotowanie danych o aktywności związków chemicznych z bazy ChEMBL do dalszej analizy oraz wykorzystania w modelach uczenia maszynowego. W tym celu surowe dane IC50 są standaryzowane poprzez ujednolicenie jednostek, przeliczane do postaci pIC50 oraz poddawane procesowi czyszczenia, który obejmuje usuwanie braków, niepoprawnych wartości i duplikatów.

## Pipeline wykorzystuje:

- **Apache Spark**– do przetwarzania danych w sposób rozproszony
- **Apache Airflow** – do orkiestracji pipeline’u
- **Docker** – do uruchamiania infrastruktury
- **Parquet** – jako docelowy format danych

## Wymagania:

- Python 3.9+
- Docker Desktop
- min. 8 GB RAM
- projekt korzysta z bazy ChEMBL 36 w formacie SQLite (ok 28 GB)

---

## Struktura projektu

```
PyCharmMiscProject/
│
├── chembl_36.db                    # Baza danych ChEMBL (28GB SQLite)
│
├── requirements.txt                # Wymagane pakiety Pythona
├── README.md
│
├── export_to_parquet.py           # Skrypt: konwersja SQLite → Parquet
├── spark_etl_job.py               # Skrypt: przetwarzanie danych w Spark
│
├── libs/                          # Kod biblioteki Python
│   ├── __init__.py
│   ├── data_processing.py         # Funkcje ładowania i przetwarzania danych
│   └── datasets/                  # Pliki danych
│       ├── chembl_raw.parquet         # Wyeksportowane dane surowe
│       ├── chembl_sample.parquet      # Mała próbka do testów
│       └── chembl_processed.parquet/  # Końcowy wynik przetwarzania
│
├── notebooks/                     # Notebooki Jupyter
│   └── eda.ipynb                  # Eksploracyjna analiza danych
│
└── spark_airflow/                 # Konfiguracja infrastruktury
    ├── DockerCompose.yml          # Konfiguracja Docker
    ├── local_cluster.yml          # Alternatywny skrypt konfiguracji
    └── dags/                      # Definicje DAG w Airflow
        └── chembl_etl_dag.py      # Konfiguracja harmonogramu pipeline

```

---

## Instalacja i uruchomienie

### Środowisko Python

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Uruchomienie infrastruktury

```bash
cd spark_airflow
docker compose -f DockerCompose.yml up -d
```

### Uruchomienie pipeline’u

1. Eksport danych

```bash
python export_to_parquet.py --limit 50000
```

lub pełny eksport:

```bash
python export_to_parquet.py
```

Powstaje plik:

```bash
libs/datasets/chembl_raw.parquet
```

2. Przetwarzanie w Spark

```bash
docker exec spark-master bash -c "cd /opt/workspace && /opt/spark/bin/spark-submit \
--master spark://spark-master:7077 \
--driver-memory 1g \
--executor-memory 1g \
spark_etl_job.py \
--input /opt/workspace/libs/datasets/chembl_raw.parquet \
--output /opt/workspace/libs/datasets/chembl_processed.parquet"
```

Efekt:
```bash
libs/datasets/chembl_processed.parquet
```
