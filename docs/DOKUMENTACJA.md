# Dokumentacja projektu

## 1. Cel

Celem projektu jest przewidywanie aktywności związków chemicznych wobec
białka BACE1. Białko to występuje u człowieka i w bazie ChEMBL ma identyfikator
`CHEMBL4822`.

Wejściem systemu jest struktura cząsteczki zapisana jako SMILES. Wynikiem jest
przewidywane `pIC50`.

## 2. Dane

Dane pochodzą z lokalnej bazy ChEMBL 36. Wybrano pomiary:

- dla pojedynczego białka BACE1,
- wykonane dla organizmu `Homo sapiens`,
- opisane jako `IC50`,
- podane w nM,
- z relacją `=`,
- posiadające wartość `pchembl_value`,
- o wiarygodności `confidence_score >= 8`.

Po zastosowaniu filtrów otrzymano 10 609 pomiarów dla 8 079 struktur SMILES.
Po standaryzacji RDKit i połączeniu powtórzonych pomiarów powstał zbiór
zawierający 8 050 cząsteczek.

Powtórzenia tej samej struktury agregowano za pomocą mediany `pIC50`. Mediana
jest mniej wrażliwa na pojedyncze nietypowe wyniki niż średnia.

## 3. Analiza danych

Najważniejsze obserwacje:

- zakres `pIC50` wynosił od 2,54 do 10,96,
- średnia wartość `pIC50` wynosiła 6,83,
- 2 150 struktur miało więcej niż jeden pomiar,
- 14 wartości zostało oznaczonych jako odstające metodą IQR,
- po standaryzacji pozostało 2 852 różnych scaffoldów.

Wartości odstających nie usunięto automatycznie. Mogą one reprezentować
rzeczywiście bardzo silne lub bardzo słabe inhibitory. Największym problemem
okazała się zmienność powtórzonych pomiarów, a nie brakujące dane.

Proste deskryptory RDKit miały słabą korelację z `pIC50`. Oznacza to, że sama
masa cząsteczkowa, LogP lub TPSA nie wystarczają do dobrego przewidywania
aktywności.

Wykresy z analizy:

- [rozkład pIC50](../reports/figures/01_pic50_distribution.png),
- [zmienność powtórzonych pomiarów](../reports/figures/02_repeat_variability.png),
- [korelacje deskryptorów](../reports/figures/03_descriptor_correlations.png).

## 4. Przygotowanie cech

Dla modeli ExtraTrees i MLP wykorzystano:

- fingerprint Morgan o długości 2 048 bitów i promieniu 2,
- masę cząsteczkową,
- LogP,
- TPSA,
- liczbę donorów i akceptorów wiązań wodorowych,
- liczbę wiązań rotowalnych,
- liczbę ciężkich atomów.

Łącznie daje to 2 055 cech. W modelu MLP deskryptory liczbowe zostały
przeskalowane na podstawie zbioru treningowego. Fingerprint pozostał binarny.

W modelu GNN każda cząsteczka jest grafem. Atomy są węzłami, a wiązania
krawędziami. Cechy atomów opisują między innymi pierwiastek, ładunek,
hybrydyzację i aromatyczność. Cechy wiązań zawierają ich typ, stereochemię
oraz informację o przynależności do pierścienia.

## 5. Podział danych

Zastosowano scaffold split oparty na scaffoldach Bemisa-Murcko:

| Zbiór | Liczba cząsteczek |
|---|---:|
| Treningowy | 6 440 |
| Walidacyjny | 805 |
| Testowy | 805 |

W każdym podzbiorze znajdują się inne scaffoldy. Dzięki temu test lepiej
sprawdza działanie modeli dla nowych rodzin związków niż zwykły podział
losowy.

## 6. Modele

### ExtraTrees

Model bazowy wykorzystuje 250 drzew i wszystkie przygotowane cechy RDKit.
Nie wymaga skalowania danych i dobrze radzi sobie z binarnym fingerprintem.

### MLP

Sieć MLP ma dwie warstwy ukryte:

```text
2055 -> 256 -> ReLU -> 128 -> ReLU -> 1
```

Model był trenowany optymalizatorem Adam. Zastosowano early stopping na
podstawie wyniku zbioru walidacyjnego.

### GNN

Model grafowy wykorzystuje trzy warstwy `GINEConv`, które przetwarzają cechy
atomów i wiązań. Po warstwach grafowych reprezentacje atomów są łączone przez
mean pooling i max pooling, a następnie przekształcane w jedną wartość
`pIC50`.

## 7. Wyniki

Wyniki na zbiorze testowym:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Dummy | 1,097 | 1,334 | -0,086 |
| ExtraTrees | 0,587 | 0,777 | 0,632 |
| MLP | 0,603 | 0,792 | 0,617 |
| GNN | 0,662 | 0,852 | 0,557 |

Wszystkie trzy modele są wyraźnie lepsze od modelu Dummy, który przewiduje
średnią wartość ze zbioru treningowego. Najlepszy wynik uzyskał ExtraTrees.
MLP osiągnął zbliżoną jakość, natomiast prosty GNN był nieco słabszy.

Modele mają tendencję do zawyżania wyników dla bardzo słabych inhibitorów i
zaniżania wyników dla bardzo silnych inhibitorów. Wynika to między innymi z
mniejszej liczby przykładów na krańcach rozkładu oraz ze zmienności pomiarów
laboratoryjnych.

## 8. Aplikacja i LLM

Aplikacja Streamlit przyjmuje SMILES, sprawdza jego poprawność i wyświetla:

- ustandaryzowany SMILES,
- strukturę 2D wygenerowaną przez RDKit,
- predykcje GNN i MLP,
- przybliżone `IC50` w nM,
- podstawowe deskryptory cząsteczki.

Użytkownik może również zadać pytanie lokalnemu modelowi językowemu
`gemma4:e2b`, uruchamianemu przez Ollama. LLM otrzymuje predykcje, deskryptory
oraz informacje o zastosowanych modelach i na tej podstawie przygotowuje
krótką interpretację.
