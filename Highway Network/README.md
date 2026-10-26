# Highway Network na CIFAR-10

Główny plik: **`Highway_Network_CIFAR10.ipynb`**. Zawiera teorię, kompletny kod
modelu, dane, trening, testy poprawności, ewaluację i analizę bramek.
Kod nie wymaga importowania własnych modułów `.py`.
Artykuł znajduje się: [https://arxiv.org/abs/1505.00387](https://arxiv.org/abs/1505.00387)

## Uruchomienie

1. Otwórz notebook w VS Code lub Jupyter, z katalogiem projektu jako katalogiem roboczym.
2. Wybierz kernel Python z `torch`, `torchvision`, `numpy`, `matplotlib` i `ipykernel`.
   Brakujące zależności można zainstalować przez `python -m pip install -r requirements.txt`.
   Dla GPU potrzebny jest PyTorch z CUDA; przetestowano torch 2.7.1+cu128,
   torchvision 0.22.1+cu128, Python 3.11.8 oraz RTX 4070 SUPER.
   [Instrukcja doboru instalacji](https://pytorch.org/get-started/locally/).
3. Przy nowym eksperymencie ustaw w `Config` nowy `output_dir` i uruchom `Run All`.
   Domyślne ustawienia: 40 epok, batch 128, 9 bloków Highway, kanały 32/64/128.
4. CIFAR-10 pobierze się automatycznie do `data/` (archiwum około 170 MB).

W zapisanym uruchomieniu wykorzystano identyczne archiwum z lustra
[MIT Open Learning](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python).
Suma MD5 została porównana z wartością wymaganą przez torchvision: 
`c58f30108f718f92721af3b95e74349a`. Pochodzenie i wynik weryfikacji zapisano w 
`data/dataset_source.json`.

Zapisane wyjścia notebooka można oglądać bez ponownego treningu. Aby wznowić
trening lub ponownie wykonać notebook z istniejącym checkpointem, należy ustawić
`resume=True`. Wznowienie wymaga zachowania hiperparametrów, w tym liczby epok
(wpływa na scheduler). Ukończony eksperyment zostanie wczytany bez dodatkowych epok.
Katalog z checkpointami jest chroniony przed przypadkowym nadpisaniem treningu.

## Protokół

- Równanie: `y = ReLU(Conv_H(x)) * sigmoid(Conv_T(x)) + x * (1 - sigmoid(Conv_T(x)))`.
- Bias bramki na starcie: -1; carry gate jest równa `1-T`.
- Zwykłe konwolucje zmieniają liczbę kanałów i rozdzielczość pomiędzy etapami.
- Brak BatchNorm i brak dodatkowej aktywacji po zsumowaniu ścieżek Highway.
- Stratyfikowany podział: 45 000 train, 5 000 validation, 10 000 test.
- Statystyki normalizacji wyłącznie z train; augmentacja wyłącznie w train.
- SGD z momentum, cosine LR, cross entropy, AMP na CUDA.
- Wybór epoki według validation accuracy; końcowa ocena na całym teście.

## Pliki

`outputs/highway_cifar10/` zawiera:

- `best.pt` i `last.pt`: najlepszy model i pełny stan treningu.
- `history.csv`, `config.json`, `summary.json`, `split_indices.npz`.
- `per_class_metrics.csv`, `test_predictions.npz`, `confusion_matrix.npy`.
- `gate_statistics.json` oraz wykresy PNG.

`summary.json` i wyjścia notebooka zawierają rzeczywiście zmierzone metryki.
Dokładne wznowienie generatorów zakłada domyślne `num_workers=0`.
Podane ziarno nie gwarantuje identycznych wyników na innych wersjach bibliotek
i innym sprzęcie.
