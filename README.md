# HW2: Image Classification con transfer learning (Oxford-IIIT Pet, MobileNetV3-Small)

Confronto tra full fine-tuning, training della sola head e feature congelate + classificatore scikit-learn.

## Ambiente
- Python 3.12, PyTorch 2.15.0+cu132, TorchVision 0.29.2, scikit-learn 1.9.1
- GPU NVIDIA (CUDA 13.2) in WSL2 Ubuntu; i tempi nel report si riferiscono a questo hardware
- Dipendenze: `pip install -r requirements.txt`

## Dataset
Oxford-IIIT Pet (Parkhi et al., CVPR 2012), licenza CC BY-SA 4.0: https://www.robots.ox.ac.uk/~vgg/data/pets/
Non è incluso nel repository: viene scaricato in `data/` da `torchvision.datasets.OxfordIIITPet(download=True)`.
Split: `trainval` ufficiale diviso 85/15 in train/val con stratificazione per razza (seed 42, file `splits/split.json`); `test` ufficiale tenuto separato.

## Riproduzione
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train_nn --mode head --lr 1e-3
python -m src.train_nn --mode head --lr 3e-3
python -m src.train_nn --mode full --lr 1e-4
python -m src.train_nn --mode full --lr 3e-4
python -m src.features_sk      # feature 576-d + LogReg / LinearSVC / RandomForest
python -m src.evaluate         # valutazione finale sul test, matrici, esempi
```
Più seed (iperparametri fissati sulla validation):
```bash
for s in 1 2; do
  python -m src.train_nn --mode head --lr 1e-3 --seed $s
  python -m src.train_nn --mode full --lr 3e-4 --seed $s
done
python -m src.multiseed
```

## Scelte (selezionate solo sulla validation)
| Approccio | Ottimizzatore | LR | Batch | Epoche | Selezione |
|---|---|---|---|---|---|
| Full FT | AdamW (wd 1e-4) | 3e-4 | 64 | 15 | miglior val acc |
| Head | AdamW (wd 1e-4) | 1e-3 | 64 | 15 | miglior val acc |
Loss: CrossEntropy. Feature: LogReg C=0.01, LinearSVC C=0.001, RF 200 alberi (StandardScaler fittato sul solo train).
Parametri allenabili: full = tutta la rete; head = `classifier[0]` (Linear 576→1024) e `classifier[3]` (Linear 1024→37).

## Riproducibilità
Seed (default 42) per Python, NumPy, PyTorch; `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`.
Due run identiche danno gli stessi valori epoca per epoca; varia solo il tempo. Possibili residui: kernel GPU non deterministici, versioni di driver/librerie.
Il test non è mai usato per selezionare checkpoint o iperparametri.

## Output
`results/`: `*_train.json` (storia di training), `features_selection.json`, `test_metrics.json`, `cm_*.png`, `examples_*.png`.