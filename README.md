# CLAUCIA-ML

Classificação de imagens de pé diabético em uma classe Normal e nos graus 0 a 5 da escala de Wagner, com modelos de aprendizado profundo executados localmente em dispositivos Android. Este repositório reúne o componente de aprendizado de máquina do ecossistema CLAUCIA: preparação dos dados, treino, avaliação, exportação para LiteRT e medições nos aparelhos.

> O sistema é uma ferramenta de **apoio à decisão**. A classificação usa apenas imagens RGB, que não contêm todos os critérios clínicos da escala de Wagner, e não substitui a avaliação do profissional de saúde.

## Ecossistema CLAUCIA

| Repositório | Função |
|---|---|
| [CLAUCIA-app](https://github.com/Eduardo-RFarias/CLAUCIA-app) | **Aplicativo móvel** Android (Flutter): cadastro de pacientes e feridas, captura de imagens e classificação no próprio aparelho |
| [CLAUCIA-api](https://github.com/Eduardo-RFarias/CLAUCIA-api) | API REST do sistema (NestJS, TypeORM, MySQL, autenticação JWT) |
| [CLAUCIA-web](https://github.com/Eduardo-RFarias/CLAUCIA-web) | Aplicação web |
| [CLAUCIA-infra](https://github.com/Eduardo-RFarias/CLAUCIA-infra) | Infraestrutura e implantação |
| **CLAUCIA-ML** (este) | Modelos, avaliação, exportação e benchmarks |

## Principais resultados

| Item | Resultado |
|---|---|
| Modelo implantado | DINOv3-Small adaptado com LoRA, quantizado em INT8 *weight-only* e executado com LiteRT |
| Tamanho do modelo | 82,9 MB (FP32) → 21,8 MB (INT8), redução de 73,7% |
| Divisão original (teste) | acurácia 0,818 e F1 macro 0,803 (modelo LiteRT) |
| Validação cruzada agrupada, conjunto estendido | F1 macro 0,751 (IC 95% 0,717–0,783), AUC-ROC 0,947, QWK 0,920 |
| Latência por imagem (CPU, 4 threads) | 218,6 ms no Galaxy S8; 375,4 ms no Galaxy Tab S6 Lite |
| Energia por inferência | 1,20 J (Galaxy S8); 1,19 J (Galaxy Tab S6 Lite) |
| Calibração (teste) | ECE 0,053; com limiar de confiança de 0,60, 41,2% dos erros são encaminhados para revisão |

Os valores detalhados estão em [`dinov3/results/`](dinov3/results/).

## Estrutura

```
CLAUCIA-ML/
├── dinov3/                     # pipeline do artigo (DINOv3 + LoRA)
│   ├── index_all.py, cluster.py, labels.py, map_patches_to_originals.py,
│   │   build_v6.py, build_v7.py, balance_v6.py   # preparação e partição dos dados
│   ├── train_dinov3.py, run_*.sh                 # treino (divisão original e validação cruzada agrupada)
│   ├── aggregate_results.py, metrics_prof.py, eval_prof.py   # métricas, calibração, IC por bootstrap
│   ├── export_tflite.py, export_arch_only.py, export_*.sh    # conversão e quantização para LiteRT
│   ├── bench_device.py, energy_bench.py          # latência, memória e energia em aparelhos Android
│   ├── notebooks/              # notebook do treino de referência (divisão original)
│   ├── data/manifest_v7.csv    # composição, grupos e folds do conjunto estendido (sem imagens)
│   └── results/                # resultados usados no artigo
├── cnn_baselines/              # experimentos com CNNs (MobileNet, EfficientNet, ResNet), TensorFlow
└── docs/                       # guias de execução (Docker, GPU AMD, fine-tuning)
```

As imagens, os pesos dos modelos e os registros de treino não são versionados; ficam em `datasets/` e nas pastas listadas no `.gitignore`.

## Reprodução (pasta `dinov3/`)

**Ambientes.** O treino e a exportação usam ambientes separados:

```bash
python -m venv .venv-treino && .venv-treino/bin/pip install -r dinov3/requirements-treino.txt
python -m venv .venv-export && .venv-export/bin/pip install -r dinov3/requirements-exportacao.txt
```

**Dados.** As imagens vêm de conjuntos públicos do Kaggle e do *Lower limb and feet wound image dataset* (Mendeley Data, doi: 10.17632/hsj38fwnvr.3). Elas devem ser colocadas em `datasets/`.

**Etapas.**
1. Indexação, agrupamento e rótulos: `index_all.py` → `cluster.py` → `labels.py` → `map_patches_to_originals.py`.
2. Conjuntos e partição em *folds* agrupados: `build_v6.py` → `build_v7.py`.
3. Balanceamento do treino: `balance_v6.py`, com as variáveis `CBEB_SRC` e `CBEB_OUT`.
4. Treino:
   - `run_prof_split.sh` para a divisão original e a ablação *head-only*/LoRA/*full fine-tuning*;
   - `run_v7.sh` e `run_v7_ablation.sh` para a validação cruzada agrupada.
5. Métricas: `aggregate_results.py` e `metrics_prof.py`.
6. Exportação para LiteRT (FP32, INT8 *weight-only*, INT8 dinâmico): `export_prof.sh` e `export_v7.sh`.
7. Medições no aparelho, com `adb` e o `benchmark_model` do LiteRT: `bench_device.py` e `energy_bench.py`. No Galaxy S8, use `--current_node batt_current_ua_now`.

**Variáveis de ambiente.**
- **Treino e exportação:** `CBEB_DATA_ROOT`, `CBEB_RESULTS`, `CBEB_TFLITE`, `CBEB_E1_ROOT` e `CBEB_E2`.
- **Treino de referência:** `CLAUCIA_PROF_RESULTS` e `CLAUCIA_V4_SPLIT`.
- **Interpretadores dos scripts `.sh`:** `PY` e `PY_EXPORT`.

## Código das CNNs (pasta `cnn_baselines/`)

Execute a partir da raiz do repositório, por exemplo `python cnn_baselines/run_all_models.py`. Detalhes em [`cnn_baselines/README.md`](cnn_baselines/README.md) e em [`docs/`](docs/).

## Artigo

A. Camargo, E. R. de Farias, A. Bezerra, A. Müller, M. Carneiro e V. Rispoli, "A Mobile Decision-Support Application for Diabetic Foot Ulcer Classification", XXX Congresso Brasileiro de Engenharia Biomédica (CBEB 2026). Uma versão estendida foi selecionada para os *Proceedings* do IEEE.

## Agradecimentos

Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq), bolsa de Desenvolvimento Tecnológico e Extensão Inovadora (processo 380610/2026-4).

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
