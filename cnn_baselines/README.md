# Experimentos com CNNs

Primeira fase do projeto: classificação em sete classes (Normal e graus 0 a 5 de Wagner) com redes convolucionais pré-treinadas no ImageNet, em TensorFlow/Keras, com registro no MLflow. Estes scripts geraram a comparação com as CNNs na Tabela I do artigo.

## Estrutura

```
cnn_baselines/
├── models/<arquitetura>/<arquitetura>_mlflow.py   # um script de treino e avaliação por modelo
├── models/model_logger.py, runtime_config.py      # registro dos resultados e configuração de execução
├── run_all_models.py, run_selected_models.py      # execução em lote
├── check_models_status.py                         # situação das execuções
├── utils/      # balanceamento da base (v3), conversão para TFLite, predição e visualizações
├── tools/      # análises auxiliares, testes de modelos .h5/.tflite e execução em Docker (GPU AMD)
├── notebooks/  # análise da base e dos resultados (resultado_comparativo_treinamento.csv)
├── examples/   # notebook introdutório de classificação com CNN
└── figuras/    # figuras da análise da base de dados
```

## Modelos

| Pasta | Modelo | No artigo |
|---|---|---|
| `mobilenet_v1_224`, `mobilenet_v2_224`, `mobilenet_v3_224` | MobileNet V1, V2 e V3-Large | sim |
| `efficientnet_b0_224` … `efficientnet_b7_600` | EfficientNet B0 a B7, cada uma na resolução nativa | sim |
| `resnet50_224` | ResNet-50 | sim |
| `inception_v3_299` | InceptionV3 | não (exploratório) |
| `vision_transformer_224` | ViT (base e large) | não (exploratório) |

Protocolo do artigo:
- cabeça com *dropout* de 0,20 e camada densa com saída *softmax* de sete classes;
- otimizador Adam;
- 20 épocas com o *backbone* congelado (taxa de aprendizado 10⁻³), seguidas de 10 épocas de ajuste fino (10⁻⁵), com a normalização em lote congelada;
- *batch* de 32.

## Uso

Execute a partir da **raiz do repositório**, para que a pasta `datasets/` seja encontrada:

```bash
pip install -r cnn_baselines/requirements.txt
python cnn_baselines/models/mobilenet_v2_224/mobilenet_v2_224_mlflow.py   # um modelo
python cnn_baselines/run_all_models.py                                    # todos os modelos
```

Por padrão, os scripts usam `datasets/perfect_balanced_dataset_wagner_v4_1000` (1.000 imagens por classe, divididas antes do aumento de dados). Outra base pode ser indicada com:

```bash
export CLAUCIA_CNN_DATASET=/caminho/para/base   # pastas treino/, validacao/, teste/, uma subpasta por classe
```

As saídas de cada execução (figuras, relatórios `.json` e modelos `.h5`/`.tflite`) ficam na pasta do modelo e não são versionadas. Para treinar em GPU AMD via Docker, veja [`../docs/COMO_TREINAR_NO_DOCKER.md`](../docs/COMO_TREINAR_NO_DOCKER.md).
