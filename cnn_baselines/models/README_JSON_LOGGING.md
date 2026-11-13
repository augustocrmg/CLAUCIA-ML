# Sistema de Logging JSON e Conversão TensorFlow Lite

Este documento descreve o sistema de logging JSON automatizado e a funcionalidade de conversão TensorFlow Lite implementados para os modelos de machine learning.

## Visão Geral

O sistema oferece duas funcionalidades principais:

1. **Logging JSON**: Captura automaticamente métricas de treinamento e salva em formato JSON padronizado
2. **Conversão TensorFlow Lite**: Converte modelos treinados para formato `.tflite` para uso em dispositivos móveis e edge computing

## Funcionalidades

### 1. Logging JSON

#### Estrutura dos Dados Salvos

Cada execução de modelo gera um arquivo JSON com as seguintes informações:

```json
{
  "dataset": "dataset_wagner_v4_balanceado",
  "modelo": "EfficientNet-B0",
  "batch_size": 32,
  "epochs": 20,
  "fine_tuning_epochs": 10,
  "train_acc": 0.9234,
  "valid_acc": 0.8876,
  "train_fine_tuning_acc": 0.9567,
  "valid_finetuning_acc": 0.9123,
  "data_rodada": "2024-12-01 14:30:22"
}
```

#### Campos Capturados

- **dataset**: Nome do dataset utilizado
- **modelo**: Nome da arquitetura do modelo
- **batch_size**: Tamanho do batch usado no treinamento
- **epochs**: Número de épocas do treinamento inicial
- **fine_tuning_epochs**: Número de épocas do fine-tuning
- **train_acc**: Acurácia final de treinamento (fase inicial)
- **valid_acc**: Acurácia final de validação (fase inicial)
- **train_fine_tuning_acc**: Acurácia final de treinamento (fine-tuning)
- **valid_finetuning_acc**: Acurácia final de validação (fine-tuning)
- **data_rodada**: Timestamp da execução

### 2. Conversão TensorFlow Lite

#### Formatos de Saída

Cada modelo treinado é salvo em dois formatos:

1. **Formato H5** (`.h5`): Formato padrão do Keras/TensorFlow
2. **Formato TensorFlow Lite** (`.tflite`): Formato otimizado para dispositivos móveis e edge

#### Vantagens do TensorFlow Lite

- **Tamanho reduzido**: Modelos menores para deploy em dispositivos com recursos limitados
- **Velocidade**: Inferência mais rápida em dispositivos móveis
- **Compatibilidade**: Suporte nativo para Android, iOS e dispositivos embarcados
- **Otimizações**: Quantização automática para melhor performance

## Localização dos Arquivos

### Arquivos JSON
- **Diretório**: `models_results/`
- **Padrão de nome**: `{nome_modelo}_{data}.json`
- **Exemplo**: `EfficientNet-B0_2024-12-01.json`

### Arquivos de Modelo
- **Diretório**: Pasta específica de cada modelo (ex: `models/efficientnet_b0_224/`)
- **Formatos salvos**:
  - `{nome_modelo}.h5` (formato Keras)
  - `{nome_modelo}.tflite` (formato TensorFlow Lite)

## Implementação Técnica

### Arquivo Central: `model_logger.py`

O sistema é implementado através de três funções principais:

1. **`save_training_results()`**: Salva métricas em JSON
2. **`extract_final_metrics()`**: Extrai métricas finais dos históricos de treinamento
3. **`convert_model_to_tflite()`**: Converte modelos para formato TensorFlow Lite

### Integração nos Modelos

Cada modelo foi atualizado para incluir:

```python
# Import das funções
from model_logger import save_training_results, extract_final_metrics, convert_model_to_tflite

# Função de salvamento atualizada
def save_model(model, filename="model_name"):
    # Salvar formato H5
    model_path = os.path.join(SCRIPT_DIR, f"{filename}.h5")
    model.save(model_path)
    
    # Salvar formato TensorFlow Lite
    tflite_path = os.path.join(SCRIPT_DIR, f"{filename}.tflite")
    convert_model_to_tflite(model, tflite_path)
    
    # Log no MLflow
    mlflow.log_artifact(model_path)
```

## Modelos Implementados

Todos os 11 modelos incluem ambas as funcionalidades:

1. **EfficientNet-B0** (224x224)
2. **EfficientNet-B1** (240x240)
3. **EfficientNet-B2** (260x260)
4. **EfficientNet-B3** (300x300)
5. **EfficientNet-B4** (380x380)
6. **EfficientNet-B5** (456x456)
7. **EfficientNet-B6** (528x528)
8. **EfficientNet-B7** (600x600)
9. **ResNet50** (224x224)
10. **InceptionV3** (224x224)
11. **MobileNetV3Large** (224x224)

## Uso dos Modelos TensorFlow Lite

### Carregamento do Modelo

```python
import tensorflow as tf

# Carregar modelo TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="modelo.tflite")
interpreter.allocate_tensors()

# Obter detalhes de entrada e saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

### Inferência

```python
import numpy as np

# Preparar dados de entrada
input_data = np.array(input_image, dtype=np.float32)
input_data = np.expand_dims(input_data, axis=0)

# Executar inferência
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Obter resultado
output_data = interpreter.get_tensor(output_details[0]['index'])
```

## Características Técnicas

### Automação
- **Execução automática**: Ativado automaticamente ao final do treinamento
- **Sem intervenção manual**: Funciona quando os modelos são executados via função main
- **Tratamento de erros**: Captura e reporta erros de conversão

### Robustez
- **Criação automática de diretórios**: Cria `models_results/` se não existir
- **Codificação UTF-8**: Suporte completo a caracteres especiais
- **Validação de dados**: Conversão automática de tipos de dados
- **Backup de segurança**: Mantém formato H5 original

### Performance
- **Conversão eficiente**: Usa `tf.lite.TFLiteConverter.from_keras_model()`
- **Otimizações padrão**: Aplica otimizações básicas do TensorFlow Lite
- **Feedback visual**: Mostra progresso e resultados da conversão

## Exemplo de Uso

Ao executar qualquer modelo:

```bash
python efficientnet_b0_224_mlflow.py
```

O sistema automaticamente:

1. Treina o modelo
2. Salva em formato H5: `efficientnet_b0_244.h5`
3. Converte para TensorFlow Lite: `efficientnet_b0_244.tflite`
4. Gera arquivo JSON: `models_results/EfficientNet-B0_2024-12-01.json`

## Manutenção

### Logs de Sistema
- ✅ Sucesso: `✓ Training results saved to: ...`
- ✅ Sucesso: `✓ TensorFlow Lite model saved to: ...`
- ❌ Erro: `✗ Error converting model to TensorFlow Lite: ...`

### Verificação de Integridade
- Arquivos JSON são validados automaticamente
- Modelos TensorFlow Lite são testados após conversão
- Tamanhos de arquivo são reportados para verificação

## Referências

- [Documentação TensorFlow Lite](https://ai.google.dev/edge/litert/models/convert_tf)
- [Guia de Conversão TensorFlow](https://www.tensorflow.org/lite/convert)
- [Otimizações TensorFlow Lite](https://www.tensorflow.org/lite/performance/model_optimization) 