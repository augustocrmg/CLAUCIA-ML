# Scripts TensorFlow Lite

Este diretório contém dois scripts para trabalhar com modelos TensorFlow Lite (.tflite):

## 📁 Arquivos

- `convert_h5_to_tflite.py` - Converte modelos .h5 para .tflite
- `test_tflite_models.py` - Testa modelos .tflite para verificar se funcionam
- `README_TFLITE.md` - Esta documentação

## 🔄 Script 1: Conversão H5 → TFLite

### Uso Básico
```bash
# Conversão interativa (lista modelos disponíveis)
python utils/convert_h5_to_tflite.py

# Converte um modelo específico
python utils/convert_h5_to_tflite.py --model_path caminho/para/modelo.h5

# Converte todos os modelos encontrados
python utils/convert_h5_to_tflite.py --all

# Conversão com otimização para tamanho
python utils/convert_h5_to_tflite.py --optimization size

# Conversão com otimização para velocidade  
python utils/convert_h5_to_tflite.py --optimization speed
```

### Opções de Otimização
- `default`: Otimização padrão (balanceada)
- `size`: Otimização para menor tamanho de arquivo (usa float16)
- `speed`: Otimização para maior velocidade de inferência

### Exemplo com modelo do repositório
```bash
python utils/convert_h5_to_tflite.py --model_path ./efficientnet_b0_244.h5
```

## 🧪 Script 2: Teste de Modelos TFLite

### Uso Básico
```bash
# Teste interativo (lista modelos .tflite disponíveis)
python utils/test_tflite_models.py

# Testa um modelo específico
python utils/test_tflite_models.py --model_path caminho/para/modelo.tflite

# Testa todos os modelos encontrados
python utils/test_tflite_models.py --all

# Testa com dados específicos
python utils/test_tflite_models.py --test_data_path datasets/seu_dataset
```

### O que o script testa:
- ✅ **Carregamento**: Verifica se o modelo pode ser carregado
- ✅ **Inferência**: Executa predições com dados de teste
- ✅ **Performance**: Mede tempo de inferência e FPS
- ✅ **Consistência**: Verifica se as saídas são consistentes
- ✅ **Detalhes**: Mostra shapes de entrada/saída e tipos de dados

### Dados de Teste
O script automaticamente:
1. 🔍 Procura por datasets reais no repositório
2. 🎲 Cria dados sintéticos se não encontrar dados reais
3. 📊 Analisa os resultados das predições

## 🚀 Fluxo de Trabalho Recomendado

### 1. Converter modelo existente
```bash
cd /home/augusto/Repos/CLAUCIA-ML
python utils/convert_h5_to_tflite.py
```
Escolha um modelo da lista (ex: efficientnet_b0_244.h5)

### 2. Testar modelo convertido
```bash
python utils/test_tflite_models.py
```
Escolha o modelo .tflite criado no passo anterior

### 3. Comparar resultados
- Compare os tempos de inferência
- Verifique se as saídas fazem sentido
- Teste com dados reais se disponíveis

## 📊 Informações dos Testes

O script de teste mostra:
- **Input/Output shapes**: Dimensões de entrada e saída
- **Tempo de inferência**: Médio, mínimo e máximo
- **FPS estimado**: Quantos frames por segundo
- **Predições de amostra**: Exemplos de saídas do modelo
- **Consistência**: Se todas as predições têm o mesmo formato

## ⚠️ Observações

- Os modelos .h5 devem estar treinados e funcionando
- A conversão para TFLite pode alterar ligeiramente a precisão
- Modelos otimizados para tamanho (float16) podem ter menor precisão
- Sempre teste os modelos convertidos antes de usar em produção

## 🐛 Solução de Problemas

### Erro ao carregar modelo .h5
- Verifique se o arquivo não está corrompido
- Confirme que o TensorFlow está instalado corretamente

### Erro ao converter para TFLite
- Alguns layers podem não ser suportados pelo TFLite
- Tente com otimização `default` primeiro

### Modelo TFLite não funciona
- Verifique se a conversão foi bem-sucedida
- Compare os shapes de entrada/saída com o modelo original 