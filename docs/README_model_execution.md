# Scripts de Execução de Modelos

Este projeto inclui scripts para executar automaticamente todos os modelos de deep learning presentes na pasta `models/`. Os scripts foram projetados para facilitar o treinamento em lote e comparação de resultados.

## 📁 Estrutura de Arquivos

```
.
├── run_all_models.py          # Executa TODOS os modelos sequencialmente
├── run_selected_models.py     # Executa apenas modelos selecionados
├── models/                    # Diretório com os modelos
│   ├── efficientnet_b0_224/
│   ├── efficientnet_b1_240/
│   ├── efficientnet_b2_260/
│   ├── ...
│   └── resnet50_224/
└── execution_logs/            # Logs e relatórios de execução (criado automaticamente)
```

## 🚀 Scripts Disponíveis

### 1. `run_all_models.py` - Execução Completa

**Funcionalidade:**
- Executa TODOS os modelos na pasta `models/` sequencialmente
- Ideal para execução completa overnight ou em fim de semana
- Não requer interação do usuário após confirmação inicial

**Como usar:**
```bash
python run_all_models.py
```

**Características:**
- ✅ Execução automática de todos os modelos
- ✅ Logging detalhado para cada modelo
- ✅ Relatório final com estatísticas
- ✅ Timeout de 2 horas por modelo
- ✅ Tratamento de erros robusto
- ✅ Estimativa de tempo total

### 2. `run_selected_models.py` - Execução Seletiva

**Funcionalidade:**
- Permite escolher quais modelos executar
- Interface interativa para seleção
- Ideal para testes ou execução parcial

**Como usar:**
```bash
python run_selected_models.py
```

**Opções de seleção:**
- `1,3,5` - Seleciona modelos específicos
- `1-5` - Seleciona um intervalo
- `all` - Seleciona todos os modelos
- `efficient` - Apenas modelos EfficientNet
- `small` - Modelos menores (B0, B1, B2, MobileNet)
- `large` - Modelos maiores (B5, B6, B7)

## 📊 Recursos dos Scripts

### Logging Avançado
- **Logs em tempo real** no terminal
- **Arquivo de log detalhado** salvo em `execution_logs/`
- **Timestamp** para cada operação
- **Progress tracking** com contadores

### Relatórios de Execução
- **JSON detalhado** com todos os resultados
- **Estatísticas gerais** (sucesso, falhas, timeouts)
- **Tempo de execução** para cada modelo
- **Mensagens de erro** capturadas

### Tratamento de Erros
- **Timeout automático** (2 horas por modelo)
- **Captura de exceções** sem interromper o lote
- **Logs de erro detalhados**
- **Continuação** mesmo com falhas individuais

### Estimativas de Tempo
- **Tempo estimado** antes do início
- **Progresso em tempo real**
- **Tempo restante** atualizado dinamicamente

## 📋 Estrutura dos Logs

### Arquivo de Log Principal
```
execution_logs/model_execution_YYYYMMDD_HHMMSS.log
```

**Conteúdo:**
- Início/fim de cada modelo
- Progresso detalhado
- Mensagens de sucesso/erro
- Estatísticas finais

### Arquivo de Resumo JSON
```
execution_logs/execution_summary_YYYYMMDD_HHMMSS.json
```

**Estrutura:**
```json
{
  "execution_info": {
    "timestamp": "2024-01-15T10:30:00",
    "total_execution_time_hours": 12.5,
    "log_file": "path/to/log"
  },
  "overall_stats": {
    "total_models": 11,
    "successful_models": 9,
    "failed_models": 2,
    "success_rate": 81.8
  },
  "model_results": [
    {
      "model_name": "efficientnet_b0_224",
      "status": "SUCCESS",
      "execution_time": 3600,
      "start_time": "2024-01-15T10:30:00",
      "end_time": "2024-01-15T11:30:00"
    }
  ]
}
```

## 🛠️ Configuração e Requisitos

### Pré-requisitos
- Python 3.7+
- Todas as dependências dos modelos instaladas
- Pasta `models/` com scripts `*_mlflow.py`
- GPU configurada (se disponível)

### Estrutura Esperada dos Modelos
Cada diretório de modelo deve conter:
```
models/model_name/
├── model_name_mlflow.py    # Script principal de treinamento
└── ...                     # Outros arquivos do modelo
```

## ⚙️ Personalização

### Modificar Timeout
No código, altere a variável:
```python
timeout=7200  # 2 horas em segundos
```

### Adicionar Novos Filtros de Seleção
No `run_selected_models.py`, na função `select_models_interactively()`:
```python
elif selection.lower() == 'custom':
    return [s for s in model_scripts if 'criteria' in s['model_name'].lower()]
```

### Modificar Formato de Logs
Na função `setup_logging()`:
```python
format='%(asctime)s - %(levelname)s - %(message)s'
```

## 📈 Interpretação dos Resultados

### Status dos Modelos
- **SUCCESS** ✅ - Modelo executado com sucesso
- **FAILED** ❌ - Modelo falhou durante execução
- **TIMEOUT** ⏰ - Modelo excedeu limite de tempo
- **ERROR** 💥 - Erro de sistema/exceção

### Métricas de Performance
- **Taxa de sucesso** - Percentual de modelos bem-sucedidos
- **Tempo médio** - Tempo médio por modelo
- **Tempo total** - Duração completa da execução

## 🔧 Solução de Problemas

### Modelo não encontrado
Verifique se:
- O diretório existe em `models/`
- Contém arquivo `*_mlflow.py`
- O script tem permissões de execução

### Erro de memória GPU
- Reduza batch size nos modelos
- Execute modelos menores primeiro
- Monitore uso de GPU com `nvidia-smi`

### Timeout frequente
- Aumente o valor de timeout
- Verifique se o dataset está acessível
- Monitore recursos do sistema

## 📞 Exemplo de Uso Completo

```bash
# 1. Execução completa (todos os modelos)
python run_all_models.py

# 2. Execução seletiva (interface interativa)
python run_selected_models.py

# 3. Verificar logs após execução
ls execution_logs/

# 4. Analisar resultados
cat execution_logs/execution_summary_*.json
```

## 🎯 Dicas de Uso

1. **Para execução overnight**: Use `run_all_models.py`
2. **Para testes**: Use `run_selected_models.py` com poucos modelos
3. **Monitoring**: Acompanhe os logs em tempo real
4. **Backup**: Mantenha backup dos datasets antes de execução longa
5. **GPU Memory**: Monitore uso de memória GPU durante execução

## 🔍 Monitoramento em Tempo Real

```bash
# Acompanhar logs em tempo real
tail -f execution_logs/model_execution_*.log

# Monitorar GPU
watch -n 1 nvidia-smi

# Verificar uso de CPU/RAM
htop
```

---

**Nota:** Os scripts foram projetados para serem robustos e continuarem a execução mesmo com falhas individuais de modelos. Sempre verifique os logs para análise detalhada dos resultados. 