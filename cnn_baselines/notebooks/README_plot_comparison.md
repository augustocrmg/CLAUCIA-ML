# Script de Comparação de Datasets

## Descrição

O script `plot_dataset_comparison.py` foi criado para comparar o dataset original com sua versão balanceada através de data augmentation, visualizando a evolução usando barras empilhadas.

## Funcionalidades

- **Comparação automática**: Identifica e compara apenas as classes que existem em ambos os datasets
- **Barras empilhadas**: Mostra visualmente as imagens originais (base) e as imagens geradas por data augmentation (topo)
- **Estatísticas detalhadas**: Apresenta informações completas sobre o processo de augmentation
- **Padrão visual consistente**: Mantém o mesmo estilo visual do script original

## Como usar

```bash
python notebooks/plot_dataset_comparison.py
```

## Datasets comparados

- **Dataset Original**: `datasets/dataset_wagner_v2` (Dataset de Wagner Curado)
- **Dataset Balanceado**: `datasets/dataset_wagner_v2_wo_grade_5_balanceado` (Dataset de Wagner Curado e Balanceado)

## Saída

O script gera:
1. **Estatísticas no terminal**: Mostra contagem detalhada por classe e fator de multiplicação
2. **Plot visual**: Arquivo `comparacao_datasets_wagner.png` com barras empilhadas
3. **Análise de data augmentation**: Mostra quantas imagens foram geradas por classe

## Exemplo de saída

```
======================================================================
COMPARAÇÃO: DATASET ORIGINAL vs DATASET BALANCEADO
======================================================================

Classes comuns encontradas: 6
Classes: Normal(Healthy skin), Grade 0, Grade 1, Grade 2, Grade 3, Grade 4

TOTAL Dataset de Wagner Curado (Original): 1031 imagens
TOTAL Dataset de Wagner Curado e Balanceado: 6000 imagens
IMAGENS ADICIONADAS (Data Augmentation): 4969 imagens
FATOR DE MULTIPLICAÇÃO: 5.8x

DETALHES POR CLASSE:
-----------------------------------------------------------------
Classe                    | Original | Augmented  | Total    | Fator
-----------------------------------------------------------------
Normal(Healthy skin)      |      559 |        441 |     1000 |  1.8x
Grade 0                   |       54 |        946 |     1000 | 18.5x
Grade 1                   |      144 |        856 |     1000 |  6.9x
Grade 2                   |      152 |        848 |     1000 |  6.6x
Grade 3                   |       62 |        938 |     1000 | 16.1x
Grade 4                   |       60 |        940 |     1000 | 16.7x
```

## Características do Plot

- **Barra inferior (verde escuro)**: Imagens originais do dataset curado
- **Barra superior (verde claro)**: Imagens geradas por data augmentation
- **Legenda**: Identifica claramente cada componente
- **Números**: Mostra contagem de imagens originais, augmentadas e total
- **Grid**: Facilita a leitura dos valores
- **Rotação**: Labels do eixo X rotacionados para melhor legibilidade

## Interpretação dos Resultados

- **Fator de multiplicação**: Indica quantas vezes cada classe foi ampliada
- **Classes mais balanceadas**: Normal(Healthy skin) teve menor augmentation (já tinha mais imagens)
- **Classes menos representadas**: Grade 0, 3 e 4 tiveram maior augmentation (fatores >15x)
- **Resultado final**: Todas as classes ficaram com exatamente 1000 imagens

## Notas

- O script automaticamente exclui classes que não existem em ambos os datasets
- A classe "Grade 5" é automaticamente excluída pois só existe no dataset original
- O arquivo de saída é salvo como `comparacao_datasets_wagner.png`
- As barras empilhadas mostram claramente a contribuição de cada fonte (original vs augmented) 