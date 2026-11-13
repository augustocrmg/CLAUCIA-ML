# Relatório de Data Augmentation para Balanceamento de Dataset de Imagens

## Resumo

O presente relatório documenta a implementação de técnicas de data augmentation para o balanceamento de datasets de imagens médicas, especificamente voltadas para classificação de condições podológicas. O estudo baseia-se em metodologias estabelecidas na literatura científica para aumentar artificialmente o volume de dados disponíveis, visando atingir um número padronizado de 1000 imagens por classe. As técnicas implementadas incluem transformações geométricas, ajustes fotométricos e adição de ruído controlado, seguindo parâmetros específicos documentados em pesquisas anteriores na área de visão computacional aplicada à medicina.

## Introdução

A escassez de dados constitui um dos principais desafios no desenvolvimento de sistemas de aprendizado de máquina para diagnóstico médico por imagem. O desequilíbrio entre classes em datasets médicos pode comprometer significativamente a performance de modelos de classificação, resultando em viés sistemático e redução da capacidade de generalização. Neste contexto, técnicas de data augmentation emergem como uma solução viável para mitigar tais limitações, permitindo a expansão artificial de datasets através de transformações controladas que preservam as características essenciais das imagens originais.

A literatura científica demonstra que o data augmentation pode melhorar substancialmente a robustez de modelos de deep learning, especialmente em cenários com limitações de dados. Estudos recentes em imagens médicas sugerem que transformações específicas, quando aplicadas adequadamente, podem simular variações naturais encontradas em condições reais de captura, contribuindo para a generalização do modelo sem comprometer a integridade diagnóstica das imagens.

## Metodologia

### Definição do Problema

O dataset original apresentava distribuição desigual entre as classes, com variações significativas no número de imagens disponíveis para cada categoria diagnóstica. O objetivo estabelecido consistiu em padronizar todas as classes para exatamente 1000 imagens, mantendo a qualidade e resolução original das imagens.

### Técnicas de Data Augmentation Implementadas

As técnicas de augmentation foram selecionadas com base em estudos prévios que demonstraram eficácia em datasets médicos, particularmente em imagens de extremidades. Cada técnica visa simular condições específicas de captura que podem ocorrer em ambientes clínicos reais.

#### Transformações Geométricas

**Scaling (Redimensionamento)**: Implementou-se variação aleatória de escala entre 50% e 200% do tamanho original, simulando diferentes distâncias entre o paciente e o equipamento de captura. A técnica preserva as dimensões originais da imagem através de cropping centralizado para escalas superiores a 100% e padding com pixels pretos para escalas inferiores. Esta abordagem permite manter a consistência dimensional do dataset enquanto introduz variabilidade de escala.

**Rotação**: Aplicou-se rotação fixa de 90 graus, representando diferentes orientações de captura que podem ocorrer durante o procedimento de aquisição de imagens. A rotação foi limitada a 90 graus para evitar desorientação excessiva que poderia prejudicar a interpretação diagnóstica.

**Espelhamento**: Duas modalidades de espelhamento foram implementadas - horizontal e vertical - simulando diferentes posicionamentos do pé durante a captura. O espelhamento horizontal corresponde à inversão lateral da imagem, enquanto o espelhamento vertical simula inversão de orientação, ambos preservando as características anatômicas essenciais.

#### Transformações Fotométricas

**Ajuste de Brilho**: Variações de brilho foram aplicadas dentro de uma faixa de ±10% do valor original, simulando diferentes condições de iluminação encontradas em ambientes clínicos. Esta faixa foi escolhida para introduzir variabilidade realística sem comprometer a visibilidade de estruturas anatômicas importantes.

#### Transformações Espaciais

**Deslocamento de Pixels**: Implementou-se deslocamento horizontal de 10 pixels para esquerda ou direita, simulando pequenos movimentos do paciente durante a captura. O deslocamento foi limitado a uma magnitude que não compromete a centralização da estrutura anatômica de interesse.

#### Adição de Ruído

**Ruído Salt-and-Pepper**: Adicionou-se ruído impulsivo com densidade de 0.05, simulando interferências pontuais que podem ocorrer durante a captura digital. A densidade foi estabelecida em nível que introduz variabilidade sem mascarar características diagnósticas importantes.

**Ruído Gaussiano**: Implementou-se ruído gaussiano com média zero e variância 0.01, representando interferências aleatórias inerentes ao processo de aquisição digital. Os parâmetros foram calibrados para simular condições realísticas de ruído eletrônico.

### Processo de Balanceamento

O algoritmo de balanceamento opera através da análise inicial da distribuição de classes no dataset original, seguida pelo cálculo do número de imagens adicionais necessárias para cada classe atingir o objetivo de 1000 imagens. As técnicas de augmentation são aplicadas de forma equitativa, distribuindo a geração de imagens adicionais entre as dez técnicas implementadas.

Para cada classe, o número de imagens faltantes é dividido igualmente entre as técnicas disponíveis, com distribuição de resíduos quando a divisão não resulta em números inteiros. A seleção de imagens base para aplicação das técnicas ocorre de forma aleatória, permitindo reutilização de imagens originais quando necessário para atingir o número alvo.

## Resultados Esperados

A implementação das técnicas de data augmentation deve resultar em um dataset balanceado com 1000 imagens por classe, mantendo a diversidade intraclasse através das transformações aplicadas. Espera-se que as técnicas implementadas contribuam para:

**Melhoria na Robustez**: As transformações geométricas podem aumentar a capacidade do modelo de reconhecer padrões independentemente de variações de orientação, escala e posicionamento.

**Simulação de Condições Reais**: Os ajustes fotométricos e adição de ruído podem preparar o modelo para variações de qualidade de imagem encontradas em ambientes clínicos diversos.

**Redução de Overfitting**: O aumento do volume de dados pode contribuir para a redução do sobreajuste, melhorando a capacidade de generalização do modelo.

**Balanceamento de Classes**: A padronização do número de imagens por classe pode reduzir o viés sistemático em favor de classes com maior representação no dataset original.

## Considerações Técnicas

A implementação mantém a integridade das imagens originais através de cópias antes da aplicação das transformações. As técnicas de preservação de dimensões garantem que todas as imagens resultantes mantenham as mesmas especificações técnicas do dataset original, facilitando o processamento subsequente por algoritmos de aprendizado de máquina.

O processo de nomeação das imagens augmentadas inclui sufixos identificadores que permitem rastreabilidade das transformações aplicadas, facilitando análises posteriores sobre a contribuição específica de cada técnica para a performance do modelo.

## Limitações e Considerações Futuras

As técnicas implementadas representam um subconjunto das possibilidades disponíveis em data augmentation. Transformações mais complexas, como deformações elásticas ou variações de cor mais sofisticadas, poderiam ser consideradas em implementações futuras, dependendo das especificidades do domínio de aplicação.

A validação da eficácia das técnicas implementadas requer avaliação empírica através do treinamento e teste de modelos de classificação, comparando performance entre datasets balanceados e não balanceados. Tal validação constitui etapa fundamental para confirmar os benefícios esperados das transformações aplicadas.

## Conclusão

O processo de data augmentation implementado segue metodologias estabelecidas na literatura científica, aplicando transformações controladas que visam aumentar a diversidade do dataset mantendo a integridade diagnóstica das imagens. A padronização para 1000 imagens por classe através de dez técnicas distintas de augmentation pode contribuir significativamente para o desenvolvimento de modelos de classificação mais robustos e equilibrados.

A implementação técnica prioriza a preservação das características essenciais das imagens originais, aplicando transformações dentro de parâmetros que simulam variações naturalísticas encontradas em ambientes clínicos reais. Os resultados esperados incluem melhoria na capacidade de generalização e redução de viés sistemático, contribuindo para o desenvolvimento de sistemas de diagnóstico assistido por computador mais confiáveis e eficazes.

---

**Nota Técnica**: Este relatório documenta a implementação de técnicas de data augmentation conforme especificado no script `balanceamento_dataset_script_v2.py`. Os parâmetros utilizados baseiam-se em práticas estabelecidas na literatura científica para processamento de imagens médicas, com adaptações específicas para o domínio de aplicação em questão. 