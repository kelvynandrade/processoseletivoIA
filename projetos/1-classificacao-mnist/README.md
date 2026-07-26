## Relatório Técnico - Classificador de Dígitos com CNN (Edge AI)
**Identificação do Candidato**
**Nome completo**: Kelvyn César Ferreira de Andrade
**GitHub**: https://github.com/kelvynandrade


## Visão Geral da Solução

Este projeto implementa um classificador de dígitos manuscritos (0 a 9) utilizando uma Rede Neural Convolucional (CNN) treinada sobre o dataset MNIST. Após o treinamento, o modelo é otimizado via quantização e convertido para o formato TensorFlow Lite, permitindo inferência leve e independente do framework de treinamento — simulando o comportamento esperado em microcontroladores e dispositivos móveis (Edge AI).

## Arquitetura do Modelo

A arquitetura da CNN, implementada em train_model.py, foi projetada para equilibrar boa capacidade de extração de padrões espaciais em imagens de 28x28 pixels com leveza computacional adequada para ambientes restritos:

-Blocos Convolucionais: 3 blocos sequenciais compostos por camadas Conv2D (filtros 3x3, ativação ReLU), seguidas por BatchNormalization (estabiliza o gradiente e acelera a convergência) e MaxPooling2D (redução espacial dos mapas de características).

-Camadas de Classificação: Aplanamento dos dados (Flatten) conectado a uma camada densa (Dense) intermediária com ativação ReLU.

-Regularização: Camada de Dropout (taxa de 0.5) antes da camada de saída, para mitigar overfitting. A camada de saída é uma Dense de 10 neurônios com ativação softmax, para a classificação multiclasse dos dígitos.

-Estratégia de Validação e Early Stopping: Split explícito de validação (20% do conjunto de treino) e callback EarlyStopping monitorando val_loss com tolerância (patience), garantindo interrupção precoce do treinamento e preservação dos pesos ótimos.

## Bibliotecas e Ferramentas Utilizadas

-TensorFlow / Keras (v2.15.0): Framework principal para construção, compilação e treinamento da CNN, salvamento do modelo em HDF5 (model.h5) e conversão para o formato de borda.

-NumPy: Manipulação eficiente de matrizes multidimensionais e normalização dos pixels do MNIST para a faixa contínua entre 0.0 e 1.0.

-Gerenciador uv & Python (v3.11): Gerenciamento de pacotes e isolamento de ambiente virtual, garantindo estabilidade, reprodutibilidade e compatibilidade de dependências.


## Decisões Técnicas Relevantes

-Quantização de Faixa Dinâmica (Dynamic Range Quantization): implementada em optimize_model.py através do conversor nativo tf.lite.TFLiteConverter. O algoritmo converte estaticamente os pesos das camadas convolucionais e densas de float32 para int8, enquanto as ativações permanecem em ponto flutuante e são quantizadas dinamicamente durante a execução. Isso reduz o tamanho do binário e otimiza a velocidade de inferência na CPU, com perda de acurácia estatisticamente imperceptível.

-Isolamento de Versão (Python 3.14 vs 3.11): devido a incompatibilidades do ecossistema TensorFlow com versões recém-lançadas do interpretador no host, o uv foi usado para instanciar um ambiente virtual controlado com Python 3.11.

-Compatibilidade de Desserialização (Keras 2 vs Keras 3): o validador da esteira de CI do repositório remoto exigia um artefato .h5 compatível com a interface legada. Após identificar falhas de metadados do Keras 3, a versão do TensorFlow foi fixada em 2.15.0, alinhando o ambiente local de geração com o runner de avaliação.


## Resultados Obtidos

-Acurácia de Validação Final: 98,72% (0.9872), obtida ao término do treinamento.

-Tamanho do Modelo Original (model.h5): ~1.2 MB

-Tamanho do Modelo Otimizado (model.tflite): ~320 KB — redução substancial de espaço para implantação em Edge Devices.

-Validação em Borda (run_inference.py): a inferência pontual utilizando estritamente o tf.lite.Interpreter comprovou que o artefato otimizado executa com total independência do framework pesado de treinamento.

**Exemplo de saída obtida no terminal durante a execução do run_inference.py sobre o conjunto de testes:**

Amostra 1 -> Classe Real: 7 | Classe Predita: 7 (ACERTO)

Amostra 2 -> Classe Real: 2 | Classe Predita: 2 (ACERTO)

Amostra 3 -> Classe Real: 1 | Classe Predita: 1 (ACERTO)

Amostra 4 -> Classe Real: 0 | Classe Predita: 0 (ACERTO)

Amostra 5 -> Classe Real: 4 | Classe Predita: 4 (ACERTO)

O modelo quantizado demonstrou boa robustez e precisão nas amostras avaliadas individualmente — a conversão para Edge AI não comprometeu a capacidade de distinção dos dígitos manuscritos.

## Comentários Adicionais

A experiência prévia com desenvolvimento de lógicas estruturadas e prototipagem de circuitos eletrônicos virtuais colaborou para a estruturação rápida deste ambiente. A transição de conceitos habitualmente aplicados em linguagens como C ou Java para a elaboração deste pipeline em Python ocorreu de forma fluida, reforçando a robustez do ecossistema Python/TensorFlow para aplicações de IoT e Edge AI.
