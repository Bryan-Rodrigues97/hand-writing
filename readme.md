# Handwriting Recognition com TensorFlow

Bem-vindo ao projeto **Handwriting Recognition**, um script desenvolvido para treinar uma rede neural convolucional para reconhecimento de escrita à mão utilizando o dataset EMNIST.

## Requisitos

Certifique-se de ter o Python 3.8 ou superior instalado. Os pacotes necessários estão listados no arquivo `requirements.txt`. Para instalá-los, execute:

```bash
pip install -r requirements.txt
```

## Arquivos do Projeto

- `handwriting.py`: Script principal que carrega o dataset, realiza o pré-processamento, treina a rede neural e salva o modelo.
- `requirements.txt`: Lista de dependências necessárias para executar o script.
- **Dataset EMNIST**: O script utiliza os seguintes arquivos do dataset EMNIST, que devem ser colocados no diretório `emnist/`:
  - `emnist-byclass-train-images-idx3-ubyte`
  - `emnist-byclass-train-labels-idx1-ubyte`
  - `emnist-byclass-test-images-idx3-ubyte`
  - `emnist-byclass-test-labels-idx1-ubyte`

## Funcionalidades

1. **Carregamento do Dataset:**
   - Utiliza o formato IDX para carregar as imagens e os rótulos de treino e teste.
2. **Pré-processamento das Imagens:**
   - Rotaciona as imagens em 270 graus.
   - Realiza um flip horizontal.
   - Converte os rótulos para codificação one-hot.
3. **Modelo de Rede Neural Convolucional:**
   - Arquitetura com três camadas convolucionais, pooling, dropout e uma camada densa final para classificação.
4. **Treinamento e Avaliação:**
   - Treina o modelo por 10 épocas usando o otimizador Adam.
   - Avalia o desempenho nos dados de teste.
5. **Salvamento do Modelo:**
   - O modelo treinado é salvo no arquivo `emnist.h5`.

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/Bryan-Rodrigues97/hand-writing.git
   cd hand-writing
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Certifique-se de que os arquivos do dataset EMNIST estejam na pasta `emnist/`.

4. Execute o script:
   ```bash
   python handwriting.py
   ```

## Estrutura do Dataset

Os arquivos IDX do dataset EMNIST contêm:

- Imagens de dimensão 28x28 (grayscale), representando caracteres manuscritos.
- Rótulos correspondentes aos 62 possíveis caracteres (26 letras maiúsculas, 26 letras minúsculas e 10 dígitos).

## Referências

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
