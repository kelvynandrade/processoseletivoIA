import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui


from tensorflow.keras import layers, models, callbacks
import numpy as np

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
# ---------------------------------------------------------------------------

def main():
    print("Carregando e pré-processando os dados...")
    # 1 e 2. Carregar o dataset MNIST via tf.keras.datasets.mnist e normalizar [0, 1]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Ajustar o shape para (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print("Construindo a arquitetura do modelo...")
    # 4. Construir uma CNN com 3 blocos Conv2D + BatchNormalization + MaxPooling2D
    model = models.Sequential([
        # Bloco 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Bloco 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Bloco 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Achatar e aplicar Dropout antes da saída
        layers.Flatten(),
        layers.Dropout(0.5), # Regularização
        layers.Dense(10, activation='softmax') # 10 classes
    ])

    # Compilar o modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Iniciando o treinamento...")
    # 5. Treinar com EarlyStopping monitorando a perda de validação
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )

    # 3. Separar um conjunto de validação (validation_split = 0.2)
    history = model.fit(
        x_train, y_train, 
        epochs=15, 
        validation_split=0.2, 
        callbacks=[early_stopping]
    )

    # 6. Exibir a acurácia de validação final no terminal
    print("\nAvaliando o modelo no conjunto de teste...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n--- RESULTADO FINAL ---")
    print(f"Acurácia de validação final (no teste): {test_acc:.4f}")

    # 7. Salvar o modelo treinado como "model.h5"
    model.save("model.h5")
    print("Modelo salvo com sucesso como 'model.h5'.")

if __name__ == "__main__":
    main()