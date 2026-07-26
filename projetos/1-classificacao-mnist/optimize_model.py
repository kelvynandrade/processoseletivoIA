import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
# ---------------------------------------------------------------------------

def main():
    model_path = "model.h5"
    
    if not os.path.exists(model_path):
        print(f"Erro: O arquivo '{model_path}' não foi encontrado. Execute o 'train_model.py' primeiro.")
        return

    print("Carregando o modelo 'model.h5'...")
    # 1. Carregar o modelo treinado
    model = tf.keras.models.load_model(model_path)

    print("Configurando a conversão para TensorFlow Lite...")
    # 2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 3. Aplicar técnica de otimização (Dynamic Range Quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Realizando a conversão (isso pode levar alguns segundos)...")
    tflite_model = converter.convert()

    # 4. Salvar o resultado como "model.tflite"
    tflite_model_path = "model.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"Conversão concluída com sucesso! Modelo otimizado salvo como '{tflite_model_path}'.")

if __name__ == "__main__":
    main()