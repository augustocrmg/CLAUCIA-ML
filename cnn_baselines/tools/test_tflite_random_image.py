#!/usr/bin/env python3
"""
Script para testar o modelo TFLite com uma imagem aleatória do dataset v4 balanceado 1000.

Este script:
1. Carrega o modelo TFLite especificado
2. Seleciona uma imagem aleatória do dataset
3. Realiza a predição e mostra os resultados
"""

import os
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import glob
import time
import argparse
from pathlib import Path

def load_tflite_model(model_path):
    """Carrega um modelo TFLite e retorna o interpretador"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"✅ Modelo carregado com sucesso: {model_path}")
        return interpreter
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {str(e)}")
        return None

def get_model_details(interpreter):
    """Extrai detalhes do modelo TFLite"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    details = {
        'input_shape': input_details[0]['shape'],
        'input_dtype': input_details[0]['dtype'],
        'output_shape': output_details[0]['shape'],
        'output_dtype': output_details[0]['dtype'],
    }
    
    return details, input_details, output_details

def get_random_image(dataset_path, target_shape, class_names):
    """Seleciona uma imagem aleatória do dataset e a classe correspondente"""
    # Lista todas as classes (diretórios)
    class_dirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not class_dirs:
        print(f"❌ Nenhuma classe encontrada em {dataset_path}")
        return None, None, None
    
    # Seleciona uma classe aleatória
    class_name = random.choice(class_dirs)
    class_path = os.path.join(dataset_path, class_name)
    # Obtém o índice da classe na lista ordenada de classes
    class_idx = class_names.index(class_name)
    
    # Encontra todas as imagens na classe
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(class_path, ext)))
        image_files.extend(glob.glob(os.path.join(class_path, ext.upper())))
    
    if not image_files:
        print(f"❌ Nenhuma imagem encontrada na classe {class_name}")
        return None, None, None
    
    # Seleciona uma imagem aleatória
    img_file = random.choice(image_files)
    print(f"📷 Imagem selecionada: {img_file}")
    print(f"🏷️ Classe: {class_name} (índice: {class_idx})")
    
    try:
        # Carrega e redimensiona a imagem
        img = tf.keras.preprocessing.image.load_img(
            img_file, target_size=target_shape[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.astype(np.float32) / 255.0  # Normaliza
        
        return img_array, class_name, class_idx, img
    except Exception as e:
        print(f"❌ Erro ao carregar imagem: {str(e)}")
        return None, None, None, None

def predict_with_tflite(interpreter, img_array, input_details, output_details):
    """Realiza a predição com o modelo TFLite"""
    # Prepara a entrada (adiciona dimensão de batch)
    input_data = np.expand_dims(img_array, axis=0)
    
    # Verifica e ajusta o shape se necessário
    if input_data.shape != tuple(input_details[0]['shape']):
        print(f"⚠️ Ajustando shape da entrada: {input_data.shape} -> {input_details[0]['shape']}")
        # Tenta redimensionar para o shape esperado
        input_data = np.resize(input_data, input_details[0]['shape'])
    
    # Executa a inferência
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Obtém o resultado
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"⏱️ Tempo de inferência: {inference_time*1000:.2f}ms")
    
    return output_data, inference_time

def main():
    parser = argparse.ArgumentParser(description='Testa modelo TFLite com uma imagem aleatória')
    parser.add_argument('--model_path', type=str, 
                        default='model_results_colab/models_colab_dataset_wagner_v4_balanceado_1000/mobilenet_v3_224/mobilenet_v3_224.tflite',
                        help='Caminho para o modelo TFLite')
    parser.add_argument('--dataset_path', type=str, 
                        default='datasets/dataset_wagner_v4_balanceado_1000',
                        help='Caminho para o dataset')
    
    args = parser.parse_args()
    
    print("🧪 Testando Modelo TFLite com Imagem Aleatória")
    print("=" * 50)
    
    # Verifica se o modelo existe
    if not os.path.exists(args.model_path):
        print(f"❌ Modelo não encontrado: {args.model_path}")
        return
    
    # Verifica se o dataset existe
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset não encontrado: {args.dataset_path}")
        return
    
    # Carrega o modelo
    interpreter = load_tflite_model(args.model_path)
    if interpreter is None:
        return
    
    # Obtém detalhes do modelo
    model_details, input_details, output_details = get_model_details(interpreter)
    
    print(f"\n📋 Detalhes do modelo:")
    print(f"   - Input shape: {model_details['input_shape']}")
    print(f"   - Output shape: {model_details['output_shape']}")
    
    # Obtém as classes do dataset
    class_names = sorted([d for d in os.listdir(args.dataset_path) 
                         if os.path.isdir(os.path.join(args.dataset_path, d))])
    print(f"\n🏷️ Classes encontradas: {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    # Seleciona uma imagem aleatória
    target_shape = model_details['input_shape'][1:]  # Remove a dimensão do batch
    img_array, true_class_name, true_class_idx, img = get_random_image(args.dataset_path, target_shape, class_names)
    
    if img_array is None:
        return
    
    # Realiza a predição
    print("\n🔮 Realizando predição...")
    predictions, inference_time = predict_with_tflite(interpreter, img_array, input_details, output_details)
    
    # Analisa os resultados
    if len(predictions.shape) > 1:
        probs = predictions[0]  # Remove dimensão do batch se necessário
    else:
        probs = predictions
    
    # Obtém a classe com maior probabilidade
    pred_class_idx = np.argmax(probs)
    confidence = probs[pred_class_idx]
    pred_class_name = class_names[pred_class_idx] if pred_class_idx < len(class_names) else "Desconhecido"
    
    print("\n🎯 Resultados da predição:")
    print(f"   - Classe predita: {pred_class_name} (índice: {pred_class_idx})")
    print(f"   - Confiança: {confidence:.4f}")
    print(f"   - Classe real: {true_class_name} (índice: {true_class_idx})")
    
    # Mostra todas as probabilidades
    print("\n📊 Probabilidades para todas as classes:")
    for i, class_name in enumerate(class_names):
        if i < len(probs):
            print(f"   {class_name}: {probs[i]:.4f}")
    
    # Visualiza a imagem com a predição
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Predição: {pred_class_name} ({confidence:.4f})\nReal: {true_class_name}")
    plt.axis('off')
    
    # Salva a imagem com os resultados
    result_filename = 'tflite_prediction_result.png'
    plt.savefig(result_filename)
    print(f"\n✅ Visualização salva em: {result_filename}")
    
    # Avalia se a predição está correta
    if pred_class_idx == true_class_idx:
        print("\n✅ Predição CORRETA!")
    else:
        print("\n❌ Predição INCORRETA!")

if __name__ == "__main__":
    main() 