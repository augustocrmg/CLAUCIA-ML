#!/usr/bin/env python3
"""
Script para testar o modelo H5 com uma imagem aleatória do dataset v4 balanceado 1000.

Este script:
1. Carrega o modelo H5 especificado
2. Seleciona uma imagem aleatória do dataset
3. Realiza a predição e mostra os resultados
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
import glob
import time
import argparse
from pathlib import Path

# Configuração para evitar avisos e erros comuns
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz verbosidade do TensorFlow

def recreate_mobilenet_v3_model(num_classes=7, image_size=(224, 224)):
    """
    Recria a arquitetura do modelo MobileNetV3 para carregar os pesos posteriormente
    """
    # Cria a base do modelo MobileNetV3Large
    base_model = keras.applications.MobileNetV3Large(
        include_top=False,
        weights="imagenet",  # Inicializa com pesos do ImageNet
        input_shape=(*image_size, 3),
        pooling="avg"
    )
    
    # Cria o modelo completo com a mesma arquitetura do original
    inputs = keras.Input(shape=(*image_size, 3))
    x = keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    
    return model

def load_h5_model(model_path, num_classes=7, image_size=(224, 224)):
    """Carrega um modelo H5 e retorna o modelo"""
    try:
        print(f"🔄 Tentando carregar modelo: {model_path}")
        
        # Abordagem 1: Tentar carregar diretamente (pode falhar)
        try:
            custom_objects = {
                'preprocess_input': tf.keras.applications.mobilenet_v3.preprocess_input
            }
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects
            )
            print("✅ Modelo carregado com abordagem direta")
            return model
        except Exception as e:
            print(f"⚠️ Falha na abordagem direta: {str(e)}")
            pass
        
        # Abordagem 2: Recriar a arquitetura e carregar apenas os pesos
        print("🔄 Tentando abordagem alternativa: recriando arquitetura...")
        model = recreate_mobilenet_v3_model(num_classes, image_size)
        
        try:
            model.load_weights(model_path)
            print("✅ Pesos carregados com sucesso")
        except Exception as e:
            print(f"⚠️ Erro ao carregar pesos: {str(e)}")
            
            # Abordagem 3: Usar o TFLite como alternativa
            tflite_path = model_path.replace('.h5', '.tflite')
            if os.path.exists(tflite_path):
                print(f"🔄 Tentando usar modelo TFLite como alternativa: {tflite_path}")
                interpreter = tf.lite.Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()
                print("✅ Modelo TFLite carregado como alternativa")
                return {"type": "tflite", "interpreter": interpreter}
            else:
                print("❌ Modelo TFLite alternativo não encontrado")
                return None
        
        print(f"✅ Modelo recriado e pesos carregados com sucesso: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {str(e)}")
        return None

def get_model_details(model):
    """Extrai detalhes do modelo"""
    try:
        # Verifica se é um modelo TFLite
        if isinstance(model, dict) and model.get("type") == "tflite":
            interpreter = model["interpreter"]
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            details = {
                'input_shape': input_details[0]['shape'],
                'output_shape': output_details[0]['shape'],
                'model_type': 'tflite'
            }
        else:
            # Modelo Keras normal
            input_shape = model.input_shape
            output_shape = model.output_shape
            
            details = {
                'input_shape': input_shape,
                'output_shape': output_shape,
                'model_type': 'keras'
            }
        
        return details
    except Exception as e:
        print(f"❌ Erro ao obter detalhes do modelo: {str(e)}")
        # Retorna valores padrão para MobileNetV3
        return {
            'input_shape': (None, 224, 224, 3),
            'output_shape': (None, 7),
            'model_type': 'unknown'
        }

def get_random_image(dataset_path, target_shape, class_names):
    """Seleciona uma imagem aleatória do dataset e a classe correspondente"""
    # Lista todas as classes (diretórios)
    class_dirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not class_dirs:
        print(f"❌ Nenhuma classe encontrada em {dataset_path}")
        return None, None, None, None
    
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
        return None, None, None, None
    
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
        
        # Não normalizar a imagem para modelos que incluem pré-processamento
        
        return img_array, class_name, class_idx, img
    except Exception as e:
        print(f"❌ Erro ao carregar imagem: {str(e)}")
        return None, None, None, None

def predict_with_model(model, img_array, model_details):
    """Realiza a predição com o modelo"""
    # Prepara a entrada (adiciona dimensão de batch)
    input_data = np.expand_dims(img_array, axis=0)
    
    try:
        # Verifica o tipo de modelo
        if model_details.get('model_type') == 'tflite':
            interpreter = model["interpreter"]
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Executa a inferência com TFLite
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            inference_time = time.time() - start_time
        else:
            # Executa a inferência com modelo Keras
            start_time = time.time()
            predictions = model.predict(input_data, verbose=0)
            inference_time = time.time() - start_time
        
        print(f"⏱️ Tempo de inferência: {inference_time*1000:.2f}ms")
        
        return predictions, inference_time
    except Exception as e:
        print(f"❌ Erro na predição: {str(e)}")
        # Retorna um array vazio em caso de erro
        return np.zeros((1, 7)), 0

def main():
    parser = argparse.ArgumentParser(description='Testa modelo com uma imagem aleatória')
    parser.add_argument('--model_path', type=str, 
                        default='model_results_colab/models_colab_dataset_wagner_v4_balanceado_1000/mobilenet_v3_224/mobilenet_v3_224.h5',
                        help='Caminho para o modelo H5')
    parser.add_argument('--dataset_path', type=str, 
                        default='datasets/dataset_wagner_v4_balanceado_1000',
                        help='Caminho para o dataset')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Número de amostras aleatórias para testar')
    parser.add_argument('--use_tflite', action='store_true',
                        help='Usar o modelo TFLite em vez do H5')
    
    args = parser.parse_args()
    
    print("🧪 Testando Modelo com Imagem Aleatória")
    print("=" * 50)
    
    # Se a opção de usar TFLite estiver ativada, ajusta o caminho do modelo
    if args.use_tflite:
        model_path = args.model_path.replace('.h5', '.tflite')
        print(f"🔄 Usando modelo TFLite: {model_path}")
    else:
        model_path = args.model_path
    
    # Verifica se o modelo existe
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    # Verifica se o dataset existe
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset não encontrado: {args.dataset_path}")
        return
    
    # Carrega o modelo
    if args.use_tflite:
        # Carrega o modelo TFLite diretamente
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            model = {"type": "tflite", "interpreter": interpreter}
            print(f"✅ Modelo TFLite carregado com sucesso: {model_path}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo TFLite: {str(e)}")
            return
    else:
        # Carrega o modelo H5
        model = load_h5_model(model_path)
    
    if model is None:
        return
    
    # Obtém detalhes do modelo
    model_details = get_model_details(model)
    
    print(f"\n📋 Detalhes do modelo:")
    print(f"   - Tipo: {model_details.get('model_type', 'desconhecido')}")
    print(f"   - Input shape: {model_details['input_shape']}")
    print(f"   - Output shape: {model_details['output_shape']}")
    
    # Obtém as classes do dataset
    class_names = sorted([d for d in os.listdir(args.dataset_path) 
                         if os.path.isdir(os.path.join(args.dataset_path, d))])
    print(f"\n🏷️ Classes encontradas: {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    # Estatísticas para múltiplas amostras
    correct_predictions = 0
    total_samples = args.num_samples
    total_time = 0
    
    for sample_idx in range(total_samples):
        if total_samples > 1:
            print(f"\n\n📊 Amostra {sample_idx+1}/{total_samples}")
            print("-" * 30)
        
        # Seleciona uma imagem aleatória
        target_shape = model_details['input_shape'][1:]  # Remove a dimensão do batch
        img_array, true_class_name, true_class_idx, img = get_random_image(args.dataset_path, target_shape, class_names)
        
        if img_array is None:
            continue
        
        # Realiza a predição
        print("\n🔮 Realizando predição...")
        predictions, inference_time = predict_with_model(model, img_array, model_details)
        total_time += inference_time
        
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
        
        # Avalia se a predição está correta
        if pred_class_idx == true_class_idx:
            print("\n✅ Predição CORRETA!")
            correct_predictions += 1
        else:
            print("\n❌ Predição INCORRETA!")
        
        # Visualiza a imagem com a predição (apenas para a primeira ou única amostra)
        if sample_idx == 0 or total_samples == 1:
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f"Predição: {pred_class_name} ({confidence:.4f})\nReal: {true_class_name}")
            plt.axis('off')
            
            # Salva a imagem com os resultados
            result_filename = 'model_prediction_result.png'
            plt.savefig(result_filename)
            print(f"\n✅ Visualização salva em: {result_filename}")
    
    # Mostra estatísticas finais se testou múltiplas amostras
    if total_samples > 1:
        accuracy = correct_predictions / total_samples
        avg_time = total_time / total_samples
        
        print("\n" + "=" * 50)
        print("📊 Estatísticas Finais:")
        print(f"   - Total de amostras: {total_samples}")
        print(f"   - Predições corretas: {correct_predictions}")
        print(f"   - Acurácia: {accuracy:.2%}")
        print(f"   - Tempo médio de inferência: {avg_time*1000:.2f}ms")

if __name__ == "__main__":
    main() 