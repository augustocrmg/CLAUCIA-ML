#!/usr/bin/env python3
"""
Script para testar modelos TensorFlow Lite (.tflite)

Este script carrega modelos .tflite e testa com dados de amostra para verificar
se estão funcionando corretamente após a conversão.

Uso:
    python test_tflite_models.py
    python test_tflite_models.py --model_path /caminho/para/modelo.tflite
    python test_tflite_models.py --test_data_path /caminho/para/dados
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import glob
import time
from pathlib import Path
import json

def find_tflite_models(root_dir="."):
    """Encontra todos os arquivos .tflite no diretório e subdiretórios"""
    tflite_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.tflite'):
                full_path = os.path.join(root, file)
                # Verifica se é um arquivo válido e não vazio
                if os.path.getsize(full_path) > 1000:  # Pelo menos 1KB
                    tflite_files.append(full_path)
    return tflite_files

def load_tflite_model(model_path):
    """Carrega um modelo TFLite e retorna o interpretador"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
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
        'input_name': input_details[0]['name'],
        'output_name': output_details[0]['name']
    }
    
    return details, input_details, output_details

def create_test_data(input_shape, input_dtype, num_samples=5):
    """Cria dados de teste sintéticos baseados na forma de entrada do modelo"""
    
    # Remove a dimensão do batch se presente
    if len(input_shape) == 4 and input_shape[0] == 1:
        test_shape = (num_samples,) + tuple(input_shape[1:])
    else:
        test_shape = (num_samples,) + tuple(input_shape)
    
    if input_dtype == np.float32:
        # Para modelos de imagem, cria dados normalizados entre 0 e 1
        if len(input_shape) >= 3:  # Provavelmente imagem
            test_data = np.random.uniform(0.0, 1.0, test_shape).astype(np.float32)
        else:
            test_data = np.random.randn(*test_shape).astype(np.float32)
    elif input_dtype == np.uint8:
        # Para imagens uint8 (0-255)
        test_data = np.random.randint(0, 256, test_shape, dtype=np.uint8)
    else:
        # Tipo genérico
        test_data = np.random.randn(*test_shape).astype(input_dtype)
    
    return test_data

def load_real_test_images(dataset_paths, target_shape, num_samples=5):
    """Tenta carregar imagens reais de um dataset para teste"""
    test_images = []
    test_labels = []
    
    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            continue
            
        # Procura por estrutura de classes
        class_dirs = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))]
        
        if not class_dirs:
            continue
            
        samples_per_class = max(1, num_samples // len(class_dirs))
        
        for class_idx, class_name in enumerate(class_dirs[:10]):  # Máximo 10 classes
            class_path = os.path.join(dataset_path, class_name)
            
            # Encontra imagens na classe
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(class_path, ext)))
                image_files.extend(glob.glob(os.path.join(class_path, ext.upper())))
            
            # Carrega algumas imagens
            for img_file in image_files[:samples_per_class]:
                try:
                    # Carrega e redimensiona a imagem
                    img = tf.keras.preprocessing.image.load_img(
                        img_file, target_size=target_shape[:2]
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array.astype(np.float32) / 255.0  # Normaliza
                    
                    test_images.append(img_array)
                    test_labels.append(class_idx)
                    
                    if len(test_images) >= num_samples:
                        break
                        
                except Exception as e:
                    continue
            
            if len(test_images) >= num_samples:
                break
    
    if test_images:
        return np.array(test_images), np.array(test_labels)
    return None, None

def test_model_inference(interpreter, test_data, input_details, output_details):
    """Testa a inferência do modelo com dados de teste"""
    results = []
    times = []
    
    print(f"🧪 Testando com {len(test_data)} amostras...")
    
    for i, sample in enumerate(test_data):
        try:
            # Prepara entrada
            if len(sample.shape) == len(input_details[0]['shape']) - 1:
                input_data = np.expand_dims(sample, axis=0)
            else:
                input_data = sample.reshape(input_details[0]['shape'])
            
            # Executa inferência
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            inference_time = time.time() - start_time
            
            # Obtém resultado
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            results.append(output_data.copy())
            times.append(inference_time)
            
            # Mostra progresso
            if (i + 1) % max(1, len(test_data) // 5) == 0:
                print(f"   ✓ Amostra {i+1}/{len(test_data)} - {inference_time*1000:.1f}ms")
                
        except Exception as e:
            print(f"   ❌ Erro na amostra {i+1}: {str(e)}")
            continue
    
    return results, times

def analyze_results(results, times, model_details):
    """Analisa os resultados dos testes"""
    if not results:
        print("❌ Nenhum resultado válido obtido")
        return
    
    print(f"\n📊 Análise dos Resultados:")
    print(f"   - Amostras processadas: {len(results)}")
    print(f"   - Tempo médio de inferência: {np.mean(times)*1000:.2f}ms")
    print(f"   - Tempo mínimo: {np.min(times)*1000:.2f}ms")
    print(f"   - Tempo máximo: {np.max(times)*1000:.2f}ms")
    print(f"   - FPS estimado: {1/np.mean(times):.1f}")
    
    # Analisa as saídas
    first_result = results[0]
    print(f"   - Shape da saída: {first_result.shape}")
    print(f"   - Tipo da saída: {first_result.dtype}")
    
    # Se parece ser classificação (softmax)
    if len(first_result.shape) >= 2 and first_result.shape[-1] > 1:
        num_classes = first_result.shape[-1]
        print(f"   - Número de classes detectadas: {num_classes}")
        
        # Mostra algumas predições
        print(f"\n🎯 Amostras de predições:")
        for i, result in enumerate(results[:3]):
            if len(result.shape) > 1:
                probs = result[0]  # Remove batch dimension if necessary
            else:
                probs = result
            
            top_class = np.argmax(probs)
            confidence = probs[top_class]
            
            print(f"   Amostra {i+1}: Classe {top_class} (confiança: {confidence:.3f})")
    
    # Verifica consistência
    all_shapes = [r.shape for r in results]
    if all(shape == all_shapes[0] for shape in all_shapes):
        print("   ✅ Todas as saídas têm shapes consistentes")
    else:
        print("   ⚠️  Shapes das saídas são inconsistentes")

def test_tflite_model(model_path, test_data_path=None):
    """Função principal para testar um modelo TFLite"""
    print(f"\n🤖 Testando modelo: {os.path.basename(model_path)}")
    print("=" * 60)
    
    # Carrega o modelo
    interpreter = load_tflite_model(model_path)
    if interpreter is None:
        return False
    
    # Obtém detalhes do modelo
    model_details, input_details, output_details = get_model_details(interpreter)
    
    print(f"📋 Detalhes do modelo:")
    print(f"   - Input: {model_details['input_shape']} ({model_details['input_dtype']})")
    print(f"   - Output: {model_details['output_shape']} ({model_details['output_dtype']})")
    
    # Tenta carregar dados reais primeiro
    test_data = None
    test_labels = None
    
    if test_data_path:
        print(f"\n📁 Procurando dados de teste em: {test_data_path}")
        target_shape = model_details['input_shape'][1:]  # Remove batch dimension
        test_data, test_labels = load_real_test_images([test_data_path], target_shape)
    
    # Se não encontrou dados reais, procura datasets padrão
    if test_data is None:
        dataset_paths = [
            "datasets/dataset_wagner_balanceado_3",
            "datasets/dataset_wagner_balanceado_2", 
            "../datasets/dataset_wagner_balanceado_3",
            "../datasets/dataset_wagner_balanceado_2"
        ]
        
        print(f"\n📁 Procurando datasets padrão...")
        target_shape = model_details['input_shape'][1:]
        test_data, test_labels = load_real_test_images(dataset_paths, target_shape)
        
        if test_data is not None:
            print(f"   ✅ Carregadas {len(test_data)} imagens reais do dataset")
    
    # Se ainda não tem dados, cria sintéticos
    if test_data is None:
        print(f"\n🎲 Criando dados de teste sintéticos...")
        test_data = create_test_data(model_details['input_shape'], model_details['input_dtype'])
        print(f"   ✅ Criadas {len(test_data)} amostras sintéticas")
    
    # Executa testes
    results, times = test_model_inference(interpreter, test_data, input_details, output_details)
    
    # Analisa resultados
    analyze_results(results, times, model_details)
    
    return len(results) > 0

def main():
    parser = argparse.ArgumentParser(description='Testa modelos TFLite')
    parser.add_argument('--model_path', type=str, help='Caminho específico para modelo .tflite')
    parser.add_argument('--test_data_path', type=str, help='Caminho para dados de teste')
    parser.add_argument('--all', action='store_true', help='Testa todos os modelos .tflite encontrados')
    
    args = parser.parse_args()
    
    print("🧪 Testador de Modelos TFLite")
    print("=" * 40)
    
    if args.model_path:
        # Testa um modelo específico
        if not os.path.exists(args.model_path):
            print(f"❌ Arquivo não encontrado: {args.model_path}")
            return
        
        success = test_tflite_model(args.model_path, args.test_data_path)
        if success:
            print("\n✅ Teste concluído com sucesso!")
        else:
            print("\n❌ Teste falhou!")
            
    elif args.all:
        # Testa todos os modelos encontrados
        tflite_files = find_tflite_models()
        if not tflite_files:
            print("❌ Nenhum arquivo .tflite encontrado")
            return
        
        print(f"📁 Encontrados {len(tflite_files)} arquivo(s) .tflite")
        
        successful = 0
        for tflite_file in tflite_files:
            success = test_tflite_model(tflite_file, args.test_data_path)
            if success:
                successful += 1
        
        print(f"\n✅ Testes concluídos: {successful}/{len(tflite_files)} modelos funcionando")
        
    else:
        # Lista modelos disponíveis e permite seleção
        tflite_files = find_tflite_models()
        if not tflite_files:
            print("❌ Nenhum arquivo .tflite encontrado no repositório")
            print("💡 Use --model_path para especificar um arquivo específico")
            print("💡 Ou execute o script convert_h5_to_tflite.py primeiro")
            return
        
        print(f"📁 Modelos .tflite encontrados:")
        for i, file_path in enumerate(tflite_files, 1):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"   {i}. {file_path} ({size_mb:.1f} MB)")
        
        try:
            choice = input(f"\n🔢 Escolha um modelo (1-{len(tflite_files)}) ou 'all' para todos: ").strip()
            
            if choice.lower() == 'all':
                successful = 0
                for tflite_file in tflite_files:
                    success = test_tflite_model(tflite_file, args.test_data_path)
                    if success:
                        successful += 1
                print(f"\n✅ Testes concluídos: {successful}/{len(tflite_files)} modelos funcionando")
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(tflite_files):
                    success = test_tflite_model(tflite_files[idx], args.test_data_path)
                    if success:
                        print("\n✅ Teste concluído com sucesso!")
                    else:
                        print("\n❌ Teste falhou!")
                else:
                    print("❌ Escolha inválida")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operação cancelada")

if __name__ == "__main__":
    main() 