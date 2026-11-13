#!/usr/bin/env python3
"""
Script para converter modelos .h5 (Keras/TensorFlow) para formato TensorFlow Lite (.tflite)

Este script procura automaticamente por arquivos .h5 no repositório e oferece 
opções de conversão com diferentes níveis de otimização.

Uso:
    python convert_h5_to_tflite.py
    python convert_h5_to_tflite.py --model_path /caminho/para/modelo.h5
    python convert_h5_to_tflite.py --optimization default  # ou size ou speed
"""

import os
import sys
import argparse
import glob
import tensorflow as tf
from pathlib import Path

def find_h5_models(root_dir="."):
    """Encontra todos os arquivos .h5 no diretório e subdiretórios"""
    h5_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.h5'):
                full_path = os.path.join(root, file)
                # Verifica se é um arquivo válido e não vazio
                if os.path.getsize(full_path) > 1000:  # Pelo menos 1KB
                    h5_files.append(full_path)
    return h5_files

def convert_to_tflite(model_path, output_path=None, optimization='default'):
    """
    Converte um modelo .h5 para .tflite
    
    Args:
        model_path: Caminho para o arquivo .h5
        output_path: Caminho de saída (opcional)
        optimization: Tipo de otimização ('default', 'size', 'speed')
    """
    try:
        print(f"\n🔄 Carregando modelo: {model_path}")
        
        # Carrega o modelo
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo carregado com sucesso")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        
        # Configura o conversor
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Aplica otimizações baseadas no parâmetro
        if optimization == 'size':
            print("🔧 Aplicando otimização para tamanho...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif optimization == 'speed':
            print("🔧 Aplicando otimização para velocidade...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        else:  # default
            print("🔧 Aplicando otimização padrão...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Converte o modelo
        print("⚡ Convertendo para TensorFlow Lite...")
        tflite_model = converter.convert()
        
        # Define o caminho de saída
        if output_path is None:
            model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            output_path = os.path.join(model_dir, f"{model_name}.tflite")
        
        # Salva o modelo convertido
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Mostra informações do arquivo
        original_size = os.path.getsize(model_path) / (1024*1024)  # MB
        tflite_size = os.path.getsize(output_path) / (1024*1024)   # MB
        compression_ratio = (1 - tflite_size/original_size) * 100
        
        print(f"✅ Conversão concluída!")
        print(f"   - Arquivo original: {original_size:.1f} MB")
        print(f"   - Arquivo TFLite: {tflite_size:.1f} MB")
        print(f"   - Compressão: {compression_ratio:.1f}%")
        print(f"   - Salvo em: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Erro na conversão: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Converte modelos .h5 para .tflite')
    parser.add_argument('--model_path', type=str, help='Caminho específico para modelo .h5')
    parser.add_argument('--optimization', choices=['default', 'size', 'speed'], 
                       default='default', help='Tipo de otimização')
    parser.add_argument('--output_dir', type=str, help='Diretório de saída (opcional)')
    parser.add_argument('--all', action='store_true', help='Converte todos os modelos .h5 encontrados')
    
    args = parser.parse_args()
    
    print("🤖 Conversor H5 → TFLite")
    print("=" * 40)
    
    if args.model_path:
        # Converte um modelo específico
        if not os.path.exists(args.model_path):
            print(f"❌ Arquivo não encontrado: {args.model_path}")
            return
        
        output_path = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            model_name = os.path.splitext(os.path.basename(args.model_path))[0]
            output_path = os.path.join(args.output_dir, f"{model_name}.tflite")
        
        convert_to_tflite(args.model_path, output_path, args.optimization)
        
    elif args.all:
        # Converte todos os modelos encontrados
        h5_files = find_h5_models()
        if not h5_files:
            print("❌ Nenhum arquivo .h5 encontrado")
            return
        
        print(f"📁 Encontrados {len(h5_files)} arquivo(s) .h5:")
        for i, file_path in enumerate(h5_files, 1):
            print(f"   {i}. {file_path}")
        
        print(f"\n🔄 Convertendo todos os modelos...")
        successful = 0
        for h5_file in h5_files:
            result = convert_to_tflite(h5_file, optimization=args.optimization)
            if result:
                successful += 1
        
        print(f"\n✅ Conversões concluídas: {successful}/{len(h5_files)}")
        
    else:
        # Lista modelos disponíveis e permite seleção
        h5_files = find_h5_models()
        if not h5_files:
            print("❌ Nenhum arquivo .h5 encontrado no repositório")
            print("💡 Use --model_path para especificar um arquivo específico")
            return
        
        print(f"📁 Modelos .h5 encontrados:")
        for i, file_path in enumerate(h5_files, 1):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"   {i}. {file_path} ({size_mb:.1f} MB)")
        
        try:
            choice = input(f"\n🔢 Escolha um modelo (1-{len(h5_files)}) ou 'all' para todos: ").strip()
            
            if choice.lower() == 'all':
                successful = 0
                for h5_file in h5_files:
                    result = convert_to_tflite(h5_file, optimization=args.optimization)
                    if result:
                        successful += 1
                print(f"\n✅ Conversões concluídas: {successful}/{len(h5_files)}")
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(h5_files):
                    convert_to_tflite(h5_files[idx], optimization=args.optimization)
                else:
                    print("❌ Escolha inválida")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operação cancelada")

if __name__ == "__main__":
    main() 