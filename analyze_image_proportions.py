#!/usr/bin/env python3
"""
Script para analisar as proporções das imagens em um diretório e seus subdiretórios.
Uso: python analyze_image_proportions.py <diretório>
"""

import os
import sys
from collections import defaultdict
from PIL import Image
import argparse

def get_image_dimensions(image_path):
    """
    Obtém as dimensões de uma imagem.
    Retorna (largura, altura) ou None se não for possível abrir a imagem.
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Erro ao abrir imagem {image_path}: {e}")
        return None

def is_image_file(filename):
    """
    Verifica se um arquivo é uma imagem baseado na extensão.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return os.path.splitext(filename.lower())[1] in image_extensions

def analyze_image_proportions(directory):
    """
    Analisa as proporções das imagens no diretório e subdiretórios.
    Retorna um dicionário com as proporções como chave e contagem como valor.
    """
    proportions_count = defaultdict(int)
    total_images = 0
    
    print(f"Analisando imagens no diretório: {directory}")
    print("=" * 50)
    
    # Percorrer recursivamente todos os arquivos
    for root, dirs, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                image_path = os.path.join(root, file)
                dimensions = get_image_dimensions(image_path)
                
                if dimensions:
                    width, height = dimensions
                    proportion_key = f"{width} x {height}"
                    proportions_count[proportion_key] += 1
                    total_images += 1
                    
                    # Mostrar progresso a cada 100 imagens
                    if total_images % 100 == 0:
                        print(f"Processadas {total_images} imagens...")
    
    return proportions_count, total_images

def main():
    parser = argparse.ArgumentParser(description='Analisa as proporções das imagens em um diretório')
    parser.add_argument('directory', help='Diretório a ser analisado')
    
    args = parser.parse_args()
    
    # Verificar se o diretório existe
    if not os.path.exists(args.directory):
        print(f"Erro: O diretório '{args.directory}' não existe.")
        sys.exit(1)
    
    if not os.path.isdir(args.directory):
        print(f"Erro: '{args.directory}' não é um diretório.")
        sys.exit(1)
    
    # Analisar as proporções
    proportions_count, total_images = analyze_image_proportions(args.directory)
    
    # Exibir resultados
    print(f"\nProcessamento concluído!")
    print(f"Total de imagens encontradas: {total_images}")
    print("\nProporções das imagens:")
    print("=" * 50)
    
    if proportions_count:
        # Ordenar por contagem (decrescente)
        sorted_proportions = sorted(proportions_count.items(), key=lambda x: x[1], reverse=True)
        
        for proportion, count in sorted_proportions:
            print(f"{proportion}: {count} imagens")
    else:
        print("Nenhuma imagem encontrada no diretório especificado.")

if __name__ == "__main__":
    main() 