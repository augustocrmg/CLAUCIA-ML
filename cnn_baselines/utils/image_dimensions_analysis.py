import os
import sys
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Permitir passar o caminho do dataset como argumento
if len(sys.argv) > 1:
    BASE_DIR = sys.argv[1]
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Listas para armazenar dimensões e nomes de arquivos
widths = []
heights = []
width_files = []
height_files = []

# Extensões de imagem suportadas
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if is_image_file(file):
            img_path = os.path.join(root, file)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size 
                    widths.append(w)
                    heights.append(h)
                    width_files.append((w, file))
                    height_files.append((h, file))
            except Exception as e:
                print(f"Erro ao ler {img_path}: {e}")

# Estatísticas
print(f"Total de imagens lidas: {len(widths)}")
if widths and heights:
    print(f"Largura - min: {min(widths)}, max: {max(widths)}, média: {sum(widths)/len(widths):.2f}, moda: {Counter(widths).most_common(1)[0][0]}")
    print(f"Altura  - min: {min(heights)}, max: {max(heights)}, média: {sum(heights)/len(heights):.2f}, moda: {Counter(heights).most_common(1)[0][0]}")

    # Histogramas com contagem acima das barras
    plt.figure(figsize=(12,5))

    # Largura
    plt.subplot(1,2,1)
    counts, bins, patches = plt.hist(widths, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribuição das Larguras')
    plt.xlabel('Largura (pixels)')
    plt.ylabel('Frequência')
    # Adiciona contagem acima das barras
    for count, patch in zip(counts, patches):
        if count > 0:
            plt.text(patch.get_x() + patch.get_width()/2, count, int(count), ha='center', va='bottom', fontsize=8)
    # Mediana e arquivo correspondente
    median_width = int(np.median(widths))
    plt.axvline(median_width, color='red', linestyle='dashed', linewidth=1)
    plt.text(median_width, plt.ylim()[0], f'Mediana: {median_width}', color='red', ha='center', va='bottom', fontsize=9)
    # Nome de arquivo com largura igual à mediana
    median_width_file = next((f for w, f in width_files if w == median_width), None)
    if median_width_file:
        print(f"Arquivo com largura igual à mediana ({median_width}): {median_width_file}")

    # Altura
    plt.subplot(1,2,2)
    counts, bins, patches = plt.hist(heights, bins=30, color='salmon', edgecolor='black')
    plt.title('Distribuição das Alturas')
    plt.xlabel('Altura (pixels)')
    plt.ylabel('Frequência')
    for count, patch in zip(counts, patches):
        if count > 0:
            plt.text(patch.get_x() + patch.get_width()/2, count, int(count), ha='center', va='bottom', fontsize=8)
    median_height = int(np.median(heights))
    plt.axvline(median_height, color='red', linestyle='dashed', linewidth=1)
    plt.text(median_height, plt.ylim()[0], f'Mediana: {median_height}', color='red', ha='center', va='bottom', fontsize=9)
    median_height_file = next((f for h, f in height_files if h == median_height), None)
    if median_height_file:
        print(f"Arquivo com altura igual à mediana ({median_height}): {median_height_file}")

    plt.tight_layout()
    plt.show()
else:
    print("Nenhuma imagem válida encontrada.") 