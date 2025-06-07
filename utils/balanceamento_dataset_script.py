import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil

def rotate_image(img, angle=None):
    """Rotaciona a imagem em um ângulo aleatório"""
    if angle is None:
        angle = random.randint(5, 355)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)

def flip_image(img, horizontal=True):
    """Espelha a imagem horizontalmente ou verticalmente"""
    if horizontal:
        return ImageOps.mirror(img)
    else:
        return ImageOps.flip(img)

def adjust_brightness(img, factor=None):
    """Ajusta o brilho da imagem"""
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def adjust_contrast(img, factor=None):
    """Ajusta o contraste da imagem"""
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def apply_augmentation(img_path, output_path, augmentation_func, suffix):
    """Aplica a função de aumentação na imagem e salva em um novo arquivo"""
    try:
        with Image.open(img_path) as img:
            augmented_img = augmentation_func(img)
            
            # Cria o novo nome de arquivo com o sufixo
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)
            new_filename = f"{name}_{suffix}{ext}"
            new_path = os.path.join(output_path, new_filename)
            
            # Salva a imagem aumentada
            augmented_img.save(new_path)
            return new_path
    except Exception as e:
        print(f"Erro ao processar {img_path}: {e}")
        return None

def count_images_in_dir(directory):
    """Conta o número de imagens em um diretório"""
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return sum(1 for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) 
               and os.path.splitext(f)[1].lower() in img_extensions)

def has_images(directory):
    """Verifica se o diretório contém pelo menos uma imagem"""
    return count_images_in_dir(directory) > 0

def balance_dataset(root_dir, output_dir=None, augmentation_factor=6):
    """
    Balanceia o dataset de imagens usando data augmentation.
    
    Parameters:
    - root_dir: Diretório raiz contendo subdiretórios para cada classe
    - output_dir: Diretório de saída (se None, usará o mesmo diretório raiz)
    - augmentation_factor: Multiplicador para o número de imagens da menor classe
    """
    root_dir = Path(root_dir)
    
    if output_dir is None:
        output_dir = root_dir.parent / f"{root_dir.name}_balanced"
    else:
        output_dir = Path(output_dir)
    
    # Verifica se o diretório de saída já existe
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Cria o diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Lista de subdiretórios (classes) que contêm imagens
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir() and has_images(d)]
    
    if not class_dirs:
        print(f"Nenhum subdiretório com imagens encontrado em {root_dir}")
        return
    
    # Conta imagens em cada classe
    class_counts = {}
    for class_dir in class_dirs:
        img_count = count_images_in_dir(class_dir)
        if img_count > 0:  # Verifica novamente para garantir
            class_counts[class_dir.name] = img_count
            print(f"Classe {class_dir.name}: {img_count} imagens")
    
    if not class_counts:
        print("Nenhum diretório com imagens encontrado")
        return
        
    # Encontra a classe com menos imagens
    min_class = min(class_counts.items(), key=lambda x: x[1])
    min_class_name, min_count = min_class
    
    # Define o número alvo de imagens por classe (6x a menor classe)
    target_count = min_count * augmentation_factor
    
    print(f"\nClasse com menos imagens: {min_class_name} ({min_count} imagens)")
    print(f"Número alvo de imagens por classe: {target_count}")
    
    # Define as funções de augmentation que serão utilizadas
    augmentation_functions = [
        (rotate_image, "rotated"),
        (lambda img: flip_image(img, horizontal=True), "h_flipped"),
        (lambda img: flip_image(img, horizontal=False), "v_flipped"),
        (adjust_brightness, "brightness"),
        (adjust_contrast, "contrast")
        # TODO: Adicionar o `random_noise`
        # TODO: Adicionar mais filtros
    ]
    
    # Processa cada classe
    for class_name, img_count in class_counts.items():
        # Cria diretório para a classe no output
        class_output_dir = output_dir / class_name
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Copia todas as imagens originais para o diretório de saída
        class_dir = root_dir / class_name
        img_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) 
                    and os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
        
        print(f"\nProcessando classe: {class_name} ({img_count} imagens originais)")
        
        # Copia as imagens originais
        for img_file in img_files:
            source = class_dir / img_file
            dest = class_output_dir / img_file
            shutil.copy2(source, dest)
        
        # Calcula quantas imagens faltam para atingir o target
        remaining = target_count - img_count
        
        if remaining <= 0:
            print(f"A classe {class_name} já possui imagens suficientes ({img_count}). Nenhuma augmentation necessária.")
            continue
        
        # Número de tratamentos aplicados
        num_treatments = len(augmentation_functions)
        
        # Quantas imagens precisamos gerar por tratamento
        per_treatment = remaining // num_treatments
        extras = remaining % num_treatments
        
        print(f"Gerando {remaining} imagens adicionais com {num_treatments} tratamentos")
        
        # Aplica cada técnica de augmentation
        for i, (aug_func, suffix) in enumerate(augmentation_functions):
            # Quantas imagens gerar com este tratamento
            to_generate = per_treatment + (1 if i < extras else 0)
            
            if to_generate == 0:
                continue
                
            print(f"Aplicando tratamento: {suffix} ({to_generate} imagens)")
            
            # Seleciona imagens aleatórias para aplicar o tratamento
            # Pode repetir a mesma imagem várias vezes se necessário
            source_images = []
            while len(source_images) < to_generate:
                source_images.extend(random.sample(img_files, min(to_generate - len(source_images), len(img_files))))
            
            # Aplica o tratamento
            aug_images_count = 0
            for j, img_file in enumerate(tqdm(source_images[:to_generate], desc=f"Tratamento {suffix}")):
                img_path = class_dir / img_file
                new_path = apply_augmentation(img_path, class_output_dir, aug_func, f"{suffix}_{j}")
                if new_path:
                    aug_images_count += 1
            
            print(f"  → {aug_images_count} imagens geradas com sucesso")
    
    # Conta o número de imagens após o balanceamento
    final_counts = {}
    for class_dir in output_dir.iterdir():
        if class_dir.is_dir():
            final_counts[class_dir.name] = count_images_in_dir(class_dir)
    
    print("\nResultado do balanceamento:")
    for class_name, count in final_counts.items():
        original = class_counts.get(class_name, 0)
        print(f"{class_name}: {original} → {count} imagens")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Balanceia um dataset de imagens usando data augmentation')
    parser.add_argument('input_dir', type=str, help='Diretório de entrada contendo os subdiretórios das classes')
    parser.add_argument('--output_dir', type=str, default=None, help='Diretório de saída (opcional)')
    parser.add_argument('--factor', type=int, default=6, help='Fator de multiplicação da menor classe (padrão: 6)')
    
    args = parser.parse_args()
    
    balanced_dir = balance_dataset(args.input_dir, args.output_dir, args.factor)
    print(f"\nDataset balanceado disponível em: {balanced_dir}")
