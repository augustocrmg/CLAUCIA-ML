import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil

def scale_image(img, scale_factor=None):
    """Redimensiona a imagem entre 50-200% para simular diferentes distâncias"""
    if scale_factor is None:
        scale_factor = random.uniform(0.5, 2.0)
    
    width, height = img.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Redimensiona a imagem
    scaled_img = img.resize((new_width, new_height), Image.BICUBIC)
    
    # Se a imagem ficou maior, corta para o tamanho original
    if scale_factor > 1.0:
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        scaled_img = scaled_img.crop((left, top, left + width, top + height))
    
    # Se a imagem ficou menor, adiciona padding para manter o tamanho original
    elif scale_factor < 1.0:
        result = Image.new('RGB', (width, height), (0, 0, 0))
        paste_x = (width - new_width) // 2
        paste_y = (height - new_height) // 2
        result.paste(scaled_img, (paste_x, paste_y))
        scaled_img = result
    
    return scaled_img

def rotate_image_90(img):
    """Rotaciona a imagem em 90° para simular diferentes locais de captura"""
    return img.rotate(90, resample=Image.BICUBIC, expand=False)

def flip_horizontal(img):
    """Espelha a imagem horizontalmente (180°)"""
    return ImageOps.mirror(img)

def flip_vertical(img):
    """Espelha a imagem verticalmente (270°)"""
    return ImageOps.flip(img)

def adjust_brightness_10_percent(img, direction=None):
    """Ajusta o brilho da imagem em ±10% para simular diferentes condições de iluminação"""
    if direction is None:
        direction = random.choice(['brighten', 'darken'])
    
    if direction == 'brighten':
        factor = random.uniform(1.0, 1.1)  # +10%
    else:
        factor = random.uniform(0.9, 1.0)  # -10%
    
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def shift_pixels(img, direction=None):
    """Desloca os pixels da imagem 10 unidades para esquerda ou direita"""
    if direction is None:
        direction = random.choice(['left', 'right'])
    
    width, height = img.size
    
    if direction == 'left':
        shift = -10
    else:
        shift = 10
    
    # Cria uma nova imagem com fundo preto
    shifted_img = Image.new('RGB', (width, height), (0, 0, 0))
    
    if shift > 0:  # Shift para direita
        shifted_img.paste(img.crop((0, 0, width - shift, height)), (shift, 0))
    else:  # Shift para esquerda
        shifted_img.paste(img.crop((-shift, 0, width, height)), (0, 0))
    
    return shifted_img

def add_salt_pepper_noise(img, noise_density=0.05):
    """Adiciona ruído salt-and-pepper com densidade 0.05"""
    img_array = np.array(img)
    noise = np.random.random(img_array.shape[:2])
    
    # Salt noise (pixels brancos)
    img_array[noise < noise_density/2] = 255
    
    # Pepper noise (pixels pretos)
    img_array[noise > 1 - noise_density/2] = 0
    
    return Image.fromarray(img_array)

def add_gaussian_noise(img, mean=0, variance=0.01):
    """Adiciona ruído gaussiano com média 0 e variância 0.01"""
    img_array = np.array(img, dtype=np.float64)
    
    # Gera ruído gaussiano
    noise = np.random.normal(mean, variance**0.5, img_array.shape) * 255
    
    # Adiciona o ruído à imagem
    noisy_img = img_array + noise
    
    # Garante que os valores estejam no range [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_img)

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

def balance_dataset(root_dir, output_dir=None, target_images_per_class=1000):
    """
    Balanceia o dataset de imagens usando data augmentation para atingir um número específico de imagens por classe.
    
    Parameters:
    - root_dir: Diretório raiz contendo subdiretórios para cada classe
    - output_dir: Diretório de saída (se None, usará o mesmo diretório raiz)
    - target_images_per_class: Número alvo de imagens por classe (padrão: 1000)
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
    
    print(f"\nNúmero alvo de imagens por classe: {target_images_per_class}")
    
    # Define as funções de augmentation baseadas no artigo
    augmentation_functions = [
        (scale_image, "scaled"),
        (rotate_image_90, "rotated_90"),
        (flip_horizontal, "h_flipped"),
        (flip_vertical, "v_flipped"),
        (lambda img: adjust_brightness_10_percent(img, 'brighten'), "brightened"),
        (lambda img: adjust_brightness_10_percent(img, 'darken'), "darkened"),
        (lambda img: shift_pixels(img, 'left'), "shifted_left"),
        (lambda img: shift_pixels(img, 'right'), "shifted_right"),
        (add_salt_pepper_noise, "salt_pepper"),
        (add_gaussian_noise, "gaussian_noise")
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
        remaining = target_images_per_class - img_count
        
        if remaining <= 0:
            print(f"A classe {class_name} já possui imagens suficientes ({img_count}). Nenhuma augmentation necessária.")
            continue
        
        # Número de técnicas de augmentation disponíveis
        num_techniques = len(augmentation_functions)
        
        # Quantas imagens precisamos gerar por técnica
        per_technique = remaining // num_techniques
        extras = remaining % num_techniques
        
        print(f"Gerando {remaining} imagens adicionais com {num_techniques} técnicas de augmentation")
        
        # Aplica cada técnica de augmentation
        for i, (aug_func, suffix) in enumerate(augmentation_functions):
            # Quantas imagens gerar com esta técnica
            to_generate = per_technique + (1 if i < extras else 0)
            
            if to_generate == 0:
                continue
                
            print(f"Aplicando técnica: {suffix} ({to_generate} imagens)")
            
            # Seleciona imagens aleatórias para aplicar a técnica
            # Pode repetir a mesma imagem várias vezes se necessário
            source_images = []
            while len(source_images) < to_generate:
                source_images.extend(random.sample(img_files, min(to_generate - len(source_images), len(img_files))))
            
            # Aplica a técnica
            aug_images_count = 0
            for j, img_file in enumerate(tqdm(source_images[:to_generate], desc=f"Técnica {suffix}")):
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
    parser.add_argument('--target', type=int, default=1000, help='Número alvo de imagens por classe (padrão: 1000)')
    
    args = parser.parse_args()
    
    balanced_dir = balance_dataset(args.input_dir, args.output_dir, args.target)
    print(f"\nDataset balanceado disponível em: {balanced_dir}")
