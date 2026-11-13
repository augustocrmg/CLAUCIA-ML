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

def split_dataset(root_dir, output_dir, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):
    """
    Divide o dataset em treino, teste e validação.
    
    Parameters:
    - root_dir: Diretório raiz contendo subdiretórios para cada classe
    - output_dir: Diretório de saída
    - train_ratio: Porcentagem para treino (padrão: 0.7)
    - test_ratio: Porcentagem para teste (padrão: 0.15)
    - val_ratio: Porcentagem para validação (padrão: 0.15)
    
    Returns:
    - Dicionário com informações sobre a divisão
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    
    # Cria estrutura de diretórios
    splits = ['treino', 'teste', 'validacao']
    for split in splits:
        os.makedirs(output_dir / split, exist_ok=True)
    
    # Lista de subdiretórios (classes) que contêm imagens
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir() and has_images(d)]
    
    if not class_dirs:
        print(f"Nenhum subdiretório com imagens encontrado em {root_dir}")
        return None
    
    split_info = {}
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    print("\n=== DIVIDINDO DATASET EM TREINO/TESTE/VALIDAÇÃO ===\n")
    
    # Para cada classe, divide as imagens
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"Processando classe: {class_name}")
        
        # Lista todas as imagens da classe
        img_files = [f for f in os.listdir(class_dir) 
                    if os.path.isfile(os.path.join(class_dir, f)) 
                    and os.path.splitext(f)[1].lower() in img_extensions]
        
        total_images = len(img_files)
        print(f"  Total de imagens: {total_images}")
        
        # Embaralha as imagens para garantir aleatoriedade
        random.shuffle(img_files)
        
        # Calcula os índices de divisão
        train_end = int(total_images * train_ratio)
        test_end = train_end + int(total_images * test_ratio)
        
        # Divide as imagens
        train_images = img_files[:train_end]
        test_images = img_files[train_end:test_end]
        val_images = img_files[test_end:]
        
        print(f"  Treino: {len(train_images)} | Teste: {len(test_images)} | Validação: {len(val_images)}")
        
        # Cria diretórios para a classe em cada split
        for split in splits:
            os.makedirs(output_dir / split / class_name, exist_ok=True)
        
        # Copia as imagens para os diretórios correspondentes
        for img_file in train_images:
            shutil.copy2(class_dir / img_file, output_dir / 'treino' / class_name / img_file)
        
        for img_file in test_images:
            shutil.copy2(class_dir / img_file, output_dir / 'teste' / class_name / img_file)
        
        for img_file in val_images:
            shutil.copy2(class_dir / img_file, output_dir / 'validacao' / class_name / img_file)
        
        # Armazena informações da divisão
        split_info[class_name] = {
            'total': total_images,
            'treino': len(train_images),
            'teste': len(test_images),
            'validacao': len(val_images)
        }
    
    print("\n=== DIVISÃO CONCLUÍDA ===\n")
    return split_info

def balance_split(split_dir, target_images_per_class):
    """
    Balanceia um split específico (treino, teste ou validação) usando data augmentation.
    
    Parameters:
    - split_dir: Diretório do split a ser balanceado
    - target_images_per_class: Número alvo de imagens por classe
    """
    split_dir = Path(split_dir)
    split_name = split_dir.name
    
    print(f"\n=== BALANCEANDO {split_name.upper()} ===\n")
    
    # Lista de subdiretórios (classes) que contêm imagens
    class_dirs = [d for d in split_dir.iterdir() if d.is_dir() and has_images(d)]
    
    if not class_dirs:
        print(f"Nenhum subdiretório com imagens encontrado em {split_dir}")
        return
    
    # Conta imagens em cada classe
    class_counts = {}
    for class_dir in class_dirs:
        img_count = count_images_in_dir(class_dir)
        if img_count > 0:
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
        class_dir = split_dir / class_name
        
        # Lista as imagens originais
        img_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) 
                    and os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
        
        print(f"\nProcessando classe: {class_name} ({img_count} imagens originais)")
        
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
                new_path = apply_augmentation(img_path, class_dir, aug_func, f"{suffix}_{j}")
                if new_path:
                    aug_images_count += 1
            
            print(f"  → {aug_images_count} imagens geradas com sucesso")
    
    # Conta o número de imagens após o balanceamento
    final_counts = {}
    for class_dir in split_dir.iterdir():
        if class_dir.is_dir():
            final_counts[class_dir.name] = count_images_in_dir(class_dir)
    
    print(f"\nResultado do balanceamento para {split_name}:")
    for class_name, count in final_counts.items():
        original = class_counts.get(class_name, 0)
        print(f"{class_name}: {original} → {count} imagens")

def balance_dataset_perfect(root_dir, output_dir=None, target_images_per_class=1000, 
                           train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):
    """
    Balanceia o dataset de forma perfeita: primeiro divide em treino/teste/validação,
    depois aplica augmentation separadamente em cada split mantendo as proporções.
    
    Parameters:
    - root_dir: Diretório raiz contendo subdiretórios para cada classe
    - output_dir: Diretório de saída (se None, usará o mesmo diretório raiz com prefixo)
    - target_images_per_class: Número alvo TOTAL de imagens por classe (padrão: 1000)
    - train_ratio: Porcentagem para treino (padrão: 0.7)
    - test_ratio: Porcentagem para teste (padrão: 0.15)
    - val_ratio: Porcentagem para validação (padrão: 0.15)
    
    Exemplo:
    - Se target_images_per_class = 1000:
      - Treino terá: 700 imagens (70%)
      - Teste terá: 150 imagens (15%)
      - Validação terá: 150 imagens (15%)
    """
    root_dir = Path(root_dir)
    
    if output_dir is None:
        output_dir = root_dir.parent / f"perfect_balanced_{root_dir.name}"
    else:
        output_dir = Path(output_dir)
    
    # Verifica se o diretório de saída já existe
    if output_dir.exists():
        print(f"Removendo diretório existente: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Cria o diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Dataset original: {root_dir}")
    print(f"Dataset balanceado será criado em: {output_dir}")
    print(f"\nTarget total: {target_images_per_class} imagens por classe")
    
    # Calcula o target para cada split baseado nas proporções
    target_train = int(target_images_per_class * train_ratio)
    target_test = int(target_images_per_class * test_ratio)
    target_val = int(target_images_per_class * val_ratio)
    
    print(f"  → Treino: {target_train} imagens ({train_ratio*100:.0f}%)")
    print(f"  → Teste: {target_test} imagens ({test_ratio*100:.0f}%)")
    print(f"  → Validação: {target_val} imagens ({val_ratio*100:.0f}%)")
    
    # Etapa 1: Divide o dataset em treino/teste/validação
    split_info = split_dataset(root_dir, output_dir, train_ratio, test_ratio, val_ratio)
    
    if split_info is None:
        return None
    
    # Etapa 2: Balanceia cada split separadamente com seus targets específicos
    splits_config = {
        'treino': target_train,
        'teste': target_test,
        'validacao': target_val
    }
    
    for split, split_target in splits_config.items():
        split_dir = output_dir / split
        balance_split(split_dir, split_target)
    
    # Resumo final
    print("\n" + "="*70)
    print("RESUMO FINAL DO BALANCEAMENTO PERFEITO")
    print("="*70)
    
    splits_config = {
        'treino': target_train,
        'teste': target_test,
        'validacao': target_val
    }
    
    for split, split_target in splits_config.items():
        print(f"\n{split.upper()} (Target: {split_target} imagens por classe):")
        split_dir = output_dir / split
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = count_images_in_dir(class_dir)
                original = split_info.get(class_dir.name, {}).get(split, 0)
                print(f"  {class_dir.name}: {original} → {count} imagens")
    
    print("\n" + "="*70)
    print(f"Dataset perfeitamente balanceado disponível em: {output_dir}")
    print(f"Total por classe: {target_train + target_test + target_val} imagens")
    print(f"  └─ Treino: {target_train} | Teste: {target_test} | Validação: {target_val}")
    print("="*70)
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Balanceia um dataset de imagens com divisão treino/teste/validação e data augmentation. '
                    'Exemplo: target=1000 resulta em 700 treino + 150 teste + 150 validação por classe.'
    )
    parser.add_argument('input_dir', type=str, 
                       help='Diretório de entrada contendo os subdiretórios das classes')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Diretório de saída (opcional, padrão: perfect_balanced_<nome_entrada>)')
    parser.add_argument('--target', type=int, default=1000, 
                       help='Número TOTAL alvo de imagens por classe (padrão: 1000, resulta em 700+150+150)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proporção de imagens para treino (padrão: 0.7 = 70%%)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Proporção de imagens para teste (padrão: 0.15 = 15%%)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Proporção de imagens para validação (padrão: 0.15 = 15%%)')
    
    args = parser.parse_args()
    
    # Valida as proporções
    total_ratio = args.train_ratio + args.test_ratio + args.val_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"ERRO: A soma das proporções deve ser 1.0 (atual: {total_ratio})")
        exit(1)
    
    balanced_dir = balance_dataset_perfect(
        args.input_dir, 
        args.output_dir, 
        args.target,
        args.train_ratio,
        args.test_ratio,
        args.val_ratio
    )
    
    if balanced_dir:
        print(f"\n✓ Processo concluído com sucesso!")

