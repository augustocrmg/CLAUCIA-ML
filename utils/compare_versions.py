import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os

# Configuration
IMAGE_SIZE = (224, 224)
DATASET_PATH = "datasets/dataset_wagner_balanceado_2"

def apply_blur_light(image_array, blur_strength=15):
    """Blur leve (versão original)"""
    if image_array.dtype == np.float32:
        image_array = (image_array * 255).astype(np.uint8)
    
    blurred = cv2.GaussianBlur(image_array, (blur_strength, blur_strength), 0)
    return blurred

def apply_blur_medium(image_array, blur_strength=25):
    """Blur médio (versão intermediária)"""
    if image_array.dtype == np.float32:
        image_array = (image_array * 255).astype(np.uint8)
    
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    blurred = cv2.GaussianBlur(image_array, (blur_strength, blur_strength), 0)
    blur_strength_2 = max(3, blur_strength // 2)
    if blur_strength_2 % 2 == 0:
        blur_strength_2 += 1
    blurred = cv2.GaussianBlur(blurred, (blur_strength_2, blur_strength_2), 0)
    return blurred

def apply_blur_strong(image_array, blur_strength=45):
    """Blur forte (versão atual)"""
    if image_array.dtype == np.float32:
        image_array = (image_array * 255).astype(np.uint8)
    
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Triplo blur
    blurred = cv2.GaussianBlur(image_array, (blur_strength, blur_strength), 0)
    
    blur_strength_2 = max(3, blur_strength // 2)
    if blur_strength_2 % 2 == 0:
        blur_strength_2 += 1
    blurred = cv2.GaussianBlur(blurred, (blur_strength_2, blur_strength_2), 0)
    
    blur_strength_3 = max(3, blur_strength // 3)
    if blur_strength_3 % 2 == 0:
        blur_strength_3 += 1
    blurred = cv2.GaussianBlur(blurred, (blur_strength_3, blur_strength_3), 0)
    return blurred

def create_blur_comparison():
    """Cria comparação visual dos diferentes níveis de blur"""
    
    # Pega uma imagem de exemplo
    class_names = sorted([d for d in os.listdir(DATASET_PATH) 
                         if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    # Seleciona uma imagem
    class_path = os.path.join(DATASET_PATH, class_names[0])
    image_files = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("Nenhuma imagem encontrada!")
        return
    
    img_path = os.path.join(class_path, image_files[0])
    
    # Carrega a imagem
    img = keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    
    # Aplica diferentes níveis de blur
    original = img_array.astype(np.uint8)
    blur_light = apply_blur_light(img_array.copy())
    blur_medium = apply_blur_medium(img_array.copy())
    blur_strong = apply_blur_strong(img_array.copy())
    
    # Cria a visualização comparativa
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Comparação dos Níveis de Blur para Preservação de Privacidade', 
                 fontsize=16, fontweight='bold')
    
    images = [original, blur_light, blur_medium, blur_strong]
    titles = ['Original\n(Não adequado)', 'Blur Leve\n(15px)', 'Blur Médio\n(25px duplo)', 'Blur Forte\n(45px triplo)']
    colors = ['red', 'orange', 'yellow', 'green']
    
    for i, (image, title, color) in enumerate(zip(images, titles, colors)):
        axes[i].imshow(image)
        axes[i].set_title(title, fontsize=12, fontweight='bold', color=color)
        axes[i].axis('off')
        
        # Adiciona borda colorida
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3)
    
    # Adiciona notas explicativas
    notes = [
        "❌ Inadequado\npara publicação",
        "⚠️ Privacidade\ninsuficiente", 
        "✅ Adequado para\nmídia digital",
        "✅ Ideal para\njornal impresso"
    ]
    
    for i, note in enumerate(notes):
        axes[i].text(0.5, -0.15, note, transform=axes[i].transAxes, 
                    ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Salva a comparação
    output_path = 'blur_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Comparação de blur salva em: {output_path}")
    print("\nRecomendação: Use sempre o 'Blur Forte' para publicações!")
    
    plt.show()
    return output_path

if __name__ == "__main__":
    create_blur_comparison() 