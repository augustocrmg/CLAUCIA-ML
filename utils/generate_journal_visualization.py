import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image, ImageFilter
import cv2

# Configuration
IMAGE_SIZE = (224, 224)
DATASET_PATH = "datasets/dataset_wagner_balanceado_2"
MODEL_PATH = "skin_condition_efficientnet_model.h5"

# Get class names from directory structure
class_names = sorted([d for d in os.listdir(DATASET_PATH) 
                     if os.path.isdir(os.path.join(DATASET_PATH, d))])
NUM_CLASSES = len(class_names)

print(f"Classes encontradas: {class_names}")
print(f"Número de classes: {NUM_CLASSES}")

def load_trained_model():
    """Carrega o modelo treinado"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modelo carregado de {MODEL_PATH}")
    return model

def apply_blur_to_image(image_array, blur_strength=15):
    """Aplica blur gaussiano na imagem para preservar privacidade"""
    # Converte de float para uint8 se necessário
    if image_array.dtype == np.float32:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Aplica blur gaussiano
    blurred = cv2.GaussianBlur(image_array, (blur_strength, blur_strength), 0)
    return blurred

def select_sample_images():
    """Seleciona 9 imagens do dataset (aproximadamente 2 de cada classe)"""
    selected_images = []
    
    # Para garantir representação de todas as classes
    images_per_class = max(1, 9 // len(class_names))
    remaining_slots = 9 - (images_per_class * len(class_names))
    
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(DATASET_PATH, class_name)
        
        # Pega todas as imagens da classe
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Determina quantas imagens pegar desta classe
        num_to_select = images_per_class
        if i < remaining_slots:
            num_to_select += 1
        
        # Seleciona aleatoriamente as imagens
        selected_files = random.sample(image_files, min(num_to_select, len(image_files)))
        
        for img_file in selected_files:
            img_path = os.path.join(class_path, img_file)
            selected_images.append((img_path, class_name))
    
    # Embaralha a ordem final e garante que temos exatamente 9 imagens
    random.shuffle(selected_images)
    return selected_images[:9]

def predict_and_visualize_for_journal():
    """Cria visualização 3x3 para o jornal com imagens borradas"""
    # Carrega o modelo
    model = load_trained_model()
    
    # Seleciona as imagens
    selected_images = select_sample_images()
    
    # Configura o plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Modelo de Classificação de Condições de Pele - Resultados', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    for idx, (img_path, true_class) in enumerate(selected_images):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Carrega e preprocessa a imagem
        img = keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = keras.preprocessing.image.img_to_array(img)
        
        # Aplica blur para preservar privacidade
        blurred_img = apply_blur_to_image(img_array.astype(np.uint8))
        
        # Faz a predição
        img_for_prediction = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_for_prediction, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Mostra a imagem borrada
        ax.imshow(blurred_img)
        
        # Define a cor do título baseada na correção da predição
        title_color = 'green' if predicted_class == true_class else 'red'
        
        # Cria o título com classificação real vs. prevista
        title = f"Real: {true_class}\nPrevisto: {predicted_class}\nConfiança: {confidence:.2f}"
        ax.set_title(title, fontsize=12, fontweight='bold', color=title_color, pad=10)
        ax.axis('off')
        
        # Adiciona uma borda colorida
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(title_color)
            spine.set_linewidth(3)
    
    # Ajusta o layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.1)
    
    # Salva a imagem
    output_path = 'visualization_for_journal.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Visualização salva em: {output_path}")
    
    # Mostra estatísticas
    correct_predictions = sum(1 for img_path, true_class in selected_images 
                            if get_prediction(model, img_path) == true_class)
    
    print(f"\nEstatísticas:")
    print(f"Total de imagens: {len(selected_images)}")
    print(f"Predições corretas: {correct_predictions}")
    print(f"Acurácia na amostra: {correct_predictions/len(selected_images):.2%}")
    
    plt.show()
    return output_path

def get_prediction(model, img_path):
    """Função auxiliar para obter a predição de uma imagem"""
    img = keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_for_prediction = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_for_prediction, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    return class_names[predicted_class_idx]

if __name__ == "__main__":
    # Define seed para reprodutibilidade
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Gera a visualização
    output_file = predict_and_visualize_for_journal()
    print(f"\nArquivo gerado: {output_file}")
    print("Esta imagem está pronta para ser enviada ao jornal!") 