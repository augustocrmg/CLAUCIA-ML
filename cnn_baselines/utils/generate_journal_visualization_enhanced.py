import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image, ImageFilter
import cv2
from datetime import datetime
import time

# Enable GPU and configure memory growth
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPU memory growth enabled for {len(gpus)} GPU(s)')
        
        # Set GPU as preferred device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f'Using GPU: {gpus[0]}')
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")

# Force TensorFlow to use GPU for operations
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    print(f"TensorFlow will use: {'GPU:0' if gpus else 'CPU:0'}")

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
    """Carrega o modelo treinado e configura para GPU"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    
    # Load model on GPU if available
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        model = keras.models.load_model(MODEL_PATH)
        print(f"Modelo carregado de {MODEL_PATH}")
        print(f"Modelo carregado no dispositivo: {'GPU:0' if gpus else 'CPU:0'}")
        
        # Verify model is on GPU
        if gpus:
            print("Confirmando que o modelo está na GPU...")
            # Create a small test tensor to verify GPU usage
            test_input = tf.random.normal((1, *IMAGE_SIZE, 3))
            with tf.device('/GPU:0'):
                _ = model(test_input)
            print("✅ Modelo confirmado na GPU")
        
    return model

def apply_blur_to_image(image_array, blur_strength=45):
    """Aplica blur gaussiano mais forte na imagem para preservar privacidade"""
    # Converte de float para uint8 se necessário
    if image_array.dtype == np.float32:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Garante que o kernel seja ímpar
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Aplica blur gaussiano triplo para máxima privacy
    blurred = cv2.GaussianBlur(image_array, (blur_strength, blur_strength), 0)
    
    # Segundo blur com kernel menor
    blur_strength_2 = max(3, blur_strength // 2)
    if blur_strength_2 % 2 == 0:
        blur_strength_2 += 1
        
    blurred = cv2.GaussianBlur(blurred, (blur_strength_2, blur_strength_2), 0)
    
    # Terceiro blur para garantir anonimização total
    blur_strength_3 = max(3, blur_strength // 3)
    if blur_strength_3 % 2 == 0:
        blur_strength_3 += 1
        
    blurred = cv2.GaussianBlur(blurred, (blur_strength_3, blur_strength_3), 0)
    return blurred

def select_sample_images():
    """Seleciona 9 imagens do dataset com representação balanceada, evitando imagens aumentadas"""
    selected_images = []
    total_images = 0
    aug_images_count = 0
    
    # Define filtros de imagens aumentadas (consistente em todo o código)
    augmentation_keywords = ['aug_', 'rotated', 'flipped', 'zoomed', 'cropped', 
                           'shifted', 'blurred', 'noisy', 'augmented', 'brightness', 'contrast']
    
    def is_augmented_image(filename):
        """Verifica se uma imagem é aumentada baseado nas palavras-chave"""
        filename_lower = filename.lower()
        return any(keyword in filename_lower for keyword in augmentation_keywords)
    
    # Primeiro, conta quantas imagens existem e quantas são aug_
    for class_name in class_names:
        class_path = os.path.join(DATASET_PATH, class_name)
        all_images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        aug_images = [f for f in all_images if is_augmented_image(f)]
        
        total_images += len(all_images)
        aug_images_count += len(aug_images)
    
    print(f"\n📊 Estatísticas do Dataset:")
    print(f"  • Total de imagens: {total_images}")
    print(f"  • Imagens aumentadas: {aug_images_count}")
    print(f"  • Imagens originais disponíveis: {total_images - aug_images_count}")
    print(f"  • Filtros aplicados: {', '.join(augmentation_keywords)}")
    
    # Garante pelo menos 1 imagem de cada classe
    for class_name in class_names:
        class_path = os.path.join(DATASET_PATH, class_name)
        
        # Filtra imagens excluindo aumentadas (filtro consistente)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                      and not is_augmented_image(f)]
        
        if image_files:
            selected_file = random.choice(image_files)
            img_path = os.path.join(class_path, selected_file)
            selected_images.append((img_path, class_name))
            print(f"  ✅ {class_name}: {len(image_files)} imagens originais disponíveis")
        else:
            print(f"  ⚠️ {class_name}: Nenhuma imagem original encontrada!")
    
    # Adiciona mais imagens para completar 9
    while len(selected_images) < 9:
        class_name = random.choice(class_names)
        class_path = os.path.join(DATASET_PATH, class_name)
        
        # Filtra imagens excluindo aumentadas (filtro consistente)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                      and not is_augmented_image(f)]
        
        if image_files:
            selected_file = random.choice(image_files)
            img_path = os.path.join(class_path, selected_file)
            # Evita duplicatas
            if (img_path, class_name) not in selected_images:
                selected_images.append((img_path, class_name))
    
    # Embaralha a ordem final
    random.shuffle(selected_images)
    
    # Log das imagens selecionadas para verificação
    print(f"\n📋 Imagens selecionadas para visualização:")
    for i, (img_path, class_name) in enumerate(selected_images[:9]):
        img_filename = os.path.basename(img_path)
        print(f"  {i+1}. {class_name}: {img_filename}")
    
    return selected_images[:9]

def create_professional_visualization():
    """Cria visualização profissional 3x3 para o jornal"""
    # Carrega o modelo
    model = load_trained_model()
    
    # Seleciona as imagens
    selected_images = select_sample_images()
    
    # Configura o plot com tema profissional e mais espaço para o título
    plt.style.use('default')
    fig, axes = plt.subplots(3, 3, figsize=(18, 22))
    
    # Título principal mais afastado das imagens
    fig.suptitle('Sistema Inteligente de Classificação de Condições Dermatológicas\nResultados do Modelo EfficientNet-B0', 
                 fontsize=24, fontweight='bold', y=0.99, ha='center')
    
    # Adiciona subtítulo com informações técnicas mais afastado
    subtitle = f'Modelo treinado com {NUM_CLASSES} classes • Arquitetura: EfficientNet-B0 • Resolução: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}px'
    fig.text(0.5, 0.92, subtitle, fontsize=14, ha='center', style='italic', color='gray')
    
    predictions_data = []
    
    for idx, (img_path, true_class) in enumerate(selected_images):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Carrega e preprocessa a imagem
        img = keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = keras.preprocessing.image.img_to_array(img)
        
        # Aplica blur muito mais forte para preservar privacidade
        blurred_img = apply_blur_to_image(img_array.astype(np.uint8), blur_strength=50)
        
        # Faz a predição na GPU
        img_for_prediction = np.expand_dims(img_array, axis=0)
        
        # Ensure prediction runs on GPU
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            predictions = model.predict(img_for_prediction, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Armazena dados para estatísticas
        is_correct = predicted_class == true_class
        predictions_data.append({
            'true': true_class, 
            'predicted': predicted_class, 
            'confidence': confidence, 
            'correct': is_correct
        })
        
        # Mostra a imagem borrada
        ax.imshow(blurred_img)
        
        # Define cores e símbolos baseados na correção
        if is_correct:
            title_color = '#2E8B57'  # Verde mais suave
            symbol = '✓'
            border_color = '#2E8B57'
        else:
            title_color = '#DC143C'  # Vermelho mais suave
            symbol = '✗'
            border_color = '#DC143C'
        
        # Formata o título de forma mais limpa
        title = f"{symbol} Diagnóstico Real: {true_class}\n   Predição do IA: {predicted_class}\n   Confiança: {confidence:.1%}"
        ax.set_title(title, fontsize=11, fontweight='bold', color=title_color, 
                    pad=15, linespacing=1.2)
        ax.axis('off')
        
        # Adiciona borda colorida mais elegante
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(4)
            spine.set_capstyle('round')
    
    # Calcula estatísticas
    correct_count = sum(1 for p in predictions_data if p['correct'])
    accuracy = correct_count / len(predictions_data)
    avg_confidence = np.mean([p['confidence'] for p in predictions_data])
    
    # Adiciona caixa de estatísticas no rodapé
    stats_text = f"""Estatísticas desta Amostra:
• Precisão: {accuracy:.1%} ({correct_count}/{len(predictions_data)} predições corretas)
• Confiança Média: {avg_confidence:.1%}
• Total de Classes: {NUM_CLASSES}
• Arquitetura: EfficientNet-B0 com Transfer Learning"""
    
    fig.text(0.5, 0.02, stats_text, fontsize=12, ha='center', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
             verticalalignment='bottom')
    
    # Adiciona nota sobre privacidade
    privacy_note = "Nota: As imagens foram propositalmente\n borradas para preservar a privacidade dos pacientes"
    fig.text(0.99, 0.01, privacy_note, fontsize=10, ha='right', style='italic', 
             color='gray', verticalalignment='bottom')
    
    # Adiciona data/hora
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    fig.text(0.01, 0.01, f"Gerado em: {timestamp}", fontsize=10, ha='left', 
             color='gray', verticalalignment='bottom')
    
    # Ajusta o layout com mais espaço para títulos
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12, hspace=0.4, wspace=0.2)
    
    # Salva com qualidade alta
    output_path = 'visualization_professional_journal.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                metadata={'Title': 'Sistema IA Classificação Dermatológica',
                         'Author': 'Modelo EfficientNet-B0'})
    
    print(f"\n{'='*60}")
    print(f"VISUALIZAÇÃO PROFISSIONAL GERADA COM SUCESSO!")
    print(f"{'='*60}")
    print(f"Arquivo: {output_path}")
    print(f"Resolução: 300 DPI (qualidade de impressão)")
    print(f"Dimensões: 18x22 polegadas (aumentado para melhor espaçamento)")
    print(f"\nConfiguração de Hardware:")
    if gpus:
        print(f"• GPU Utilizada: {gpus[0].name}")
        print(f"• Dispositivo TensorFlow: GPU:0")
        print(f"• Aceleração GPU: ✅ Ativa")
    else:
        print(f"• Dispositivo TensorFlow: CPU")
        print(f"• Aceleração GPU: ❌ Não disponível")
    print(f"\nEstatísticas da Amostra:")
    print(f"• Precisão: {accuracy:.1%}")
    print(f"• Confiança Média: {avg_confidence:.1%}")
    print(f"• Predições Corretas: {correct_count}/{len(predictions_data)}")
    print(f"\nMelhorias aplicadas:")
    print(f"• Blur muito mais forte (kernel 45-51px)")
    print(f"• Título reposicionado sem sobreposição")
    print(f"• Layout otimizado para jornal")
    print(f"• Processamento acelerado por GPU")
    print(f"\nEsta imagem está otimizada para publicação em jornal!")
    print(f"{'='*60}")
    
    plt.show()
    return output_path, predictions_data

if __name__ == "__main__":
    # Usa seed baseado no tempo para seleção aleatória diferente a cada execução
    # Para reproduzir resultados específicos, mude para um valor fixo como random.seed(42)
    seed_value = int(time.time()) % 10000  # Usa os últimos 4 dígitos do timestamp
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    print(f"🎲 Seed utilizado: {seed_value} (para reproduzir estes resultados, use random.seed({seed_value}))")
    
    # Gera a visualização profissional
    output_file, stats = create_professional_visualization() 