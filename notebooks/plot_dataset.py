import matplotlib.pyplot as plt
import os
from pathlib import Path

def count_images_in_folders(dataset_path):
    """
    Conta o número de imagens (jpg, jpeg, png) em cada pasta do dataset
    """
    image_count = {}
    
    # Lista das pastas no dataset
    folders = [
        "Normal(Healthy skin)",
        "Grade 0", 
        "Grade 1",
        "Grade 2", 
        "Grade 3",
        "Grade 4",
        "Grade 5"
    ]
    
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            # Conta arquivos de imagem (jpg, jpeg, png)
            count = 0
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    count += 1
            image_count[folder] = count
        else:
            image_count[folder] = 0
    
    return image_count

def create_bar_plot(image_count):
    """
    Cria um bar plot da distribuição das imagens por classe
    """
    # Preparar os dados
    folders = list(image_count.keys())
    counts = list(image_count.values())
    
    # Configurar o plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(folders, counts, color='#3A7F58', alpha=0.8)  # Verde mais escuro (SeaGreen)
    
    # Adicionar números acima das barras
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Configurar o plot
    plt.title('Quantidade de Imagens por Estágio - Grau', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Quantidade de Imagens', fontsize=12)
    
    # Adicionar grid
    plt.grid(True, alpha=0.3)
    
    # Rotacionar labels do eixo x para melhor legibilidade
    plt.xticks(rotation=45, ha='right')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar o plot
    plt.savefig('distribuicao_dataset_wagner.png', dpi=300, bbox_inches='tight')
    
    # Mostrar o plot
    plt.show()
    
    return counts

def main():
    # Caminho para o dataset
    dataset_path = "datasets/dataset_wagner_v2"
    
    # Contar imagens
    print("Contando imagens nas pastas...")
    image_count = count_images_in_folders(dataset_path)
    
    # Exibir contagem
    print("\nDistribuição das imagens:")
    for folder, count in image_count.items():
        print(f"{folder}: {count} imagens")
    
    # Criar o bar plot
    print("\nGerando bar plot...")
    create_bar_plot(image_count)
    
    print("Plot salvo como 'distribuicao_dataset_wagner.png'")

if __name__ == "__main__":
    main()
