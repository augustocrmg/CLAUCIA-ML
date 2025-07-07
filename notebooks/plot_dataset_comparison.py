import matplotlib.pyplot as plt
import os
from pathlib import Path

def count_images_in_folders(dataset_path):
    """
    Conta o número de imagens (jpg, jpeg, png) em cada pasta do dataset
    """
    image_count = {}
    
    # Lista das pastas possíveis no dataset
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

def get_common_classes(dataset1_counts, dataset2_counts):
    """
    Retorna apenas as classes que existem em ambos os datasets
    """
    common_classes = []
    for class_name in dataset1_counts.keys():
        if (dataset1_counts[class_name] > 0 and 
            class_name in dataset2_counts and 
            dataset2_counts[class_name] > 0):
            common_classes.append(class_name)
    
    return common_classes

def create_stacked_bar_plot(dataset1_counts, dataset2_counts, common_classes):
    """
    Cria um plot de barras empilhadas comparando os dois datasets
    """
    # Preparar os dados apenas para as classes comuns
    folders = common_classes
    counts1 = [dataset1_counts[folder] for folder in folders]
    # Para o segundo dataset, calculamos apenas as imagens adicionadas (augmentation)
    counts_augmented = [dataset2_counts[folder] - dataset1_counts[folder] for folder in folders]
    
    # Configurar o plot
    plt.figure(figsize=(14, 8))
    
    # Criar barras empilhadas
    bars1 = plt.bar(folders, counts1, color='#3A7F58', alpha=0.8, label='Dataset de Wagner Curado (Original)')
    bars2 = plt.bar(folders, counts_augmented, bottom=counts1, color='#90EE90', alpha=0.8, label='Imagens Adicionadas (Data Augmentation)')
    
    # Adicionar números nas barras
    for i, (bar1, bar2, count1, count_aug) in enumerate(zip(bars1, bars2, counts1, counts_augmented)):
        # Número no meio da primeira barra (imagens originais)
        if count1 > 0:
            plt.text(bar1.get_x() + bar1.get_width()/2., count1/2,
                    str(count1), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        # Número no meio da segunda barra (imagens augmentadas)
        if count_aug > 0:
            plt.text(bar2.get_x() + bar2.get_width()/2., count1 + count_aug/2,
                    str(count_aug), ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        
        # Total no topo
        total = count1 + count_aug
        plt.text(bar1.get_x() + bar1.get_width()/2., total + 10,
                f'Total: {total}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Configurar o plot
    plt.title('Evolução do Dataset - Original vs Balanceado com Data Augmentation', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Quantidade de Imagens', fontsize=12)
    
    # Ajustar os limites do eixo Y para dar mais espaço para a legenda e números no topo
    max_height = max([c1 + c2 for c1, c2 in zip(counts1, counts_augmented)])
    plt.ylim(0, max_height * 1.25)  # 25% de espaço extra no topo (mais espaço)
    
    # Adicionar legenda no canto superior direito, aproveitando o espaço criado
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11, 
               framealpha=0.9, fancybox=True, shadow=True, ncol=1)
    
    # Adicionar grid
    plt.grid(True, alpha=0.3)
    
    # Rotacionar labels do eixo x para melhor legibilidade
    plt.xticks(rotation=45, ha='right')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar o plot
    plt.savefig('comparacao_datasets_wagner.png', dpi=300, bbox_inches='tight')
    
    # Mostrar o plot
    plt.show()
    
    return counts1, counts_augmented

def print_comparison_stats(dataset1_counts, dataset2_counts, common_classes):
    """
    Imprime estatísticas de comparação entre os datasets
    """
    print("\n" + "="*70)
    print("COMPARAÇÃO: DATASET ORIGINAL vs DATASET BALANCEADO")
    print("="*70)
    
    print(f"\nClasses comuns encontradas: {len(common_classes)}")
    print(f"Classes: {', '.join(common_classes)}")
    
    total1 = sum(dataset1_counts[cls] for cls in common_classes)
    total2 = sum(dataset2_counts[cls] for cls in common_classes)
    total_augmented = total2 - total1
    
    print(f"\nTOTAL Dataset de Wagner Curado (Original): {total1} imagens")
    print(f"TOTAL Dataset de Wagner Curado e Balanceado: {total2} imagens")
    print(f"IMAGENS ADICIONADAS (Data Augmentation): {total_augmented} imagens")
    print(f"FATOR DE MULTIPLICAÇÃO: {total2/total1:.1f}x")
    
    print(f"\nDETALHES POR CLASSE:")
    print("-" * 65)
    print(f"{'Classe':<25} | {'Original':<8} | {'Augmented':<10} | {'Total':<8} | {'Fator'}")
    print("-" * 65)
    for cls in common_classes:
        count1 = dataset1_counts[cls]
        count2 = dataset2_counts[cls]
        augmented = count2 - count1
        factor = count2 / count1 if count1 > 0 else 0
        print(f"{cls:<25} | {count1:>8d} | {augmented:>10d} | {count2:>8d} | {factor:>5.1f}x")

def main():
    # Caminhos para os datasets
    dataset1_path = "datasets/dataset_wagner_v2"
    dataset2_path = "datasets/dataset_wagner_v2_wo_grade_5_balanceado"
    
    # Contar imagens em ambos os datasets
    print("Contando imagens no dataset original...")
    dataset1_counts = count_images_in_folders(dataset1_path)
    
    print("Contando imagens no dataset balanceado (com data augmentation)...")
    dataset2_counts = count_images_in_folders(dataset2_path)
    
    # Identificar classes comuns
    common_classes = get_common_classes(dataset1_counts, dataset2_counts)
    
    # Imprimir estatísticas
    print_comparison_stats(dataset1_counts, dataset2_counts, common_classes)
    
    # Criar o plot de barras empilhadas
    print("\nGerando plot de barras empilhadas...")
    create_stacked_bar_plot(dataset1_counts, dataset2_counts, common_classes)
    
    print("\nPlot salvo como 'comparacao_datasets_wagner.png'")
    print("Barra inferior (verde escuro): Imagens originais")
    print("Barra superior (verde claro): Imagens geradas por data augmentation")

if __name__ == "__main__":
    main() 