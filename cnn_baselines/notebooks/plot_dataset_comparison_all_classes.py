import matplotlib.pyplot as plt
import os
from pathlib import Path

def count_images_in_folders(dataset_path):
    """
    Counts the number of images (jpg, jpeg, png) in each folder of the dataset
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

def get_all_classes(dataset1_counts, dataset2_counts):
    """
    Returns all classes that exist in at least one of the datasets
    """
    all_classes = []
    all_possible_classes = [
        "Normal(Healthy skin)",
        "Grade 0", 
        "Grade 1",
        "Grade 2", 
        "Grade 3",
        "Grade 4",
        "Grade 5"
    ]
    
    for class_name in all_possible_classes:
        if (dataset1_counts.get(class_name, 0) > 0 or 
            dataset2_counts.get(class_name, 0) > 0):
            all_classes.append(class_name)
    
    return all_classes

def create_stacked_bar_plot(dataset1_counts, dataset2_counts, all_classes):
    """
    Creates a stacked bar plot comparing the two datasets
    """
    # Preparar os dados para todas as classes
    folders = all_classes
    counts1 = [dataset1_counts.get(folder, 0) for folder in folders]
    # Para o segundo dataset, calculamos apenas as imagens adicionadas (augmentation)
    counts_augmented = [max(0, dataset2_counts.get(folder, 0) - dataset1_counts.get(folder, 0)) for folder in folders]
    
    # Configurar o plot
    plt.figure(figsize=(14, 8))
    
    # Create stacked bars
    bars1 = plt.bar(folders, counts1, color='#3A7F58', alpha=0.8, label='Curated Wagner Dataset')
    bars2 = plt.bar(folders, counts_augmented, bottom=counts1, color='#90EE90', alpha=0.8, label='Curated Wagner Dataset with Data Augmentation')
    
    # Adicionar números nas barras
    for i, (bar1, bar2, count1, count_aug) in enumerate(zip(bars1, bars2, counts1, counts_augmented)):
        # Número no meio da primeira barra (imagens originais)
        if count1 > 0:
            plt.text(bar1.get_x() + bar1.get_width()/2., count1/2,
                    str(count1), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        # Number of added images
        if count_aug > 0:
            # Place number in the middle of the second bar for all classes
            plt.text(bar2.get_x() + bar2.get_width()/2., count1 + count_aug/2,
                    str(count_aug), ha='center', va='center', fontsize=11, fontweight='bold', color='black')
            # Total at the top for all classes
            total = count1 + count_aug
            plt.text(bar1.get_x() + bar1.get_width()/2., total + 10,
                    f'Total: {total}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            # If there are no added images, only show the total
            total = count1
            if total > 0:
                plt.text(bar1.get_x() + bar1.get_width()/2., total + 10,
                        f'Total: {total}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Marcar classes que existem apenas em um dataset
        dataset1_total = dataset1_counts.get(folders[i], 0)
        dataset2_total = dataset2_counts.get(folders[i], 0)
        
        # Removido: anotação para classes que existem apenas em um dataset
        # (não há mais espaço negativo para posicionar essas anotações)
    
    # Configure the plot
    plt.title('Databse - Curated vs Curated with Data Augmentation', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    
    # Ajustar os limites do eixo Y para dar mais espaço para a legenda e números no topo
    max_height = max([c1 + c2 for c1, c2 in zip(counts1, counts_augmented)]) if any(c1 + c2 > 0 for c1, c2 in zip(counts1, counts_augmented)) else 100
    plt.ylim(0, max_height * 1.25)  # Removido espaço negativo
    
    # Adicionar legenda no canto superior direito, aproveitando o espaço criado
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11, 
               framealpha=0.9, fancybox=True, shadow=True, ncol=1)
    
    # Adicionar grid
    plt.grid(True, alpha=0.3)
    
    # Rotacionar labels do eixo x para melhor legibilidade
    plt.xticks(rotation=45, ha='right')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('wagner_datasets_comparison_all_classes.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    return counts1, counts_augmented

def print_comparison_stats(dataset1_counts, dataset2_counts, all_classes):
    """
    Prints comparison statistics between the datasets
    """
    print("\n" + "="*75)
    print("COMPARISON: ORIGINAL WAGNER DATASET vs CURATED")
    print("="*75)
    
    print(f"\nAll classes found: {len(all_classes)}")
    print(f"Classes: {', '.join(all_classes)}")
    
    # Separate classes by availability
    only_dataset1 = []
    only_dataset2 = []
    both_datasets = []
    
    for cls in all_classes:
        count1 = dataset1_counts.get(cls, 0)
        count2 = dataset2_counts.get(cls, 0)
        
        if count1 > 0 and count2 == 0:
            only_dataset1.append(cls)
        elif count1 == 0 and count2 > 0:
            only_dataset2.append(cls)
        elif count1 > 0 and count2 > 0:
            both_datasets.append(cls)
    
    print(f"\nCLASSES PRESENT IN BOTH DATASETS: {len(both_datasets)}")
    if both_datasets:
        print(f"  {', '.join(both_datasets)}")
    
    print(f"\nCLASSES ONLY IN ORIGINAL DATASET: {len(only_dataset1)}")
    if only_dataset1:
        print(f"  {', '.join(only_dataset1)}")
    
    print(f"\nCLASSES ONLY IN CURATED DATASET: {len(only_dataset2)}")
    if only_dataset2:
        print(f"  {', '.join(only_dataset2)}")
    
    # Totals
    total1 = sum(dataset1_counts.get(cls, 0) for cls in all_classes)
    total2 = sum(dataset2_counts.get(cls, 0) for cls in all_classes)
    total_difference = total2 - total1
    
    print(f"\nTOTAL Original Wagner Dataset: {total1} images")
    print(f"TOTAL Curated Wagner Dataset: {total2} images")
    print(f"DIFFERENCE (Curated - Original): {total_difference:+d} images")
    if total1 > 0:
        print(f"CHANGE FACTOR: {total2/total1:.1f}x")
    
    print(f"\nDETAILS BY CLASS:")
    print("-" * 75)
    print(f"{'Class':<25} | {'Original':<8} | {'Curated':<10} | {'Difference':<10} | {'Status'}")
    print("-" * 75)
    for cls in all_classes:
        count1 = dataset1_counts.get(cls, 0)
        count2 = dataset2_counts.get(cls, 0)
        difference = count2 - count1
        
        if count1 > 0 and count2 == 0:
            status = "Removed"
        elif count1 == 0 and count2 > 0:
            status = "Added"
        elif count1 > 0 and count2 > 0:
            if difference > 0:
                status = f"+{difference}"
            elif difference < 0:
                status = f"{difference}"
            else:
                status = "Equal"
        else:
            status = "Empty"
        
        print(f"{cls:<25} | {count1:>8d} | {count2:>10d} | {difference:>+10d} | {status}")

def main():
    # Paths to datasets
    dataset1_path = "../datasets/dataset_wagner_v4"
    dataset2_path = "../datasets/dataset_wagner_v4_balanceado"
    
    # Count images in both datasets
    print("Counting images in Wagner v1 dataset...")
    dataset1_counts = count_images_in_folders(dataset1_path)
    
    print("Counting images in Wagner v2 dataset...")
    dataset2_counts = count_images_in_folders(dataset2_path)
    
    # Identify all classes
    all_classes = get_all_classes(dataset1_counts, dataset2_counts)
    
    # Print statistics
    print_comparison_stats(dataset1_counts, dataset2_counts, all_classes)
    
    # Create stacked bar plot
    print("\nGenerating stacked bar plot...")
    create_stacked_bar_plot(dataset1_counts, dataset2_counts, all_classes)
    
    print("\nPlot saved as 'wagner_datasets_comparison_all_classes.png'")
    print("Lower bar (dark green): Images from original dataset")
    print("Upper bar (light green): Images added in curated dataset")
    print("Special annotations: Classes that exist only in one version")

if __name__ == "__main__":
    main() 