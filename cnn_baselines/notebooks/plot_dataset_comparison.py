import matplotlib.pyplot as plt
import os
from pathlib import Path

def count_images_in_folders(dataset_path):
    """
    Counts the number of images (jpg, jpeg, png) in each folder of the dataset
    """
    image_count = {}
    
    # List of possible folders in the dataset
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
            # Count image files (jpg, jpeg, png)
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
    Returns only the classes that exist in both datasets
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
    Creates a stacked bar plot comparing the two datasets
    """
    # Prepare data only for common classes
    folders = common_classes
    counts1 = [dataset1_counts[folder] for folder in folders]
    # For the second dataset, we calculate only the added images (augmentation)
    counts_augmented = [dataset2_counts[folder] - dataset1_counts[folder] for folder in folders]
    
    # Configure the plot
    plt.figure(figsize=(14, 8))
    
    # Create stacked bars
    bars1 = plt.bar(folders, counts1, color='#3A7F58', alpha=0.8, label='Curated Wagner Dataset')
    bars2 = plt.bar(folders, counts_augmented, bottom=counts1, color='#90EE90', alpha=0.8, label='Added Images (Data Augmentation)')
    
    # Add numbers to the bars
    for i, (bar1, bar2, count1, count_aug) in enumerate(zip(bars1, bars2, counts1, counts_augmented)):
        # Number in the middle of the first bar (original images)
        if count1 > 0:
            plt.text(bar1.get_x() + bar1.get_width()/2., count1/2,
                    str(count1), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        # Number in the middle of the second bar (augmented images)
        if count_aug > 0:
            plt.text(bar2.get_x() + bar2.get_width()/2., count1 + count_aug/2,
                    str(count_aug), ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        
        # Total at the top
        total = count1 + count_aug
        plt.text(bar1.get_x() + bar1.get_width()/2., total + 10,
                f'Total: {total}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Configure the plot
    plt.title('Dataset Comparison - Original vs Balanced using Data Augmentation', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    
    # Adjust Y-axis limits to give more space for the legend and numbers at the top
    max_height = max([c1 + c2 for c1, c2 in zip(counts1, counts_augmented)])
    plt.ylim(0, max_height * 1.25)  # 25% extra space at the top (more space)
    
    # Add legend to the upper right corner, using the created space
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11, 
               framealpha=0.9, fancybox=True, shadow=True, ncol=1)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('wagner_datasets_comparison.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    return counts1, counts_augmented

def print_comparison_stats(dataset1_counts, dataset2_counts, common_classes):
    """
    Prints comparison statistics between the datasets
    """
    print("\n" + "="*70)
    print("COMPARISON: ORIGINAL DATASET vs BALANCED DATASET")
    print("="*70)
    
    print(f"\nCommon classes found: {len(common_classes)}")
    print(f"Classes: {', '.join(common_classes)}")
    
    total1 = sum(dataset1_counts[cls] for cls in common_classes)
    total2 = sum(dataset2_counts[cls] for cls in common_classes)
    total_augmented = total2 - total1
    
    print(f"\nTOTAL Curated Wagner Dataset: {total1} images")
    print(f"TOTAL Curated and Balanced Wagner Dataset: {total2} images")
    print(f"ADDED IMAGES (Data Augmentation): {total_augmented} images")
    print(f"MULTIPLICATION FACTOR: {total2/total1:.1f}x")
    
    print(f"\nDETAILS BY CLASS:")
    print("-" * 65)
    print(f"{'Class':<25} | {'Original':<8} | {'Augmented':<10} | {'Total':<8} | {'Factor'}")
    print("-" * 65)
    for cls in common_classes:
        count1 = dataset1_counts[cls]
        count2 = dataset2_counts[cls]
        augmented = count2 - count1
        factor = count2 / count1 if count1 > 0 else 0
        print(f"{cls:<25} | {count1:>8d} | {augmented:>10d} | {count2:>8d} | {factor:>5.1f}x")

def main():
    # Paths to datasets
    dataset1_path = "../datasets/dataset_wagner_v1"
    dataset2_path = "../datasets/dataset_wagner_v4"
    
    # Count images in both datasets
    print("Counting images in the original dataset...")
    dataset1_counts = count_images_in_folders(dataset1_path)
    
    print("Counting images in the balanced dataset (with data augmentation)...")
    dataset2_counts = count_images_in_folders(dataset2_path)
    
    # Identify common classes
    common_classes = get_common_classes(dataset1_counts, dataset2_counts)
    
    # Print statistics
    print_comparison_stats(dataset1_counts, dataset2_counts, common_classes)
    
    # Create stacked bar plot
    print("\nGenerating stacked bar plot...")
    create_stacked_bar_plot(dataset1_counts, dataset2_counts, common_classes)
    
    print("\nPlot saved as 'wagner_datasets_comparison.png'")
    print("Lower bar (dark green): Original images")
    print("Upper bar (light green): Images generated by data augmentation")

if __name__ == "__main__":
    main() 