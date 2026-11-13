import matplotlib.pyplot as plt
import os
from pathlib import Path

def count_images_in_folders(dataset_path):
    """
    Counts the number of images (jpg, jpeg, png) in each folder of the dataset
    """
    image_count = {}
    
    # List of folders in the dataset
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

def create_bar_plot(image_count):
    """
    Creates a bar plot of the image distribution by class
    """
    # Prepare the data
    folders = list(image_count.keys())
    counts = list(image_count.values())
    
    # Configure the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(folders, counts, color='#3A7F58', alpha=0.8)  # Darker green (SeaGreen)
    
    # Add numbers above the bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Configure the plot
    plt.title('Number of Images by Class', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('wagner_dataset_distribution.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    return counts

def main():
    # Path to dataset
    dataset_path = "datasets/dataset_wagner_v1"
    
    # Count images
    print("Counting images in folders...")
    image_count = count_images_in_folders(dataset_path)
    
    # Display count
    print("\nImage distribution:")
    for folder, count in image_count.items():
        print(f"{folder}: {count} images")
    
    # Create the bar plot
    print("\nGenerating bar plot...")
    create_bar_plot(image_count)
    
    print("Plot saved as 'wagner_dataset_distribution.png'")

if __name__ == "__main__":
    main()
