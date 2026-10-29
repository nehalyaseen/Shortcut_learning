
import matplotlib.pyplot as plt
import torch

def show_first_images_of_label(loader, title, target_label=9, num_images=20):
    images_list = []
    labels_list = []
    
    # Iterate batches until you have enough images of target_label
    for images, labels in loader:
        # Select images with label == target_label
        mask = (labels == target_label)
        filtered_images = images[mask]
        filtered_labels = labels[mask]
        
        if filtered_images.size(0) > 0:
            images_list.append(filtered_images)
            labels_list.append(filtered_labels)
        
        total_collected = sum(x.size(0) for x in images_list)
        if total_collected >= num_images:
            break

    # Concatenate all collected images/labels and limit to num_images
    images_all = torch.cat(images_list)[:num_images]
    labels_all = torch.cat(labels_list)[:num_images]
    
    fig, axes = plt.subplots(1, num_images, figsize=(num_images*1.2, 1.5))
    fig.suptitle(f"{title} - Label {target_label}", fontsize=14)
    
    for i in range(num_images):
        img = images_all[i].permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
        axes[i].imshow(img)
        axes[i].set_title(str(labels_all[i].item()))
        axes[i].axis("off")
    
    plt.savefig(f"{title.replace(' ', '_')}_label{target_label}.png")
    plt.close()