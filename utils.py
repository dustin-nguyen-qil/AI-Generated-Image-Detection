import matplotlib.pyplot as plt
import torch 

def texture_diversity(patch):
    """
        Compute texture diversity of a patch by:
        sum of residuals of the gradients along four directions: 
            diagonal, counter-diagonal, horizontal, and vertical
    """
    # convert the patch tensor to grayscale
    grayscale_patch = torch.mean(patch, dim=2)

    # gradients along four directions: diagonal, counter-diagonal, horizontal, and vertical  
    gradients = [
        torch.abs(grayscale_patch[:-1, :-1] - grayscale_patch[1:, 1:]),  # Diagonal
        torch.abs(grayscale_patch[:-1, 1:] - grayscale_patch[1:, :-1]),  # Counter-diagonal
        torch.abs(grayscale_patch[:, :-1] - grayscale_patch[:, 1:]),      # Horizontal
        torch.abs(grayscale_patch[:-1, :] - grayscale_patch[1:, :])       # Vertical
    ]

    # sum the residuals of the gradients
    diversity = sum(torch.sum(grad) for grad in gradients)

    return diversity.item()

def break_into_patches(image, patch_size):
    """
        Break image into patches 
    """
    _, height, width = image.shape
    
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    
    # initialize a list to store the patches
    patches = []
    
    # iterate through each patch and extract it from the image tensor
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            patch = image[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)
    
    return patches

def get_rich_and_poor_patches(img_tensor, patch_size):
    """
        Get 64 richest patches and 64 poorest patches 
    """
    # break the image and get patch
    patches = break_into_patches(img_tensor, patch_size)
    
    # compute texture diversity for each patch
    diversity = {i: texture_diversity(patch) for (i, patch) in enumerate(patches)}
    # sort the patches based on diversity
    diversity = dict(sorted(diversity.items(), key=lambda item: item[1]))

    # get 64 rich patches and 64 poor patches
    poor_patches_idx = [k for k in list(diversity.keys())[:64]] 
    rich_patches_idx = [k for k in list(diversity.keys())[-64:]]
    poor_patches = [patches[i] for i in poor_patches_idx]
    rich_patches = [patches[j] for j in rich_patches_idx]


    return rich_patches, poor_patches

def get_rich_and_poor_images(img_tensor, patch_size):
    """
        Reconstruct rich texture image and poor texture image from rich patches and poor patches
    """
    rich_patches, poor_patches = get_rich_and_poor_patches(img_tensor, patch_size)

    rich_image_tensor = torch.zeros(3, 256, 256)
    poor_image_tensor = torch.zeros(3, 256, 256)

    num_patches = len(poor_patches)
    patches_per_side = int(num_patches**0.5)

    # iterate through the patches and place them in the output reconstructed image
    def combine_patches_into_image(patches, out_image_tensor):
        for i, patch in enumerate(patches):
            row = i // patches_per_side
            col = i % patches_per_side
            start_row = row * patch_size
            start_col = col * patch_size
            out_image_tensor[:, start_row:start_row+patch_size, start_col:start_col+patch_size] = patch

        return out_image_tensor

    rich_image_tensor = combine_patches_into_image(rich_patches, rich_image_tensor)
    poor_image_tensor = combine_patches_into_image(poor_patches, poor_image_tensor)

    return rich_image_tensor, poor_image_tensor

def plot_rich_poor_patch(original_image_tensor, rich_image_tensor, poor_image_tensor):
    original_image_np = original_image_tensor.permute(1, 2, 0).numpy()
    rich_image_np = rich_image_tensor.permute(1, 2, 0).numpy()
    poor_image_np = poor_image_tensor.permute(1, 2, 0).numpy()

    # Plotting
    plt.figure(figsize=(6, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image_np)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(poor_image_np)
    plt.title("Poor texture image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(rich_image_np)
    plt.title("Rich texture image")
    plt.axis('off')

    plt.show()

# Test
if __name__ == '__main__':
    img = torch.rand(3, 512, 512)
    rich, poor = get_rich_and_poor_images(img, 32)
    print(rich.size(), poor.size())