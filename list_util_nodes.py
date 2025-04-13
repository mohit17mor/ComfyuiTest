import torch

class ImagePairCombiner:
    """
    Takes two lists of images (e.g., characters and outfits) and outputs
    two corresponding batches representing all pairwise combinations.
    - Output 1 repeats each character image for every outfit.
    - Output 2 repeats the sequence of outfit images for every character.
    The i-th image in Output 1 pairs with the i-th image in Output 2.

    NO RESIZING is performed. Assumes all images within the first input list
    share the same dimensions, and all images within the second list share
    the same dimensions (but the two lists can have different dimensions).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images_characters": ("IMAGE", ), # List of character images/batches
                "images_outfits": ("IMAGE", ),    # List of outfit images/batches
            }
        }

    INPUT_IS_LIST = True # Both inputs are lists of tensors

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("combined_characters", "combined_outfits")
    FUNCTION = "combine_pairs"

    CATEGORY = "SimpleCombinations" # Choose your category

    def _flatten_and_check_shape(self, image_list, name):
        """Helper to flatten list of batches and ensure consistent shape."""
        if not image_list:
            return None, None # Return None if list is empty

        all_images = []
        ref_shape = None
        device = None
        dtype = None

        for i, batch in enumerate(image_list):
            if batch is None:
                print(f"Warning: Found None item at index {i} in '{name}' list, skipping.")
                continue
            if batch.shape[0] == 0:
                # print(f"Warning: Found empty batch at index {i} in '{name}' list, skipping.") # Less verbose
                continue # Skip empty batches

            if ref_shape is None:
                ref_shape = batch.shape[1:] # Get H, W, C from first valid batch
                device = batch.device
                dtype = batch.dtype
                print(f"Reference shape for '{name}' set to {ref_shape} on device {device}")

            # Check subsequent batches
            elif batch.shape[1:] != ref_shape:
                raise ValueError(f"Dimension mismatch in '{name}' list: "
                                 f"Expected {ref_shape} (H, W, C), but found "
                                 f"{batch.shape[1:]} in batch at index {i}.")
            elif batch.device != device or batch.dtype != dtype:
                 raise ValueError(f"Device/dtype mismatch in '{name}' list: "
                                 f"Expected {dtype} on {device}, but found "
                                 f"{batch.dtype} on {batch.device} in batch at index {i}.")

            # Add individual images from the batch to the list
            # Use list comprehension for efficiency if batches aren't huge
            # Or loop if memory is a concern for very large batches
            for j in range(batch.shape[0]):
                 all_images.append(batch[j:j+1]) # Keep shape (1, H, W, C)

        if not all_images:
             print(f"Warning: No valid images found in '{name}' list after processing.")
             return None, ref_shape # Return None if only empty/None batches found

        # Concatenate all individual images into a single batch
        final_batch = torch.cat(all_images, dim=0)
        print(f"Flattened '{name}' list into a single batch of shape: {final_batch.shape}")
        return final_batch, ref_shape


    def combine_pairs(self, images_characters, images_outfits):
        # 1. Flatten and validate characters list
        flat_chars, shape_chars = self._flatten_and_check_shape(images_characters, "images_characters")

        # 2. Flatten and validate outfits list
        flat_outfits, shape_outfits = self._flatten_and_check_shape(images_outfits, "images_outfits")

        # Handle cases where one or both lists were empty or invalid
        if flat_chars is None or flat_outfits is None:
            print("Error: One or both input lists contained no valid images. Cannot combine.")
            # Return empty tensors matching the shapes if possible, otherwise default empty
            empty_chars = torch.zeros((0,) + (shape_chars if shape_chars else (64, 64, 3)),
                                      dtype=torch.float32, device="cpu") # Use detected shape/dtype if possible
            empty_outfits = torch.zeros((0,) + (shape_outfits if shape_outfits else (64, 64, 3)),
                                        dtype=torch.float32, device="cpu")
            if flat_chars is not None: # Get device/dtype from valid input
                empty_chars = empty_chars.to(flat_chars.device, dtype=flat_chars.dtype)
            if flat_outfits is not None:
                empty_outfits = empty_outfits.to(flat_outfits.device, dtype=flat_outfits.dtype)

            return (empty_chars, empty_outfits)


        num_chars = flat_chars.shape[0]
        num_outfits = flat_outfits.shape[0]

        if num_chars == 0 or num_outfits == 0:
             print("Error: One or both lists resulted in zero images after flattening.")
             # Return empty tensors (already created above based on shape checks)
             return (flat_chars, flat_outfits) # Will be (0, H, W, C)


        print(f"Combining {num_chars} characters with {num_outfits} outfits.")
        print(f"Character shape: {shape_chars}, Outfit shape: {shape_outfits}")

        # 3. Repeat characters
        # Each char needs to be repeated num_outfits times consecutively
        # Example: A1, A1, A1, A2, A2, A2 (if 3 outfits)
        combined_characters = flat_chars.repeat_interleave(num_outfits, dim=0)

        # 4. Repeat outfits
        # The whole sequence of outfits needs to be repeated num_chars times
        # Example: B1, B2, B3, B1, B2, B3 (if 2 chars)
        combined_outfits = flat_outfits.repeat(num_chars, 1, 1, 1)

        print(f"Output characters shape: {combined_characters.shape}") # Should be (N*M, H, W, C)
        print(f"Output outfits shape: {combined_outfits.shape}")    # Should be (N*M, H', W', C')

        return (combined_characters, combined_outfits)

# ========================================
# --- Node Registration ---
# ComfyUI needs these dictionaries to find and display the node
NODE_CLASS_MAPPINGS = {
    "ImagePairCombiner": ImagePairCombiner
}

# Optional: A dictionary mapping internal class names to user-friendly display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePairCombiner": "ImagePairCombiner"
}