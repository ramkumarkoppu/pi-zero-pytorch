import torch
import time
from pi_zero_pytorch import PiZero
from safetensors.torch import load_file
import os

# Step 0: Detect the Device
device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# Step 1: Manually Load Pretrained Weights
base_path = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c/"
)

# Load weights from safetensors files
pretrained_weights = {}
for i in range(1, 4):  # Loop over model-00001, model-00002, model-00003
    part_file = f"model-0000{i}-of-00003.safetensors"
    pretrained_weights.update(load_file(os.path.join(base_path, part_file), device=device))

print("Pretrained weights loaded.")

# Step 2: Initialize the π₀ Model
model = PiZero(
    dim=512,  # Embedding dimension for vision and language
    dim_action_input=6,  # Action input dimension
    dim_joint_state=12,  # Joint state input dimension
    num_tokens=20_000    # Vocabulary size
).to(device)

print("Keys in pretrained weights:")
print(pretrained_weights.keys())
print("Keys in PiZero model state dict:")
print(model.state_dict().keys())

# Step 3: Load Weights into the π₀ Model
# Use the updated `load_pretrained_vlm_weights_` method
model.load_pretrained_vlm_weights_(pretrained_weights)
print("Weights successfully loaded into the PiZero model.")

# Step 4: Save the Updated π₀ Model
model_save_path = 'pi_zero_with_paligemma_weights.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Updated PiZero model saved as {model_save_path}")

# Step 5: Test the Updated π₀ Model
# Prepare dummy inputs on the detected device
vision = torch.randn(1, 1024, 512, device=device)  # Batch size of 1, sequence length of 1024, embedding size of 512
commands = torch.randint(0, 20_000, (1, 1024), device=device)  # Batch size of 1, sequence length of 1024
joint_state = torch.randn(1, 12, device=device)  # Batch size of 1, joint state dimension of 12
actions = torch.randn(1, 32, 6, device=device)  # Batch size of 1, trajectory length of 32, action dimension of 6

# Perform a forward pass to compute the loss
model.eval()
start = time.time()
loss, _ = model(vision, commands, joint_state, actions)
loss.backward()  # Backpropagation (useful during training)

# Generate sampled actions from the model
with torch.no_grad():
    sampled_actions = model(vision, commands, joint_state, trajectory_length=32)  # (1, 32, 6)

end = time.time()

# Display the results
print(f"Sampled actions: {sampled_actions}")
print(f"Type of sampled actions: {type(sampled_actions)}")
print(f"Shape of sampled actions: {sampled_actions.shape}")
print(f"Time taken: {round(end - start)} seconds")
