import torch
from pi_zero_pytorch import π0

def test_pi_zero_with_vit():
    from vit_pytorch import ViT
    from vit_pytorch.extractor import Extractor

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    v = Extractor(v, return_embeddings_only = True)

    model = π0(
        dim = 512,
        vit = v,
        vit_dim = 1024,
        dim_action_input = 6,
        dim_joint_state = 12,
        num_tokens = 20_000
    )

    vision = torch.randn(2, 1024, 512)

    images = torch.randn(2, 3, 2, 256, 256)

    commands = torch.randint(0, 20_000, (2, 1024))
    joint_state = torch.randn(2, 12)
    actions = torch.randn(2, 32, 6)

    loss, _ = model(images, commands, joint_state, actions)
    loss.backward()

    # after much training

    sampled_actions = model(images, commands, joint_state, trajectory_length = 32) # (1, 32, 6)

    assert sampled_actions.shape == (2, 32, 6)
