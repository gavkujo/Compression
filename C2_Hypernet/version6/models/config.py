# models/config.py
MODEL_CONFIGS = {
    "mnist_mlp": {
        "genome_dim": 64,
        "block_type": "mlp",
        "block_params": {
            "layer_dims": [784, 256, 128, 10],
            "hidden_dim": 128
        },
        "classifier": None
    },
    "mnist_attn": {
        "genome_dim": 64,
        "block_type": "attention",
        "block_params": {
            "embed_dim": 28,
            "num_heads": 4,
            "hidden_dim": 128,
            "tokens": 28
        },
        "classifier": {"in_dim": 28, "out_dim": 10}
    },
    # add more configs here...
}
