import torch

def extract_folded_weights(model):
    with torch.no_grad():
        W, b = model.hypernet(model.z.expand(1, -1))

    state_dict = {}
    for i, (w, bias) in enumerate(zip(W, b)):
        # Folded weights shape: [1, out, in], squeeze batch dim
        state_dict[f'layers.{2*i}.weight'] = w.squeeze(0)
        state_dict[f'layers.{2*i}.bias'] = bias.squeeze(0)
    return state_dict

if __name__ == "__main__":
    from models.metalearner import MetaLearnerLarge
    model = MetaLearnerLarge()
    model.load_state_dict(torch.load("C2_Hypernet/version3/checkpoints/full_model.pt"))
    weights = extract_folded_weights(model)
    torch.save(weights, "vanilla_target_weights.pt")
    print("✅ Vanilla weights extracted and saved.")
