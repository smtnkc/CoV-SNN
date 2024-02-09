from transformers import RobertaModel, RobertaConfig

# Load pre-trained RoBERTa model and configuration
model_name = 'hunarbatra/CoVBERT'  # You can change this to other RoBERTa variants
roberta_config = RobertaConfig.from_pretrained(model_name)
roberta_model = RobertaModel.from_pretrained(model_name)
num_params = sum(p.numel() for p in roberta_model.parameters())
print("Number of parameters in model: ", num_params)
print(roberta_model)

# Print model architecture, class type, and number of parameters for each layer
for i, (name, layer) in enumerate(roberta_model.named_modules()):
    if hasattr(layer, 'weight'):
        num_params = sum(p.numel() for p in layer.parameters())
        print(f"Layer {i}: {name}, Type: {layer.__class__.__name__}, Parameters: {num_params}")