import pandas as pd
from transformers import RobertaForMaskedLM, RobertaModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import RobertaForMaskedLM, RobertaModel, RobertaConfig
import random
from captum.attr import LayerIntegratedGradients
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

CHECKPOINT=sys.argv[1]
DISTANCE= sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sbert_model = SentenceTransformer("../outputs/ContrastiveLoss_omicron_vs_delta_Pmax_R0.2" + \
                                f"_D0.1_E10_LR_0.001_B32_M2.0/checkpoints/checkpoint-{CHECKPOINT}")

transformer_model = sbert_model[0].auto_model.to(device)  # Extract the core transformer model

# Get the configuration of the SentenceTransformer model
sbert_config = transformer_model.config

# Create a custom RobertaConfig with the same parameters
custom_config = RobertaConfig(
    vocab_size=sbert_config.vocab_size,
    hidden_size=sbert_config.hidden_size,
    num_hidden_layers=sbert_config.num_hidden_layers,
    num_attention_heads=sbert_config.num_attention_heads,
    intermediate_size=sbert_config.intermediate_size,
    max_position_embeddings=sbert_config.max_position_embeddings,
    type_vocab_size=sbert_config.type_vocab_size,
    initializer_range=sbert_config.initializer_range,
    layer_norm_eps=sbert_config.layer_norm_eps,
    hidden_dropout_prob=sbert_config.hidden_dropout_prob,
    attention_probs_dropout_prob=sbert_config.attention_probs_dropout_prob,
)

# Initialize RobertaForMaskedLM and RobertaModel with the custom configuration
masked_lm_model = RobertaForMaskedLM(custom_config).to(device)
embedding_model = RobertaModel(custom_config).to(device)
tokenizer = sbert_model[0].tokenizer

# Copy the weights from transformer_model to masked_lm_model and embedding_model
transformer_model_state_dict = transformer_model.state_dict()
masked_lm_model.roberta.load_state_dict(transformer_model_state_dict, strict=False)
embedding_model.load_state_dict(transformer_model_state_dict, strict=False)

print(f"Weights successfully copied from checkpoint {CHECKPOINT}!")

# Set the models in evaluation mode
masked_lm_model.eval()
embedding_model.eval()

def mask_sentence(sentence):
    tokens = tokenizer.tokenize(sentence)
    masked_sentences = []
    for i in range(len(tokens)):
        masked_tokens = tokens.copy()
        masked_tokens[i] = '<mask>'
        masked_sentence = ' '.join(masked_tokens)
        masked_sentences.append(masked_sentence)
    return masked_sentences

def predict_masked_words(masked_sentences):
    predictions = []
    for masked_sentence in masked_sentences:
        inputs = tokenizer.encode(masked_sentence, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = masked_lm_model(inputs)
            predictions.append(outputs.logits)
    return predictions

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.squeeze().cpu().numpy()

def calculate_semantic_change(average_embedding, target_sentence):
    target_embedding = get_sentence_embedding(target_sentence)
    if DISTANCE == "L2":
        semantic_change = np.linalg.norm(average_embedding - target_embedding) # Euclidean distance
    elif DISTANCE == "L1":
        semantic_change = np.linalg.norm(average_embedding - target_embedding, ord=1) # Manhattan distance
    return semantic_change

def calculate_sequence_probability(predictions, masked_sentences, target_sentence):
    total_score = 0
    tokens = tokenizer.tokenize(target_sentence)
    for i, logits in enumerate(predictions):
        masked_index = masked_sentences[i].split().index('<mask>')
        softmax = torch.nn.functional.softmax(logits[0, masked_index], dim=-1)
        original_token_id = tokenizer.convert_tokens_to_ids(tokens[i])
        score = softmax[original_token_id].item()
        total_score += score
    average_score = total_score / len(masked_sentences)
    return average_score

def calculate_inverse_perplexity(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)
    with torch.no_grad():
        outputs = masked_lm_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return 1/perplexity.item()


# Load target data
eris = pd.read_csv("../data/unique_Eris_2k.csv")["sequence"].tolist()[:2000]
delta = pd.read_csv("../data/unique_Delta_2k.csv")["sequence"].tolist()[:2000]

average_embedding = np.load("../data/average_omicron_embedding.npy")
print(f"Average Omicron embedding loaded from file with shape {average_embedding.shape}")

target_sentence = delta[1000]

masked_sentences = mask_sentence(target_sentence)
predictions = predict_masked_words(masked_sentences)
sp_score = calculate_sequence_probability(predictions, masked_sentences, target_sentence)
sc_score = calculate_semantic_change(average_embedding, target_sentence)
ip_score = calculate_inverse_perplexity(target_sentence)

print(f"sc score: {sc_score:.6f}")
print(f"sp score: {sp_score:.6f}")
print(f"ip score: {ip_score:.6f}")


### VISUALIZE ATTRIBUTION OF EACH LAYER TO SEMANTIC CHANGE
# Documentation: https://captum.ai/api/layer.html
# Tutorial: https://captum.ai/tutorials/Bert_SQUAD_Interpret

# Load average embedding from file
average_embeddings = np.load("../data/average_omicron_embedding.npy")
average_embeddings = torch.tensor(average_embeddings, device=device)
print(f"Average Omicron embedding shape {average_embedding.shape}")

# Take a random sequence from delta
target_seq = delta[1000]
target_seq_embeddings = get_sentence_embedding(target_seq)
print(f"Target sequence embedding shape {target_seq_embeddings.shape}")

# Wrapper classes for different metrics
class SemanticChangeModel(torch.nn.Module):
    def __init__(self, embedding_model, average_embedding):
        super().__init__()
        self.embedding_model = embedding_model
        self.register_buffer('average_embedding', average_embedding)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        distances = torch.norm(embeddings - self.average_embedding, p=2, dim=-1)
        # Return as a tensor with shape (batch_size, 1)
        return distances.sum(dim=-1).unsqueeze(-1)

class InversePerplexityModel(torch.nn.Module):
    def __init__(self, masked_lm_model):
        super().__init__()
        self.masked_lm_model = masked_lm_model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.masked_lm_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        # Return as a tensor with shape (batch_size, 1)
        return (1/perplexity).unsqueeze(0).unsqueeze(-1)

class SequenceProbabilityModel(torch.nn.Module):
    def __init__(self, masked_lm_model, tokenizer):
        super().__init__()
        self.masked_lm_model = masked_lm_model
        self.tokenizer = tokenizer
    
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        total_score = torch.zeros(batch_size, device=input_ids.device)
        
        for pos in range(seq_length):
            if attention_mask[0, pos] == 0:  # Skip padding tokens
                continue
            
            masked_input_ids = input_ids.clone()
            original_token = input_ids[0, pos].item()
            masked_input_ids[0, pos] = self.tokenizer.mask_token_id
            
            outputs = self.masked_lm_model(masked_input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probs = F.softmax(logits[0, pos], dim=-1)
            score = probs[original_token]
            total_score += score
        
        # Return as a tensor with shape (batch_size, 1)
        return (total_score / seq_length).unsqueeze(-1)

def calculate_layer_attributions(model, layer_module, inputs, n_steps=50):
    lig = LayerIntegratedGradients(model, layer_module)
    attributions, _ = lig.attribute(
        inputs=inputs['input_ids'],
        additional_forward_args=(inputs['attention_mask'],),
        internal_batch_size=1,
        n_steps=n_steps,
        return_convergence_delta=True
    )
    return torch.sum(torch.abs(attributions)).item()

def visualize_attributions(all_attributions, save_path):
    font_size = 22
    # Set global font size
    plt.rcParams.update({'font.size': font_size})
    
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot for each metric
    metrics = ['Semantic Change', 'Sequence Probability', 'Inverse Perplexity']
    axes = [ax1, ax2, ax3]
    
    for ax, metric, attributions in zip(axes, metrics, all_attributions):
        # Normalize attributions to sum to 1
        normalized_attributions = np.array(attributions)
        normalized_attributions = normalized_attributions / np.sum(normalized_attributions)
        
        # Create bar plot
        ax.bar(range(len(normalized_attributions)), normalized_attributions)
        ax.set_xlabel('Layer Index', fontsize=font_size)
        ax.set_ylabel('Normalized Total Attribution', fontsize=font_size)

        title_map = {
            'Semantic Change': 'Semantic Change',
            'Sequence Probability': 'Token-level log-likelihood', #r'Grammaticality ($\lambda_T$)',
            'Inverse Perplexity': 'Average log-likelihood', #r'Grammaticality ($\lambda_A$)'
        }

        ax.set_title(f'{title_map[metric]}', fontsize=font_size)
        
        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=font_size)

        # Show all x-ticks
        ax.set_xticks(range(len(normalized_attributions)))
        
        if CHECKPOINT == "0":
            step_size = 0.1
        else:  
            step_size = 0.2

        ax.yaxis.set_major_locator(plt.MultipleLocator(step_size))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Create model wrappers
sc_model = SemanticChangeModel(embedding_model, average_embeddings).to(device)
ip_model = InversePerplexityModel(masked_lm_model).to(device)
sp_model = SequenceProbabilityModel(masked_lm_model, tokenizer).to(device)

# Set models to evaluation mode
sc_model.eval()
ip_model.eval()
sp_model.eval()

# Prepare input
inputs = tokenizer(target_seq, return_tensors='pt', truncation=True, 
                  padding='max_length', max_length=128).to(device)

# Collect all attributions
all_metric_attributions = []
metrics = {
    'Semantic Change': sc_model,
    'Sequence Probability': sp_model,
    'Inverse Perplexity': ip_model
}

for metric_name, model in metrics.items():
    layer_attributions = []
    
    # Calculate attributions for each layer
    for layer_idx in range(model.embedding_model.config.num_hidden_layers 
                          if hasattr(model, 'embedding_model') 
                          else model.masked_lm_model.config.num_hidden_layers):
        
        if hasattr(model, 'embedding_model'):
            layer = model.embedding_model.encoder.layer[layer_idx]
        else:
            layer = model.masked_lm_model.roberta.encoder.layer[layer_idx]
        
        attr_score = calculate_layer_attributions(model, layer, inputs)
        layer_attributions.append(attr_score)
    
    all_metric_attributions.append(layer_attributions)
    print(f"Finished calculating attributions for {metric_name}")

# Create single visualization with all three plots
visualize_attributions(all_metric_attributions, f'figures/layer_attributions_comparison_{CHECKPOINT}.pdf')
print(f"Layer attribution comparison visualization saved as 'figures/layer_attributions_comparison_{CHECKPOINT}.pdf'")


