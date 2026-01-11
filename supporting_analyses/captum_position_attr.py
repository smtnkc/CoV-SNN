import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import random
from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from transformers import RobertaForMaskedLM

CHECKPOINT = sys.argv[1]
DISTANCE = "L2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load model
sbert_model = SentenceTransformer("../outputs/ContrastiveLoss_omicron_vs_delta_Pmax_R0.2" + \
                                f"_D0.1_E10_LR_0.001_B32_M2.0/checkpoints/checkpoint-{CHECKPOINT}")
transformer_model = sbert_model[0].auto_model.to(device)
tokenizer = sbert_model[0].tokenizer

# Create masked language model with same config
custom_config = transformer_model.config
masked_lm_model = RobertaForMaskedLM(custom_config).to(device)
# Copy weights from transformer model to masked LM model
masked_lm_model.roberta.load_state_dict(transformer_model.state_dict(), strict=False)
masked_lm_model.eval()

# Wrapper class for semantic change calculation
class SemanticChangeModel(torch.nn.Module):
    def __init__(self, model, average_embedding):
        super().__init__()
        self.model = model
        self.register_buffer('average_embedding', average_embedding)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        distances = torch.norm(embeddings - self.average_embedding, p=2, dim=-1)
        score = distances.mean()
        return score.reshape(1)

# Wrapper class for sequence probability calculation
class SequenceProbabilityModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model  # This should be masked_lm_model
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
            
            outputs = self.model(masked_input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probs = F.softmax(logits[0, pos], dim=-1)
            score = probs[original_token]
            total_score += score
        
        return (total_score / seq_length).reshape(1)

# Wrapper class for inverse perplexity calculation
class InversePerplexityModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # This should be masked_lm_model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        return (1/perplexity).reshape(1)

def calculate_position_attributions(model, inputs, target_seq_length, n_steps=50):
    # Get the embedding layer - handle both RobertaModel and RobertaForMaskedLM
    if hasattr(model.model, 'embeddings'):
        embedding_layer = model.model.embeddings.word_embeddings
    elif hasattr(model.model, 'roberta'):
        embedding_layer = model.model.roberta.embeddings.word_embeddings
    else:
        raise AttributeError("Could not find embeddings layer in model")
    
    def forward_func(input_ids, attention_mask):
        score = model(input_ids, attention_mask)
        return score.unsqueeze(0)
    
    lig = LayerIntegratedGradients(forward_func, embedding_layer)
    
    # Create baseline (pad token embeddings)
    baseline = torch.ones_like(inputs['input_ids']) * tokenizer.pad_token_id
    
    try:
        attributions, delta = lig.attribute(
            inputs=(inputs['input_ids'], inputs['attention_mask']),
            baselines=(baseline, inputs['attention_mask']),
            n_steps=n_steps,
            return_convergence_delta=True,
            internal_batch_size=1,
            target=0
        )
    except Exception as e:
        print(f"Attribution error: {e}")
        raise e
    
    # Sum attributions across embedding dimension
    token_attributions = torch.sum(torch.abs(attributions), dim=-1).squeeze(0)
    
    # Only consider the actual sequence positions (excluding special tokens and padding)
    # Add 1 to include the last token
    sequence_attributions = token_attributions[1:target_seq_length+1].detach().cpu().numpy()
    
    return sequence_attributions

def visualize_position_attributions(attributions_dict, save_path):
    # Set global font size
    plt.rcParams.update({'font.size': 16})
    
    plt.figure(figsize=(15, 5))
    
    metrics = ['Semantic Change', 'Sequence Probability', 'Inverse Perplexity']
    colors = ['blue', 'red', 'green']
    
    # Find the maximum sequence length across all metrics
    max_seq_length = max(len(attributions_dict[metric]) for metric in metrics)
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        plt.subplot(1, 3, idx+1)
        attributions = attributions_dict[metric]
        
        # Normalize attributions to sum to 1
        attributions = attributions / np.sum(attributions)
        
        # Convert positions to percentages
        positions = np.linspace(0, 100, len(attributions))
        
        # Calculate moving average for smoother visualization
        window_size = min(5, len(positions) // 4)  # Adjust window size based on sequence length
        if window_size < 1:
            window_size = 1
            
        smoothed_attributions = np.convolve(attributions, 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
        smoothed_positions = positions[window_size-1:]
        
        # Plot the smoothed line
        plt.plot(smoothed_positions, smoothed_attributions, color=color, linewidth=2)
        
        # Add confidence interval
        std_dev = []
        for i in range(len(attributions) - window_size + 1):
            std_dev.append(np.std(attributions[i:i+window_size]))
        std_dev = np.array(std_dev)
        
        plt.fill_between(smoothed_positions,
                         smoothed_attributions - std_dev,
                         smoothed_attributions + std_dev,
                         alpha=0.3,
                         color=color)
        
        plt.xlabel('Position as % of length', fontsize=16)
        plt.ylabel('Normalized Importance', fontsize=16)
        plt.title(f'{metric}', fontsize=16)
        
        # Set x-axis limits to show full percentage range
        plt.xlim(0, 100)
        
        # Set tick label size
        plt.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Load data
    delta = pd.read_csv("../data/unique_Delta_2k.csv")["sequence"].tolist()[:2000]
    
    # Load average embedding
    average_embedding = np.load("../data/average_omicron_embedding.npy")
    average_embedding = torch.tensor(average_embedding, device=device)
    
    # Create model wrappers
    sc_model = SemanticChangeModel(transformer_model, average_embedding).to(device)
    sp_model = SequenceProbabilityModel(masked_lm_model, tokenizer).to(device)
    ip_model = InversePerplexityModel(masked_lm_model).to(device)
    
    models = {
        'Semantic Change': sc_model,
        'Sequence Probability': sp_model,
        'Inverse Perplexity': ip_model
    }
    
    # Set all models to evaluation mode
    for model in models.values():
        model.eval()
    
    # Calculate attributions for multiple sequences and average them
    all_attributions = {metric: [] for metric in models.keys()}
    num_sequences = 0  # Number of sequences to analyze
    max_seq_length = 10  # Track maximum sequence length
    # random shuffle
    random.shuffle(delta)
    
    # First pass: calculate attributions and find max sequence length
    for idx in range(2000):
        target_seq = delta[idx]
        inputs = tokenizer(target_seq, return_tensors='pt', truncation=True,
                         padding='max_length', max_length=128).to(device)
        
        # Get actual sequence length (excluding special tokens)
        seq_length = len(tokenizer.tokenize(target_seq))
        if seq_length >= max_seq_length:
            continue

        try:
            # Calculate attributions for each metric
            for metric, model in models.items():
                attributions = calculate_position_attributions(model, inputs, seq_length)
                all_attributions[metric].append(attributions)
            print(f"Successfully processed sequence {idx}")
        except Exception as e:
            print(f"Error processing sequence {idx}: {e}")
            continue
        num_sequences += 1
        if num_sequences == 1000:
            break
    
    if not any(all_attributions.values()):
        raise RuntimeError("No successful attributions were calculated")
    
    # Pad sequences to max_seq_length
    for metric in models.keys():
        padded_attributions = []
        for attr in all_attributions[metric]:
            # Pad with zeros if needed
            if len(attr) < max_seq_length:
                padding = np.zeros(max_seq_length - len(attr))
                attr = np.concatenate([attr, padding])
            padded_attributions.append(attr[:max_seq_length])  # Trim if longer
        all_attributions[metric] = padded_attributions
    
    # Average attributions across all sequences for each metric
    avg_attributions = {
        metric: np.mean(np.array(attrs), axis=0)
        for metric, attrs in all_attributions.items()
        if attrs  # Only process metrics that have attributions
    }
    
    # Visualize the results
    visualize_position_attributions(avg_attributions, 
                                  f'figures/position_attributions_{CHECKPOINT}.pdf')
    print(f"Position attribution visualization saved as 'figures/position_attributions_{CHECKPOINT}.pdf'")

if __name__ == "__main__":
    main()
