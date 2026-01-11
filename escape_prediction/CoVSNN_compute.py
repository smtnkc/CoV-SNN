import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaModel
import torch
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import RobertaForMaskedLM, RobertaModel, RobertaConfig
import random

# Load the SentenceTransformer model
CHECKPOINT=sys.argv[1] # X for CoV-RoBERTa, 0 for CoV-SNN with best test accuracy, 4 for CoV-SNN with best zero-shot accuracy 
DISTANCE= "L2" # sys.argv[2] # L1 for Manhattan distance, L2 for Euclidean distance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reproducibility
seed = 500 if CHECKPOINT == "0" else 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if CHECKPOINT == "X":
    tokenizer = RobertaTokenizerFast.from_pretrained("trained_tokenizer/")
    MODEL_NAME = f"../mlm_checkpoints/CoV-RoBERTa_2048"
    masked_lm_model = RobertaForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    embedding_model = RobertaModel.from_pretrained(MODEL_NAME).to(device)
else:
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

def evaluate_grammaticality(predictions, masked_sentences, target_sentence):
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

def calculate_perplexity(sentence):
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = masked_lm_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()


# Load data
o = pd.read_csv("../data/unique_Omicron_2k.csv")["sequence"].tolist()[:2000]
e = pd.read_csv("../data/unique_Eris_2k.csv")["sequence"].tolist()[:2000]
n = pd.read_csv("../data/unique_New_2k.csv")["sequence"].tolist()[:2000]
gpt_10 = pd.read_csv("../data/unique_Gpt_1.0_2k.csv")["sequence"].tolist()[:2000]
gpt_11 = pd.read_csv("../data/unique_Gpt_1.1_2k.csv")["sequence"].tolist()[:2000]
gpt_12 = pd.read_csv("../data/unique_Gpt_1.2_2k.csv")["sequence"].tolist()[:2000]
gpt_15 = pd.read_csv("../data/unique_Gpt_1.5.csv")["sequence"].tolist()[:2000]


import time
# Calculate the average embedding of "o"
begin = time.time()
embeddings = []
for i, sentence in enumerate(o):
    emb = get_sentence_embedding(sentence)
    embeddings.append(emb)
    print(f"Embedding calculated for omicron sequence {i} with shape {emb.shape}")
average_embedding = np.mean(embeddings, axis=0)
end = time.time()
# hh:mm:ss
print(f"Average embedding calculated in {time.strftime('%H:%M:%S', time.gmtime(end - begin))}!")

# print("Average embedding calculated!")


def cscs(average_embedding, target_sentences, dataset_name):
    # Process each item in e
    semantic_change_scores = []
    grammaticality_scores = []
    perplexity_scores = []
    cscs_values_by_grammaticality = []
    cscs_values_by_perplexity = []

    begin = time.time()
    for i, target_sentence in enumerate(target_sentences):
        # Grammaticality score
        masked_sentences = mask_sentence(target_sentence)
        predictions = predict_masked_words(masked_sentences)
        grammaticality_score = evaluate_grammaticality(predictions, masked_sentences, target_sentence)
        grammaticality_scores.append(grammaticality_score)
        
        # Semantic change score
        semantic_change_score = calculate_semantic_change(average_embedding, target_sentence)
        semantic_change_scores.append(semantic_change_score)

        # Perplexity score
        perplexity_score = calculate_perplexity(target_sentence)
        perplexity_scores.append(perplexity_score)
        print(f"{dataset_name} sequence {i} ->  SC={semantic_change_score:.6f}  GR = {grammaticality_score:.6f}  PPL={perplexity_score:.6f}")
        
        # CSCS value
        cscs_gr = np.log10(grammaticality_score) - np.log10(semantic_change_score)
        cscs_ppl = np.log10(1/perplexity_score) - np.log10(semantic_change_score)
        cscs_values_by_grammaticality.append(cscs_gr)
        cscs_values_by_perplexity.append(cscs_ppl)
    end = time.time()
    print(f"{dataset_name} sequences processed in {time.strftime('%H:%M:%S', time.gmtime(end - begin))}!")

    # Optionally, save the results to a file
    semantic_change_scores = [round(score, 6) for score in semantic_change_scores]
    grammaticality_scores = [round(score, 6) for score in grammaticality_scores]
    perplexity_scores = [round(score, 6) for score in perplexity_scores]
    cscs_values_by_grammaticality = [round(score, 6) for score in cscs_values_by_grammaticality]
    cscs_values_by_perplexity = [round(score, 6) for score in cscs_values_by_perplexity]

    results_df = pd.DataFrame({"semantic_change": semantic_change_scores,
                            "grammaticality": grammaticality_scores,
                            "perplexity": perplexity_scores,
                            "log10(semantic_change)": np.log10(semantic_change_scores),
                            "log10(grammaticality)": np.log10(grammaticality_scores),
                            "log10(1/perplexity)": np.log10(1/np.array(perplexity_scores)),
                            "cscs_gr": cscs_values_by_grammaticality,
                            "cscs_ppl": cscs_values_by_perplexity,
                            "sentence": target_sentences
                            })
    out_file_name = f"../outputs/CSCS/CP{CHECKPOINT}_{DISTANCE}/cscs_values_{dataset_name}.csv"
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    results_df.to_csv(f"{out_file_name}", index=False)


#cscs(average_embedding, e, "eris")
#cscs(average_embedding, n, "new")
#cscs(average_embedding, gpt_11, "GPT_1.1")
#cscs(average_embedding, gpt_12, "GPT_1.2")
cscs(average_embedding, gpt_15, "GPT_1.5")
