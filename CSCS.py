import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaModel
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer
from transformers import RobertaForMaskedLM, RobertaModel, RobertaConfig

# Load the SentenceTransformer model
CHECKPOINT="X"
DISTANCE="L2" # L1 for Manhattan distance, L2 for Euclidean distance

# if CHECKPOINT == "X":
#     tokenizer = RobertaTokenizerFast.from_pretrained("tok/")
#     MODEL_NAME = f"./mlm_checkpoints/CoV-RoBERTa_2048"
#     masked_lm_model = RobertaForMaskedLM.from_pretrained(MODEL_NAME)
#     embedding_model = RobertaModel.from_pretrained(MODEL_NAME)
# else:
#     sbert_model = SentenceTransformer("outputs/ContrastiveLoss_omicron_vs_delta_Pmax_R0.2" + \
#                                     f"_D0.1_E10_LR_0.001_B32_M2.0/checkpoints/checkpoint-{CHECKPOINT}")

#     transformer_model = sbert_model[0].auto_model  # Extract the core transformer model

#     # Get the configuration of the SentenceTransformer model
#     sbert_config = transformer_model.config

#     # Create a custom RobertaConfig with the same parameters
#     custom_config = RobertaConfig(
#         vocab_size=sbert_config.vocab_size,
#         hidden_size=sbert_config.hidden_size,
#         num_hidden_layers=sbert_config.num_hidden_layers,
#         num_attention_heads=sbert_config.num_attention_heads,
#         intermediate_size=sbert_config.intermediate_size,
#         max_position_embeddings=sbert_config.max_position_embeddings,
#         type_vocab_size=sbert_config.type_vocab_size,
#         initializer_range=sbert_config.initializer_range,
#         layer_norm_eps=sbert_config.layer_norm_eps,
#         hidden_dropout_prob=sbert_config.hidden_dropout_prob,
#         attention_probs_dropout_prob=sbert_config.attention_probs_dropout_prob,
#     )

#     # Initialize RobertaForMaskedLM and RobertaModel with the custom configuration
#     masked_lm_model = RobertaForMaskedLM(custom_config)
#     embedding_model = RobertaModel(custom_config)
#     tokenizer = sbert_model[0].tokenizer

#     # Copy the weights from transformer_model to masked_lm_model and embedding_model
#     transformer_model_state_dict = transformer_model.state_dict()
#     masked_lm_model.roberta.load_state_dict(transformer_model_state_dict, strict=False)
#     embedding_model.load_state_dict(transformer_model_state_dict, strict=False)

# print(f"Weights successfully copied from checkpoint {CHECKPOINT}!")

# # Set the models in evaluation mode
# masked_lm_model.eval()
# embedding_model.eval()

# def mask_sentence(sentence):
#     tokens = tokenizer.tokenize(sentence)
#     masked_sentences = []
#     for i in range(len(tokens)):
#         masked_tokens = tokens.copy()
#         masked_tokens[i] = '<mask>'
#         masked_sentence = ' '.join(masked_tokens)
#         masked_sentences.append(masked_sentence)
#     return masked_sentences

# def predict_masked_words(masked_sentences):
#     predictions = []
#     for masked_sentence in masked_sentences:
#         inputs = tokenizer.encode(masked_sentence, return_tensors='pt')
#         with torch.no_grad():
#             outputs = masked_lm_model(inputs)
#             predictions.append(outputs.logits)
#     return predictions

# def evaluate_grammaticality(predictions, masked_sentences, target_sentence):
#     total_score = 0
#     tokens = tokenizer.tokenize(target_sentence)
#     for i, logits in enumerate(predictions):
#         masked_index = masked_sentences[i].split().index('<mask>')
#         softmax = torch.nn.functional.softmax(logits[0, masked_index], dim=-1)
#         original_token_id = tokenizer.convert_tokens_to_ids(tokens[i])
#         score = softmax[original_token_id].item()
#         total_score += score
#     average_score = total_score / len(masked_sentences)
#     return average_score

# def get_sentence_embedding(sentence):
#     inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
#     with torch.no_grad():
#         outputs = embedding_model(**inputs)
#     return outputs.last_hidden_state.squeeze().numpy()

# def calculate_semantic_change(average_embedding, target_sentence):
#     target_embedding = get_sentence_embedding(target_sentence)
#     if DISTANCE == "L2":
#         semantic_change = np.linalg.norm(average_embedding - target_embedding) # Euclidean distance
#     elif DISTANCE == "L1":
#         semantic_change = np.linalg.norm(average_embedding - target_embedding, ord=1) # Manhattan distance
#     return semantic_change

# def calculate_perplexity(sentence):
#     inputs = tokenizer(sentence, return_tensors='pt')
#     with torch.no_grad():
#         outputs = masked_lm_model(**inputs, labels=inputs["input_ids"])
#     loss = outputs.loss
#     perplexity = torch.exp(loss)
#     return perplexity.item()


# # Load data
# o = pd.read_csv("data/unique_Omicron_2k.csv")["sequence"].tolist()[:2000]
# e = pd.read_csv("data/unique_Eris_2k.csv")["sequence"].tolist()[:2000]
# n = pd.read_csv("data/unique_New_2k.csv")["sequence"].tolist()[:2000]

# # Calculate the average embedding of "o"
# embeddings = []
# for i, sentence in enumerate(o):
#     emb = get_sentence_embedding(sentence)
#     embeddings.append(emb)
#     print(f"Embedding calculated for omicron sequence {i} with shape {emb.shape}")
# average_embedding = np.mean(embeddings, axis=0)
# #print("Average embedding calculated!")


# def cscs(average_embedding, target_sentences, dataset_name):
#     # Process each item in e
#     semantic_change_scores = []
#     grammaticality_scores = []
#     perplexity_scores = []
#     cscs_values_by_grammaticality = []
#     cscs_values_by_perplexity = []
#     for i, target_sentence in enumerate(target_sentences):
#         # Grammaticality score
#         masked_sentences = mask_sentence(target_sentence)
#         predictions = predict_masked_words(masked_sentences)
#         grammaticality_score = evaluate_grammaticality(predictions, masked_sentences, target_sentence)
#         grammaticality_scores.append(grammaticality_score)
        
#         # Semantic change score
#         semantic_change_score = calculate_semantic_change(average_embedding, target_sentence)
#         semantic_change_scores.append(semantic_change_score)

#         # Perplexity score
#         perplexity_score = calculate_perplexity(target_sentence)
#         perplexity_scores.append(perplexity_score)
#         print(f"{dataset_name} sequence {i} ->  SC={semantic_change_score:.6f}  GR = {grammaticality_score:.6f}  PPL={perplexity_score:.6f}")
        
#         # CSCS value
#         cscs_gr = np.log10(grammaticality_score) - np.log10(semantic_change_score)
#         cscs_ppl = np.log10(1/perplexity_score) - np.log10(semantic_change_score)
#         cscs_values_by_grammaticality.append(cscs_gr)
#         cscs_values_by_perplexity.append(cscs_ppl)

#     # Optionally, save the results to a file
#     semantic_change_scores = [round(score, 6) for score in semantic_change_scores]
#     grammaticality_scores = [round(score, 6) for score in grammaticality_scores]
#     perplexity_scores = [round(score, 6) for score in perplexity_scores]
#     cscs_values_by_grammaticality = [round(score, 6) for score in cscs_values_by_grammaticality]
#     cscs_values_by_perplexity = [round(score, 6) for score in cscs_values_by_perplexity]

#     results_df = pd.DataFrame({"semantic_change": semantic_change_scores,
#                             "grammaticality": grammaticality_scores,
#                             "perplexity": perplexity_scores,
#                             "log10(semantic_change)": np.log10(semantic_change_scores),
#                             "log10(grammaticality)": np.log10(grammaticality_scores),
#                             "log10(1/perplexity)": np.log10(1/np.array(perplexity_scores)),
#                             "cscs_gr": cscs_values_by_grammaticality,
#                             "cscs_ppl": cscs_values_by_perplexity,
#                             "sentence": target_sentences
#                             })
#     out_file_name = f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_values_{dataset_name}.csv"
#     os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
#     results_df.to_csv(f"{out_file_name}", index=False)
#     return results_df

# df_eris_cscs = cscs(average_embedding, e, "eris")
# df_new_cscs = cscs(average_embedding, n, "new")


######################################################################

import matplotlib.pyplot as plt

# Reading the data
df_eris_cscs = pd.read_csv(f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_values_eris.csv")
df_new_cscs = pd.read_csv(f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_values_new.csv")

# Creating subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plotting Semantic Change vs Grammaticality
axes[0].scatter(df_eris_cscs["log10(grammaticality)"], df_eris_cscs["log10(semantic_change)"], color='blue', alpha=0.5, s=30, label='Eris')
axes[0].scatter(df_new_cscs["log10(grammaticality)"], df_new_cscs["log10(semantic_change)"], color='red', alpha=0.5, s=30, label='New Omicron')
axes[0].set_xlabel("Sequence Probability (log$_{10}$)", fontsize=16)
axes[0].set_ylabel("Semantic change (log$_{10}$)", fontsize=16)
axes[0].legend()
axes[0].set_title("Semantic Change vs Sequence Probability", fontsize=16)

# Plotting Semantic Change vs log10(1/Perplexity)
axes[1].scatter(df_eris_cscs["log10(1/perplexity)"], df_eris_cscs["log10(semantic_change)"], color='blue', alpha=0.5, s=30, label='Eris')
axes[1].scatter(df_new_cscs["log10(1/perplexity)"], df_new_cscs["log10(semantic_change)"], color='red', alpha=0.5, s=30, label='New Omicron')
axes[1].set_xlabel("Inverse Perplexity (log$_{10}$)", fontsize=16)
axes[1].set_ylabel("Semantic change (log$_{10}$)", fontsize=16)
axes[1].legend()
axes[1].set_title("Semantic Change vs Inverse Perplexity", fontsize=16)

# Font size settings
font_size = 16
plt.rc('font', size=font_size)
plt.rc('axes', titlesize=font_size)
plt.rc('axes', labelsize=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('figure', titlesize=font_size)

# Overall title
if CHECKPOINT == "X":
    MODEL = "CoV-RoBERTa without Contrastive Learning"
elif CHECKPOINT == "4":
    MODEL = "CoV-SNN Transformer with Best Zero-shot Acc"
else:
    MODEL = "CoV-SNN Transformer with Best Test Acc"
#plt.suptitle(f"Semantic Change Analysis ({MODEL})", fontsize=font_size)

# Layout and saving the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_file_name = f"outputs/cscs_CP{CHECKPOINT}_{DISTANCE}/cscs_CP{CHECKPOINT}_{DISTANCE}.pdf"
plt.savefig(fig_file_name)
plt.show()