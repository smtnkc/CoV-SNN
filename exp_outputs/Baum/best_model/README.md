---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity

---

# {MODEL_NAME}

This is a [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 153 dimensional dense vector space and can be used for tasks like clustering or semantic search.

<!--- Describe your model here -->

## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('{MODEL_NAME}')
embeddings = model.encode(sentences)
print(embeddings)
```



## Evaluation Results

<!--- Describe how your model was evaluated -->

For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net?model_name={MODEL_NAME})


## Training
The model was trained with the parameters:

**DataLoader**:

`torch.utils.data.dataloader.DataLoader` of length 128 with parameters:
```
{'batch_size': 32, 'sampler': 'torch.utils.data.sampler.RandomSampler', 'batch_sampler': 'torch.utils.data.sampler.BatchSampler'}
```

**Loss**:

`sentence_transformers.losses.ContrastiveLoss.ContrastiveLoss` with parameters:
  ```
  {'distance_metric': 'SiameseDistanceMetric.EUCLIDEAN', 'margin': 0.2, 'size_average': True}
  ```

Parameters of the fit()-Method:
```
{
    "epochs": 10,
    "evaluation_steps": 0,
    "evaluator": "sentence_transformers.evaluation.BinaryClassificationEvaluator.BinaryClassificationEvaluator",
    "loss_name": "ContrastiveLoss",
    "max_grad_norm": 1,
    "optimizer_class": "<class 'torch.optim.adamw.AdamW'>",
    "optimizer_params": {
        "lr": 0.0001
    },
    "scheduler": "WarmupLinear",
    "steps_per_epoch": null,
    "tester": "sentence_transformers.evaluation.BinaryClassificationEvaluator.BinaryClassificationEvaluator",
    "warmup_steps": 10000,
    "weight_decay": 0.0001,
    "zero_shot_tester": "sentence_transformers.evaluation.BinaryClassificationEvaluator.BinaryClassificationEvaluator"
}
```


## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 1280, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': True, 'pooling_mode_global_max': False, 'pooling_mode_global_avg': False, 'pooling_mode_attention': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False})
  (2): Dense({'in_features': 768, 'out_features': 153, 'bias': True, 'activation_function': 'torch.nn.modules.activation.ReLU'})
  (3): Dropout(
    (dropout_layer): Dropout(p=0.5, inplace=False)
  )
)
```

## Citing & Authors

<!--- Describe where people can find more information -->