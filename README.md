# CoV-SNN
Modeling Viral Evolution With Natural Language Processing

## Dataset Summary

| VOC      | Downloaded | High Quality  | Unique   | Sampled |
|----------|------------|-----------|----------|---------|
| Alpha    | 1,205,325  | 906,406   | 44,979   | 2,000   |
| Beta     | 44,458     | 12,793    | 2,714    | 2,000   |
| Delta    | 4,590,550  | 2,878,018 | 215,531  | 2,000   |
| Gamma    | 134,881    | 95,005    | 8,623    | 2,000   |
| Omicron  | 8,137,683  | 455,487   | 54,762   | 2,000   |
| **Total**| **15,589,601** | **4,347,709** | **326,609** | **10,000** |

## Variant Classification Performance

| Model| Embed Size | Batch Size | Parameters | Time | Eval F1 |
|------------------------|----------------|------------|------------|--------------------------|----------------|
| ProtBERT               | 2048           | 3          | 420M       | 2:57:54                  | 99.00          |
| CoVBERT                | 2048           | 12         | 45M        | 0:26:06                  | 95.85          |
| CoV-RoBERTa            | 2048           | 12         | 52M        | 0:23:52                  | 98.65          |
| CoV-RoBERTa            | 128            | 12         | 50M        | 0:00:47                  | 98.45          |

:information_source: All values are for 3 epochs of training.

## CoV-RoBERTa Runtime Stats on AUDP (48Gb GPU)

|Step | Task                                              | Duration   |
|-----|---------------------------------------------------|------------|
|0    | Tokenizer Training on 326,609 Unique Sequences    | 0:07:27    |
|1    | Sampling 10K Unique Sequences             | 0:00:15    |
|2    | Masking 10K Unique Sequences              | 0:02:18    |
|3    | Tokenization of 12M Masked Sequences      | 1:14:04    |
|4    | MLM Training on 12M Tokenized Sequences   | 8:48:40    |
|5    | Classification on 10K Unique Sequences    | 0:00:47    |

:information_source: The sequence lengths vary between 1235-1278. We perform single-amino-acid masking on 10K sequences. This results in 12,710,183 masked sequences.
