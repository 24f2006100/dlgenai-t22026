# Smart MCQ Solver

A deep learning project developed for the **Smart MCQ Solver Challenge**. The objective of the challenge is to rank five answer choices for each multiple-choice question such that the correct answer appears within the top three predictions.

The project implements and compares four approaches: **TF-IDF with Cosine Similarity, BiLSTM, BERT fine-tuning, and LoRA-based BERT fine-tuning**. The final BERT model was selected for the competition submission and was also deployed as an interactive **Streamlit** application.

## Problem Statement

The Smart MCQ Solver Challenge consists of multiple-choice questions where each question is accompanied by five possible answer options labeled **A, B, C, D, and E**. Instead of simply predicting one answer, the task requires the model to rank all five options according to their likelihood of being correct.

The primary evaluation metric is **Mean Average Precision at 3 (MAP@3)**. For each question, a score of `1` is awarded if the correct answer is ranked first, `1/2` if it is ranked second, `1/3` if it is ranked third, and `0` if it does not appear in the top three predictions.

## Dataset

The training dataset contains **2,000 multiple-choice questions**. Each sample contains an `id`, a `prompt`, five answer options (`A`, `B`, `C`, `D`, and `E`), and the correct `answer`.

The data was explored using basic exploratory data analysis, including dataset shape, sample records, missing values, duplicate entries, unique values, answer-label distribution, prompt lengths, and memory usage.

The training data was transformed into question-option pairs so that every question generates five individual examples. The correct option receives a label of `1`, while the other four options receive a label of `0`.

## Text Preprocessing

A custom `clean_text()` function was used to normalize the text. It converts values to strings, removes unnecessary whitespace using regular expressions, and removes leading and trailing spaces.

The preprocessing was applied consistently to the question prompt and all five answer options in both the training and test datasets.

## Models

### TF-IDF + Cosine Similarity

TF-IDF with Cosine Similarity was implemented as the traditional NLP baseline.

The text representation used English stop-word removal, unigrams and bigrams through `ngram_range=(1,2)`, and `min_df=2`. Question and option texts were converted into TF-IDF vectors, and cosine similarity was used to calculate how closely an option matched the question.

The five options were ranked according to their similarity scores and the top three were selected for MAP@3 evaluation.

### BiLSTM

A **Bidirectional Long Short-Term Memory (BiLSTM)** model was implemented as a deep learning approach.

A BiLSTM processes a sequence in both forward and backward directions, allowing the model to capture information from both sides of the sequence. The model was trained on the question-option classification task and its prediction scores were subsequently used to rank the five answer options.

### BERT

A pretrained **`bert-base-uncased`** model was fine-tuned for binary sequence classification.

Each question-option pair was treated as an individual classification example. The model predicts whether the given option is the correct answer, where class `0` represents an incorrect option and class `1` represents the correct option.

The pretrained WordPiece tokenizer associated with `bert-base-uncased` was used. Question and option pairs were tokenized with a maximum sequence length of `256`, `padding="max_length"`, and truncation enabled.

The BERT model was trained using a learning rate of `2e-5`, batch size of `16`, `3` training epochs, and weight decay of `0.01`.

During inference, the probability of class `1` was used as the score for each answer option. The five options were ranked according to their scores and the top three were selected.

BERT produced the strongest overall results among the transformer approaches and was selected for generating the final competition submission.

### LoRA

**LoRA (Low-Rank Adaptation)** was used to fine-tune BERT using a parameter-efficient fine-tuning approach.

Instead of updating the complete pretrained BERT model, LoRA keeps most of the original parameters frozen and introduces trainable low-rank matrices into selected attention layers. In this project, LoRA was applied to the `query`, `key`, and `value` components of the BERT attention mechanism.

The LoRA configuration used `r=16`, `lora_alpha=32`, and `lora_dropout=0.1`. The classifier layer was also kept trainable.

LoRA provided a parameter-efficient alternative to full BERT fine-tuning by significantly reducing the number of parameters that needed to be updated.

## Model Comparison

TF-IDF provided a simple traditional baseline based on lexical similarity. BiLSTM improved upon this by learning sequential representations from the text. LoRA provided an efficient way to adapt BERT while training fewer parameters. BERT was selected as the final model based on the overall validation experiments.

## Evaluation

The primary evaluation metric was **MAP@3**, since the competition evaluates the ranking of the five answer options.

Additional metrics used during experimentation were **Accuracy, F1-score, and Validation Loss**.

Accuracy measures the percentage of correctly classified question-option pairs. F1-score combines precision and recall for the binary classification task. Validation loss measures the classification error on unseen validation data.

MAP@3 evaluates the position of the correct answer within the top three predictions. A higher score indicates that the model is more successful at placing the correct answer near the top of its ranking.

## Experiment Tracking

**Weights & Biases (W&B)** was used to track and visualize the model experiments.

The training and evaluation runs recorded metrics such as training loss, validation loss, accuracy, F1-score, learning rate, gradient norm, training runtime, evaluation runtime, and training and evaluation steps.

The W&B dashboards were used to monitor model training and compare the different approaches.

## Inference

For inference, each test question is converted into five question-option pairs. The trained BERT tokenizer processes each question-option pair using the same tokenization configuration used during training.

The trained model produces classification logits for each pair. Softmax is then applied to obtain class probabilities, and the probability of class `1` is used as the score for the corresponding answer option.

The five options are sorted according to their scores, and the three highest-scoring options are selected as the final prediction.