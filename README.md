# Toxic Text Classifier using DistilBERT SST-2

A robust text classifier for identifying toxic content using DistilBERT model fine-tuned on SST-2 dataset and further trained on the Jigsaw Toxic Comment Classification Challenge dataset.

## Project Overview

This project implements a binary classifier that can identify toxic content in text. It uses DistilBERT (a distilled version of BERT) that was initially fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset and then further fine-tuned on the Jigsaw Toxic Comment Classification dataset for toxic content detection.

### Key Features

- Binary classification (toxic vs safe)
- Fast inference with DistilBERT
- High accuracy and balanced precision/recall
- Optimized for community hardware constraints

## Technical Approach

### Model Architecture

- Base Model: DistilBERT ([distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english))
- Dataset: [Jigsaw Toxic Comment Classification Challenge](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge)
- Final classification layer with 2 outputs (safe/unsafe)
- Maximum sequence length: 128 tokens

### Training Configuration

- Learning rate: 2e-5
- Batch size: 16 (32 effective with gradient accumulation)
- Training epochs: 3
- Weight decay: 0.01
- Optimizer: AdamW with weight decay

### Optimizations

- Mixed precision training (FP16)
- Gradient accumulation steps: 2
- Best model checkpoint saving based on F1 score

## Performance Metrics

### Classification Metrics

- Accuracy: ~97%
- F1 Score: ~85%
- Precision: ~85%
- Recall: ~85%

### Environmental Impact

- Total energy consumption: 0.060490 kWh
- CO2 emissions: 0.021124 kg
- Estimated yearly CO2 if run continuously: 362.30 kg

## Usage Guide

### Installation

1. Download the model from [Google Drive](https://drive.google.com/file/d/1ACeFGelgA5T8kMlFf4USnVzdbTni-Jw_/view?usp=drive_link)

2. Install required packages:

```bash
pip install transformers torch
```

### Example Usage

```python
from inference import TextGuardrailSST

# Initialize the model
model = TextGuardrailSST(model_path="path_to_downloaded_model")

# Single text classification
text = "Hello, how are you today?"
result = model.check_text(text)
print(result)

# Batch processing
texts = [
    "Hello, how are you today?",
    "I really enjoyed the movie!",
    "You're so stupid and worthless",
    "Let's work together to solve this problem"
]
results = model.batch_check(texts)
for result in results:
    print(result)
```

## Example Results

Here are some example classifications demonstrating the model's performance:

```python
# Safe Examples
"Hello, how are you today?"
# Result: {'text': 'Hello, how are you today?', 'label': 'safe', 'confidence': '98.45%'}

"I really enjoyed the movie!"
# Result: {'text': 'I really enjoyed the movie!', 'label': 'safe', 'confidence': '97.23%'}

"Let's work together to solve this problem"
# Result: {'text': "Let's work together to solve this problem", 'label': 'safe', 'confidence': '96.78%'}

# Unsafe Examples
"You're so stupid and worthless"
# Result: {'text': "You're so stupid and worthless", 'label': 'unsafe', 'confidence': '92.34%'}

"I hate you"
# Result: {'text': 'I hate you', 'label': 'unsafe', 'confidence': '88.56%'}
```

## Trade-offs and Considerations

1. Model Size vs Speed

   - DistilBERT chosen for balance between accuracy and inference speed
   - 40% smaller than BERT while retaining 97% of performance

2. Sequence Length

   - Limited to 128 tokens for efficiency
   - Suitable for most social media and comment content
   - Longer texts will be truncated

3. Binary vs Multi-label

   - Binary classification chosen for simplicity and reliability
   - Trade-off between granular toxicity categories and robust performance

4. Hardware Constraints
   - FP16 mixed precision for memory efficiency
   - Batch processing available for throughput optimization
   - Can run on CPU or GPU

## License

This project uses the pre-trained DistilBERT model from Hugging Face, which is licensed under the Apache License 2.0.
