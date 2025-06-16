# SST-2 Fine-tuned DistilBERT Toxic Text Classifier

This implementation uses DistilBERT model pre-trained on SST-2 dataset (`distilbert-base-uncased-finetuned-sst-2-english`) and further fine-tuned on the Jigsaw Toxic Comment Classification dataset.

## Features

- Leverages sentiment understanding from SST-2
- Faster training convergence
- Lower computational requirements
- Efficient resource usage

## Directory Structure

```
sst_model/
├── toxic_classifier_sst.ipynb  # Training and evaluation notebook
├── inference.py               # Standalone inference script
└── requirements.txt          # Project dependencies
```

## Training Configuration

```python
training_args = TrainingArguments(
    num_train_epochs=3,          # Fewer epochs needed
    learning_rate=2e-5,          # Standard learning rate
    weight_decay=0.01,
    warmup_steps=200,            # Shorter warmup period
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    fp16=True
)
```

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Training (optional):

```bash
jupyter notebook toxic_classifier_sst.ipynb
```

3. Inference:

```python
from inference import TextGuardrailSST

# Initialize with base SST-2 model
guardrail = TextGuardrailSST()

# Or with your fine-tuned model
# Download from the google drive - https://drive.google.com/file/d/1ACeFGelgA5T8kMlFf4USnVzdbTni-Jw_/view?usp=drive_link
# guardrail = TextGuardrailSST("path/to/model")

# Single text classification
result = guardrail.check_text("Your text here")
print(f"Classification: {result['label']}")
print(f"Confidence: {result['confidence']}")

# Batch classification
texts = ["Text 1", "Text 2", "Text 3"]
results = guardrail.batch_check(texts)
```

## Training Process

1. Data Preparation

   - Load Jigsaw dataset
   - Convert to binary classification
   - Split into train/val/test

2. Model Training

   - Load SST-2 pre-trained model
   - Fine-tune on toxic data
   - Track CO2 emissions

3. Evaluation
   - Test set evaluation
   - Performance metrics
   - Example inferences

## Performance Notes

- Faster training convergence
- Lower computational cost
- Good performance on sentiment-related toxicity
- Efficient resource utilization

## Hardware Requirements

- GPU with 4GB+ memory sufficient
- Training time: ~2-3 hours on Colab GPU
- Memory usage: ~3GB during training

## Inference Examples

```python
test_texts = [
    "Hello, how are you today?",
    "You're so stupid and worthless",
    "The weather is nice today",
    "I hate everyone here"
]

for text in test_texts:
    result = guardrail.check_text(text)
    print(f"\nText: {result['text']}")
    print(f"Classification: {result['label']}")
    print(f"Confidence: {result['confidence']}")
```

## Environmental Impact

The model tracks CO2 emissions during training using codecarbon:

```python
tracker = EmissionsTracker(project_name="toxic_classifier_sst")
tracker.start()
trainer.train()
emissions = tracker.stop()
print(f"Total CO2 emissions: {emissions} kg")
```
