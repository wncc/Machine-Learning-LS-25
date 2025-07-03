## What is Hugging Face?

Hugging Face is an open-source platform offering pre-trained models, datasets, and tools for NLP, computer vision, and audio tasks. Its **Transformers** library simplifies ML, and the **Model Hub** hosts community-contributed resources.

**Why Use It?**

- Beginner-friendly APIs
- Supports multiple domains
- Active community

---

## Installation and Setup

### Prerequisites

- Python 3.8+
- Basic Python knowledge
- Optional: GPU with CUDA

### Steps

1. **Virtual environment**:

   ```bash
   python -m venv hf_env
   source hf_env/bin/activate  # Windows: hf_env\Scripts\activate
   ```

2. **Install libraries**:

   ```bash
   pip install transformers datasets tokenizers torch
   ```

3. **Verify**:

   ```python
   import transformers
   print(transformers.__version__)
   ```

**Troubleshooting**:

- **Dependency issues**: Use a fresh virtual environment.
- **GPU errors**: Ensure CUDA compatibility.

---

## Core Libraries

### Transformers

Provides pre-trained models and pipelines for tasks like text classification.

- **Example**: Sentiment analysis

  ```python
  from transformers import pipeline
  classifier = pipeline("sentiment-analysis")
  print(classifier("I love Hugging Face!"))  # [{'label': 'POSITIVE', 'score': 0.999}]
  ```

### Datasets

Access and preprocess datasets efficiently.

- **Example**: Load IMDB dataset

  ```python
  from datasets import load_dataset
  dataset = load_dataset("imdb")
  print(dataset["train"][0])
  ```

### Tokenizers

Converts text to model inputs.

- **Example**: Tokenize text

  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  tokens = tokenizer("Hello, Hugging Face!", return_tensors="pt")
  print(tokens)
  ```

---

## Model Loading and Inference

### Basic Loading

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### Inference with Pipeline

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model=model_name)
print(classifier("Great tutorial!"))  # [{'label': 'POSITIVE', 'score': 0.999}]
```

### Cross-Domain Examples

- **NLP (Text Classification)**:

  ```python
  classifier = pipeline("text-classification")
  print(classifier("Fun movie!"))
  ```

- **Vision (Image Classification)**:

  ```python
  vision_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
  print(vision_classifier("https://example.com/cat.jpg"))
  ```

**Warning**: Match model to task (e.g., text model for text tasks).

---

## Fine-Tuning

Fine-tune a model on a custom dataset.

### Steps

1. **Load dataset**:

   ```python
   from datasets import load_dataset
   dataset = load_dataset("imdb")
   ```

2. **Load model**:

   ```python
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   model_name = "distilbert-base-uncased"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
   ```

3. **Preprocess**:

   ```python
   def tokenize(examples):
       return tokenizer(examples["text"], padding="max_length", truncation=True)
   tokenized_dataset = dataset.map(tokenize, batched=True)
   ```

4. **Train**:

   ```python
   from transformers import Trainer, TrainingArguments
   training_args = TrainingArguments(
       output_dir="./results",
       evaluation_strategy="epoch",
       learning_rate=2e-5,
       per_device_train_batch_size=8,
       num_train_epochs=3,
   )
   trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"])
   trainer.train()
   ```

5. **Save**:

   ```python
   model.save_pretrained("./fine_tuned_model")
   ```

**Best Practices**:

- Use small learning rates (e.g., 2e-5).
- Monitor validation loss to avoid overfitting.

---

## Model Sharing

Share models on the **Model Hub**.

1. **Log in**:

   ```python
   from huggingface_hub import login
   login()  # Use your Hugging Face token
   ```

2. **Push model**:

   ```python
   model.push_to_hub("my-model")
   tokenizer.push_to_hub("my-model")
   ```

---

## Use Cases

- **NLP**: Sentiment analysis

  ```python
  classifier = pipeline("sentiment-analysis")
  print(classifier(["Great book!", "Boring plot."]))
  ```

- **Vision**: Object detection

  ```python
  classifier = pipeline("image-classification")
  print(classifier("https://example.com/dog.jpg"))
  ```

---

## Helpful Resources 

- Hugging Face in 10 Minutes - https://www.youtube.com/watch?v=9gBC9R-msAk
- Official Hugging Face Course - https://huggingface.co/learn/nlp-course
- CodeBasics Transformers & HF playlist (very beginner friendly) - https://www.youtube.com/playlist?list=PLKnIA16_Rmvb7F5cnA6WhgZfz3BlvkxLx
- Crash Course for Hugging face (in 1 hour) - https://www.youtube.com/watch?v=b665B04CWkI 
- Hugging Face GitHub Repo - https://github.com/huggingface/awesome-huggingface

---

## Learning Path

1. **Beginner**: Install libraries, try pipelines, explore Model Hub.
2. **Intermediate**: Fine-tune models, experiment with vision/audio tasks.
3. **Advanced**: Build ML pipelines, contribute to the community.

This guide equips you to leverage Hugging Face for ML projects. Happy learning!
