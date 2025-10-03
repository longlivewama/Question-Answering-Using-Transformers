# Question Answering with Transformers

This project demonstrates how to build a **Question Answering (QA) system** using **transformer-based models** such as **BERT, RoBERTa, and ALBERT**. The system is designed to understand a passage of text and answer questions based on the information contained in that passage. The project leverages the **SQuAD v1.1 dataset**, one of the most widely used benchmarks for evaluating QA systems, and uses the **Hugging Face Transformers** and **Datasets** libraries for model handling and data management.

---

## Features

1. **Data Loading and Preprocessing**

   * Automatically downloads the **SQuAD v1.1 dataset** using the Hugging Face Datasets library.
   * Performs **tokenization** using the appropriate tokenizer for each model.
   * Handles **context-question pair formatting** for input to transformer models.
   * Supports padding, truncation, and mapping of answer positions to token indices.

2. **Model Comparison**

   * Implements three popular transformer models for QA:

     * **BERT (Bidirectional Encoder Representations from Transformers)**
     * **RoBERTa (Robustly Optimized BERT Pretraining Approach)**
     * **ALBERT (A Lite BERT for Self-supervised Learning of Language Representations)**
   * Easily switch between models by changing the model name in the configuration.
   * Demonstrates differences in **accuracy, speed, and resource usage** between models.

3. **Training and Fine-tuning**

   * Supports **fine-tuning pre-trained models** on the SQuAD v1.1 dataset.
   * Handles batching, gradient accumulation, and training loop using PyTorch.
   * Optionally allows **evaluation during training** to monitor metrics like Exact Match (EM) and F1.

4. **Evaluation**

   * Evaluates model performance using standard QA metrics:

     * **Exact Match (EM):** Measures the percentage of predictions that exactly match the ground truth answers.
     * **F1 Score:** Measures the overlap between predicted and ground truth answers at the token level.
   * Provides a **detailed performance report** to compare different models.

5. **Interactive QA Interface**

   * Allows users to input any passage and ask questions interactively.
   * Uses the trained model to generate answers in real time.
   * Can be used for experimentation or demonstration purposes.

---

## Requirements

* Python 3.7+
* PyTorch
* Transformers (Hugging Face)
* Datasets (Hugging Face)
* Evaluate (Hugging Face)
* tqdm (for progress bars)

Install all dependencies with:

```bash
pip install torch transformers datasets evaluate tqdm
```

> **Optional:** For GPU acceleration, make sure to install the appropriate PyTorch version for your CUDA setup.

---

## Usage

1. Open `Question Answering with Transformers.ipynb` in **Jupyter Notebook** or **VS Code**.
2. Run each cell step by step:

   * Load and preprocess the dataset.
   * Load the pre-trained model and tokenizer.
   * Fine-tune the model (optional).
   * Evaluate the model on the validation set.
3. Use the **interactive interface** at the end to ask questions on any input passage.

---

## Tips

* For large datasets or models, using a **GPU** is highly recommended to speed up training and inference.
* To try other transformer models, simply replace the model name in the code (e.g., `bert-base-uncased`) with another model from Hugging Face’s **Model Hub**.
* Save fine-tuned models using `model.save_pretrained()` for future use.

---

## Citation

If you use this code in your research or projects, please cite:

* **Hugging Face Transformers library**: [https://huggingface.co/transformers](https://huggingface.co/transformers)
* **SQuAD v1.1 dataset**: [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)

---
