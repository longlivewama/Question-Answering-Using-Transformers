# Question Answering with Transformers

This project demonstrates how to build a **Question Answering (QA) system** using **transformer-based models** such as **BERT, RoBERTa, and ALBERT**. The system is designed to understand a passage of text and answer questions based on the information contained within that passage. It leverages the **SQuAD v1.1 dataset**, one of the most widely used benchmarks for QA tasks, and utilizes the **Hugging Face Transformers** and **Datasets** libraries for model management and data processing.

---

## 🚀 Features

1. **Data Loading and Preprocessing**

   * Automatically downloads the **SQuAD v1.1 dataset** using Hugging Face Datasets.
   * Performs **tokenization** with the appropriate tokenizer for each model.
   * Formats **context–question pairs** for transformer input.
   * Supports padding, truncation, and mapping of answer positions to token indices.

2. **Model Comparison**

   * Implements three widely used transformer models for QA:

     * **BERT (Bidirectional Encoder Representations from Transformers)**
     * **RoBERTa (Robustly Optimized BERT Pretraining Approach)**
     * **ALBERT (A Lite BERT for Self-supervised Learning of Language Representations)**
   * Easily switch between models by changing the configuration.
   * Highlights differences in **accuracy, inference speed, and resource consumption**.

3. **Training and Fine-Tuning**

   * Supports **fine-tuning pre-trained models** on the SQuAD v1.1 dataset.
   * Includes batching, gradient accumulation, and PyTorch training loops.
   * Optionally evaluates during training with **Exact Match (EM)** and **F1 score**.

4. **Evaluation**

   * Uses standard QA metrics:

     * **Exact Match (EM):** Measures the percentage of predictions that exactly match the ground truth.
     * **F1 Score:** Measures token-level overlap between predictions and ground truth.
   * Provides a **detailed performance report** for model comparison.

5. **Interactive QA Interface**

   * Input any passage and interactively ask questions.
   * The trained model provides answers in real time.
   * Useful for experimentation and live demonstrations.

6. **Export Results to CSV**

   * A dedicated function allows saving the **context, question, and predicted answer** into a **CSV file**.
   * Supports appending multiple QA results for easy reporting and analysis.

---

## 💡 Use Cases

* Build and benchmark **QA models** for research or production.
* Serve as an **educational tool** to learn how transformer-based QA works.
* Provide answers on **any custom text/topic** by supplying your own passage and question.
* Log results by exporting **context–question–answer triplets** to CSV for documentation or downstream processing.

---

## 🛠 Requirements

* Python 3.7+
* PyTorch
* Transformers (Hugging Face)
* Datasets (Hugging Face)
* Evaluate (Hugging Face)
* tqdm (progress bars)

Install dependencies:

```bash
pip install torch transformers datasets evaluate tqdm
```

> **Tip:** For GPU acceleration, install the PyTorch version compatible with your CUDA setup.

---

## 📖 Usage

1. Open `Question Answering with Transformers.ipynb` in **Jupyter Notebook** or **VS Code**.
2. Run cells step by step:

   * Load and preprocess the dataset.
   * Load the pre-trained model and tokenizer.
   * Fine-tune the model (optional).
   * Evaluate the model on the validation set.
3. Use the **interactive QA interface** to:

   * Provide any passage and ask a question.
   * Export the **context, question, and answer** to CSV.

---

## 📝 Example: Saving QA Results to CSV

```python
import pandas as pd

def save_to_csv(context, question, answer, filename="qa_results.csv"):
    # Create a DataFrame for the result
    df = pd.DataFrame([{
        "Context": context,
        "Question": question,
        "Answer": answer
    }])
    
    # Append if the file exists, otherwise create new
    try:
        existing = pd.read_csv(filename)
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    df.to_csv(filename, index=False)
    print(f"Saved QA result to {filename}")

# Example usage
context = "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."
question = "Who developed the theory of relativity?"
answer = "Albert Einstein"

save_to_csv(context, question, answer)
```

Generated **qa_results.csv** file:

| Context                                                                                         | Question                                | Answer          |
| ----------------------------------------------------------------------------------------------- | --------------------------------------- | --------------- |
| Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. | Who developed the theory of relativity? | Albert Einstein |

---

## 📌 Tips

* Use a **GPU** for large datasets or heavy models to significantly reduce training and inference time.
* Try other transformer models by simply changing the model name (e.g., `bert-base-uncased`) with another from the Hugging Face **Model Hub**.
* Save fine-tuned models with `model.save_pretrained()` for reuse.

---

## 📚 Citation

If you use this project, please cite:

* **Hugging Face Transformers library**: [https://huggingface.co/transformers](https://huggingface.co/transformers)
* **SQuAD v1.1 dataset**: [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)

---
