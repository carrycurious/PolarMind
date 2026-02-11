# PolarMind at SemEval-2026 Task 9: Leveraging LaBSE with Progressive Curriculum Learning for Multicultural Polarization

**PolarMind** is an advanced NLP system developed for **SemEval-2026 Task 9**, focusing on the detection of online polarization in multicultural and multilingual contexts. Our approach integrates the high-performance cross-lingual alignment of **LaBSE** with a custom **Progressive Curriculum Learning** strategy and a sophisticated **Multi-Layer Hybrid Pooling** architecture.

---

## 🛠 Methodology

### 1. Data Difficulty Scoring (via XLM-RoBERTa)
To implement curriculum learning, we first categorize the dataset by difficulty using **XLM-RoBERTa-base**. 
* **Scoring**: The model is evaluated on the training set to calculate the **Cross-Entropy Loss** for every sample.
* **Splitting**: We divide the dataset into three equal-sized buckets based on these loss values:
    * **Easy**: Samples with the lowest $1/3$ of loss (high model confidence).
    * **Medium**: Samples in the middle $1/3$ tier of loss.
    * **Hard**: Samples with the highest $1/3$ of loss, typically containing sarcasm, code-switching, or subtle polarization.

### 2. Progressive Curriculum Learning
We train the final LaBSE-based classifier in three distinct stages, resetting the learning rate scheduler at each transition to allow the model to adapt to increasing complexity:
* **Phase 1 (Easy)**: Establishes a foundational multilingual baseline.
* **Phase 2 (Medium)**: Refines the decision boundary with more nuanced examples.
* **Phase 3 (Hard)**: Performs "model sharpening" on the most difficult edge cases to improve F1-score.



### 3. LaBSE Multi-Layer Hybrid Architecture
Our model architecture leverages the structural depth of LaBSE to capture both syntactic and semantic features:

* **Weighted Layer Aggregation**: We extract hidden states from the **last 4 layers** of the encoder.
* **Learnable Parameters**: These layers are weighted using learnable parameters ($w_1, w_2, w_3, w_4$), initialized with small values of ($w_1, w_2, w_3$) to let the model learn the optimal layer importance.
* **Hybrid Multi-Layer Pooling**: For each layer, we compute a combined representation:
  
$$\text{Layer Output} = \text{Mean Pooling} + \lambda \times (\text{Attention Pooling})$$

  This hybrid approach ensures the model captures both the general sentence context and specific high-impact tokens.

---

##  Getting Started

### Prerequisites
Ensure you have a local environment with Python 3.8+ installed. It is recommended to use a virtual environment.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/PolarMind.git](https://github.com/your-username/PolarMind.git)
   cd PolarMind
