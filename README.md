# PolarMind at SemEval-2026 Task 9: Leveraging LaBSE with Progressive Curriculum Learning for Multicultural Polarization

**PolarMind** is an advanced NLP system developed for **SemEval-2026 Task 9**, focusing on the detection of online polarization in multicultural and multilingual contexts. Our approach integrates the robust cross-lingual alignment of **LaBSE** with a custom **Progressive Curriculum Learning** strategy and a sophisticated **Multi-Layer Hybrid Pooling** architecture.

---
**Please note:** code for the exploration models have not been uploaded in this repo yet and it will be uploaded once the paper is submitted. Stay tuned!!
## 🛠 Methodology

### I. Primary Model: LaBSE with Curriculum Learning

#### 1. Data Difficulty Scoring (via XLM-RoBERTa)
To implement curriculum learning, we first categorize the dataset by difficulty using **XLM-RoBERTa-base**.
* **Scoring**: The model is evaluated on the training set to calculate the **Cross-Entropy Loss** per sample.
* **Splitting**: We divide the dataset into three equal-sized buckets based on these loss values:
    * **Easy**: Samples with the lowest $1/3$ of loss (high model confidence).
    * **Medium**: Samples in the middle $1/3$ tier of loss.
    * **Hard**: Samples with the highest $1/3$ of loss, typically containing sarcasm, code-switching, or subtle polarization.

#### 2. Progressive Curriculum Learning
We train the final LaBSE-based classifier in three distinct stages, resetting the learning rate scheduler at each transition to allow the model to adapt to increasing complexity:
* **Phase 1 (Easy)**: Establishes a foundational multilingual baseline.
* **Phase 2 (Medium)**: Refines the decision boundary with more nuanced examples.
* **Phase 3 (Hard)**: Performs "model sharpening" on the most difficult edge cases to improve F1-score.



#### 3. Multi-Layer Hybrid Architecture
Our architecture leverages the structural depth of LaBSE to capture both syntactic and semantic features:
* **Weighted Layer Aggregation**: We extract hidden states from the **last 4 layers** of the encoder.
* **Learnable Parameters**: These layers are weighted using learnable parameters ($w_1, w_2, w_3, w_4$), initialized with small values to let the model learn the optimal layer importance.
* **Hybrid Multi-Layer Pooling**: For each layer, we compute a combined representation:
  $$\text{Layer Output} = \text{Mean Pooling} + \lambda \times (\text{Attention Pooling})$$
  This ensures the model captures both global sentence context and specific high-impact tokens.

![Architecture](./images/labsearchitecture.png)

---
### II. Exploration: RemBERT with Continual Pre-training

In parallel with our primary model, we explore **Domain-Adaptive Pre-training (DAPT)** using **RemBERT** to benchmark against modern multilingual models.

* **Unified Language MLM**: We perform **Masked Language Modeling (MLM)** on the provided dataset, training on all 15 languages simultaneously. This allows the model to find cross-lingual synergies in how polarization is expressed across different cultures.
* **Domain Adaptation**: Through MLM, RemBERT learns domain-specific slang, polarizing hashtags, and sentiment-heavy vocabulary often missing from general pre-training.
* **CLS Pooling**: For classification, we utilize standard **CLS Pooling** (extracting the `<s>` token representation) to summarize the sequence.
* **We do the same curricullam learning technique along with cross lingual learning techniques used for our ""primary model**
---



### III. Baseline Comparison: EuroBERT (State-of-the-Art Multilingual Encoder)

To evaluate the effectiveness of our specialized architectures, we benchmark against **EuroBERT-210m**, a member of the EuroLLM family designed for high-performance European language processing.

* **Temporal Awareness**: Unlike many classic encoders, EuroBERT was trained on a massive web corpus with data extending up to **early February 2026**. This makes it uniquely "event-aware" and capable of understanding recent socio-political trends and modern internet slang relevant to online polarization.
* **Fine-Tuning Strategy**: We perform **Full Fine-Tuning** of the encoder rather than linear probing. This allows all 210 million parameters to adapt to the specific nuances of the SemEval polarization task.
* **Optimization Profile**: Following official recommendations for EuroBERT, we utilize specific hyperparameters to ensure training stability:
    * **Learning Rate**: $3.6 \times 10^{-5}$ for classification.
    * **Betas**: Adjusted to `(0.9, 0.95)` to manage gradient momentum more effectively.
    * **Epsilon**: Increased to $1 \times 10^{-5}$ to prevent numerical instability during backpropagation.
* **Learning Rate Scheduler**: We implement a **Linear Warmup-Stable-Decay (WSD)** scheduler. This involves a $10\%$ warmup phase to prevent early weight destabilization, followed by a stable training period and a final linear decay to zero.



#### **Key Advantages for Task 9**
* **Multilingual Alignment**: EuroBERT exhibits stronger performance in cross-lingual transfer compared to similarly sized models (like XLM-RoBERTa-base), making it a formidable baseline for multicultural polarization detection.

---

---

### IV. Hybrid LLM Approach: Sentence Retrieval with MMR Re-ranking

In this approach, we leverage **LaBSE**'s high-quality multilingual embedding space to identify relevant contextual examples, which are then used to guide **Qwen-14B** through in-context learning.

#### 1. Multilingual Sentence Retrieval
We utilize **LaBSE** as a dense retriever to find the most semantically similar sentences from our reference set for a given input query. This allows the model to "see" how similar polarization manifests across different languages and cultural contexts before making a final prediction.

#### 2. Maximal Marginal Relevance (MMR) for Diversity
To ensure that the retrieved examples are not just relevant but also diverse, we implement the **Maximal Marginal Relevance (MMR)** algorithm. This prevents the model from being biased by redundant or repetitive examples in the prompt.

**The Mathematics of MMR:**
For a given query $Q$, MMR iteratively selects a sentence $D_i$ from the candidate set $R$ that maximizes the following objective function:

$$\text{MMR} = \arg\max_{D_i \in R\setminus S} \left[ \alpha \cdot \text{Sim}_1(D_i, Q) - (1 - \alpha) \cdot \max_{D_j \in S} \text{Sim}_2(D_i, D_j) \right]$$

* **$\text{Sim}_1(D_i, Q)$**: The cosine similarity between the candidate sentence and the query (ensures **relevance**).
* **$\max \text{Sim}_2(D_i, D_j)$**: The maximum similarity between the candidate and sentences already selected for the prompt (minimizes **redundancy**).
* **$\alpha$**: A weighting factor (typically 0.5) that balances the trade-off between relevance and diversity.



#### 3. Zero-Shot / Few-Shot Prompting with Qwen-14B
The top $k$ diverse sentences selected via MMR are formatted into a structured prompt. **Qwen-14B** then analyzes the current input based on these retrieved "anchors," allowing it to better navigate the complexities of multicultural polarization without requiring extensive fine-tuning.

---

### V. Phonemic Branch: Text + IPA Augmentation

A novel sub-branch of our architecture investigates the impact of **Acoustic-Linguistic features** on detection performance. We augment the standard UTF-8 text with its **International Phonetic Alphabet (IPA)** transcription.

* **Multilingual Phonemic Mapping**: We concatenate the raw text with its phonemic equivalent: `[Text] <SEP> [IPA_Transcription]`.
* **Robustness to Creative Spelling**: Polarizing content often uses intentional misspellings or "leetspeak" to bypass text filters. By analyzing the phonemic "sound" of the words, the model can identify hidden intent that remains consistent regardless of character manipulation.
* **Performance Gain**: This phonemic augmentation consistently improves the **macro F1-score by 0.02** across nearly all 15 languages in the dataset.

  ![Architecture](./images/qwenretrieval.png)



---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* CUDA-enabled GPU (recommended)
* Virtual Environment (recommended)

### Installation
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/PolarMind.git](https://github.com/your-username/PolarMind.git)
   cd PolarMind
