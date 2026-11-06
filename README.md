# ⚡ Applied AI & Deep Learning Lab: Google Colab Production Notebooks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU%20Enabled-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Generative%20AI-000000?style=for-the-badge)](https://www.langchain.com/)
[![HuggingFace](https://img.shields.io/badge/Transformers-Attention%20Head-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

<p align="center">
  <strong>Curated collection of 17 end-to-end applied AI, deep learning, computer vision, and machine learning research notebooks engineered and trained on Google Colab GPU runtimes.</strong>
</p>

</div>

---

## 📌 Executive Architecture & Laboratory Modules

This repository houses production-grade notebooks spanning modern foundational artificial intelligence. Each notebook is fully documented, pre-configured with dependencies, and includes one-click **"Open in Colab"** execution badges for immediate GPU-accelerated experimentation.

```
applied-ai-deep-learning-colab-lab/
├── 01-rag-and-generative-ai/             # Retrieval-Augmented Generation & Autonomous Agents
│   ├── RAG_Implementation_Colab.ipynb
│   ├── LangChain_Data_Agent.ipynb
│   ├── custom_chatbot_beginner.ipynb
│   └── Generative_AI_Part_1.ipynb
├── 02-transformers-and-nlp/              # Self-Attention Mechanisms & Sequence-to-Sequence
│   ├── Transformer.ipynb
│   ├── transformer_programs.ipynb
│   ├── emotion_text_classification.ipynb
│   └── Sentiment_Analysis.ipynb
├── 03-computer-vision-cnn/               # Deep Convolutional Feature Extraction & Classification
│   └── CNN_Project.ipynb
├── 04-deep-learning-ann-rnn/             # Multi-Layer Perceptrons, Backpropagation & Sequential RNNs
│   ├── DPL_ANN_Project.ipynb
│   ├── Deep_Learning.ipynb
│   ├── RNN.ipynb
│   └── RNN_Second_Project.ipynb
└── 05-machine-learning-predictive-modeling/ # Statistical Modeling, Feature Attribution & EDA
    ├── Heart_EDA.ipynb
    ├── Ford_Analysis.ipynb
    ├── Data_Exploration_and_Modelling.ipynb
    └── Project_Cricket.ipynb
```

---

## 🚀 Interactive Laboratory Directory & One-Click Colab Launch

### 1. 🤖 Retrieval-Augmented Generation (RAG) & Generative AI Agents

Modern LLMs suffer from knowledge cutoff and hallucinatory outputs. These notebooks implement semantic vector indexing, chunking strategies, dense retrieval pipelines, and autonomous tool-augmented LLM reasoning.

| Notebook | Focus & Architecture | Key Libraries | Launch in Colab |
| :--- | :--- | :--- | :---: |
| **`RAG_Implementation_Colab.ipynb`** | End-to-end vector embeddings, cosine similarity search, chunking & context injection for grounded Q&A. | `LangChain`, `ChromaDB`, `Sentence-Transformers` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/01-rag-and-generative-ai/RAG_Implementation_Colab.ipynb) |
| **`LangChain_Data_Agent.ipynb`** | Autonomous data reasoning agent utilizing dynamic schema inspection and natural language analytical query resolution. | `LangChain`, `OpenAI`, `Pandas` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/01-rag-and-generative-ai/LangChain_Data_Agent.ipynb) |
| **`custom_chatbot_beginner.ipynb`** | Conversational state management, memory persistence buffers, and prompt orchestration. | `Python`, `LangChain`, `Transformers` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/01-rag-and-generative-ai/custom_chatbot_beginner.ipynb) |
| **`Generative_AI_Part_1.ipynb`** | Foundations of generative sampling: temperature, Top-$k$, Top-$p$ (nucleus) decoding, and token likelihoods. | `HuggingFace`, `PyTorch` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/01-rag-and-generative-ai/Generative_AI_Part_1.ipynb) |

---

### 2. ⚡ Transformer Architectures & Natural Language Processing

Mathematical implementation of scaled dot-product attention and multi-head representation learning:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Notebook | Focus & Architecture | Key Libraries | Launch in Colab |
| :--- | :--- | :--- | :---: |
| **`Transformer.ipynb`** | Full structural decomposition of encoder/decoder layers, residual normalization, and sinusoidal positional embeddings. | `PyTorch`, `TensorFlow` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/02-transformers-and-nlp/Transformer.ipynb) |
| **`transformer_programs.ipynb`** | Sequence modeling, attention matrix visualization, and language modeling fine-tuning. | `Transformers`, `PyTorch` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/02-transformers-and-nlp/transformer_programs.ipynb) |
| **`emotion_text_classification.ipynb`** | Multi-class granular text classification with tokenization, embedding spaces, and evaluation metrics. | `Scikit-Learn`, `Keras`, `NLTK` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/02-transformers-and-nlp/emotion_text_classification.ipynb) |
| **`Sentiment_Analysis.ipynb`** | End-to-end sentiment classification with TF-IDF, Word2Vec representation, and classification report auditing. | `NLTK`, `Scikit-Learn`, `Seaborn` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/02-transformers-and-nlp/Sentiment_Analysis.ipynb) |

---

### 3. 👁️ Computer Vision & Convolutional Neural Networks (CNN)

Extracting spatial hierarchies via parameter sharing and localized receptive fields:

| Notebook | Focus & Architecture | Key Libraries | Launch in Colab |
| :--- | :--- | :--- | :---: |
| **`CNN_Project.ipynb`** | 2D Convolutional layers, Max Pooling, Batch Normalization, Dropout regularization, and visual classification metrics. | `TensorFlow`, `Keras`, `OpenCV` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/03-computer-vision-cnn/CNN_Project.ipynb) |

---

### 4. 🧠 Deep Learning Architectures (ANN & RNN/LSTM)

Foundational neural learning dynamics from forward activation passes to sequential temporal dependencies:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

| Notebook | Focus & Architecture | Key Libraries | Launch in Colab |
| :--- | :--- | :--- | :---: |
| **`DPL_ANN_Project.ipynb`** | Multi-Layer Artificial Neural Network (ANN) architecture with backpropagation, gradient descent, and cross-entropy optimization. | `Keras`, `TensorFlow`, `NumPy` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/04-deep-learning-ann-rnn/DPL_ANN_Project.ipynb) |
| **`Deep_Learning.ipynb`** | Deep network hyperparameter exploration: learning rate decay, Adam optimization, and loss convergence dynamics. | `TensorFlow`, `Matplotlib` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/04-deep-learning-ann-rnn/Deep_Learning.ipynb) |
| **`RNN.ipynb`** | Recurrent Neural Network (RNN) sequential hidden state processing on time-series and sequential data. | `Keras`, `NumPy` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/04-deep-learning-ann-rnn/RNN.ipynb) |
| **`RNN_Second_Project.ipynb`** | Extended sequential model handling vanishing gradients, sequence memory, and multi-step prediction. | `Keras`, `Matplotlib` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/04-deep-learning-ann-rnn/RNN_Second_Project.ipynb) |

---

### 5. 📊 Machine Learning, Feature Attribution & Exploratory Analysis

Quantitative statistical modeling, outlier detection, distribution analysis, and regression forecasting:

| Notebook | Focus & Architecture | Key Libraries | Launch in Colab |
| :--- | :--- | :--- | :---: |
| **`Heart_EDA.ipynb`** | Clinical diagnostic cardiology exploratory data analysis on Framingham/Cleveland biometric cohorts. | `Pandas`, `Seaborn`, `Scipy` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/05-machine-learning-predictive-modeling/Heart_EDA.ipynb) |
| **`Ford_Analysis.ipynb`** | Secondary market vehicle pricing regression, depreciation curves across mileage and age. | `Pandas`, `Matplotlib`, `Scikit-Learn` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/05-machine-learning-predictive-modeling/Ford_Analysis.ipynb) |
| **`Data_Exploration_and_Modelling.ipynb`** | Full statistical feature engineering pipeline, collinearity diagnostics, and benchmark modeling. | `Pandas`, `NumPy`, `Scikit-Learn` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/05-machine-learning-predictive-modeling/Data_Exploration_and_Modelling.ipynb) |
| **`Project_Cricket.ipynb`** | Empirical performance analytics, statistical distributions, and predictive feature modeling. | `Pandas`, `Seaborn`, `Scikit-Learn` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab/blob/main/05-machine-learning-predictive-modeling/Project_Cricket.ipynb) |

---

## 🛠️ Local Environment Setup

If you prefer to run these notebooks locally rather than in Google Colab:

```bash
# Clone the repository
git clone https://github.com/Ayan-Rajput-Analysis/applied-ai-deep-learning-colab-lab.git
cd applied-ai-deep-learning-colab-lab

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install essential dependencies
pip install torch torchvision tensorflow langchain chromadb sentence-transformers scikit-learn pandas numpy matplotlib seaborn jupyterlab

# Launch JupyterLab
jupyter lab
```

---

## 👨‍💻 Author & Connect

**Ayan Rajput** — *AI & Deep Learning Engineer | Computer Vision & 3D WebGL Specialist*
- 💼 **LinkedIn**: [linkedin.com/in/ayan-rajput-8781a7397](https://www.linkedin.com/in/ayan-rajput-8781a7397/)
- 🌐 **GitHub Profile**: [@Ayan-Rajput-Analysis](https://github.com/Ayan-Rajput-Analysis)
- 📧 **Email**: [ayanrajput779900@gmail.com](mailto:ayanrajput779900@gmail.com)
