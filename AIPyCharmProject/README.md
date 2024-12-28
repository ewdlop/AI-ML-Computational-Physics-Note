# AIPyCharmProject

This directory contains various scripts and projects related to AI and machine learning. Below is a detailed description of the `AIPyCharmProject` directory and its purpose, along with instructions for running the scripts and any dependencies.

## Purpose

The `AIPyCharmProject` directory is designed to provide a collection of Python scripts and projects that demonstrate various AI and machine learning techniques. These scripts cover a range of topics, including text analysis, vector search, and text embedding.

## Scripts

### 1. `analyze_title_text.py`

This script performs text analysis on YouTube titles. It includes functions for extracting keywords, performing sentiment analysis, assessing readability, and comparing titles with top-performing titles.

### 2. `basic_vector_search.py`

This script demonstrates a basic vector search using FAISS (Facebook AI Similarity Search). It generates random vectors, builds an index, and performs a search to find the nearest neighbors.

### 3. `text_embedding.py`

This script converts text to embeddings using the BERT model and performs a search using FAISS. It includes functions for converting text to embeddings, creating a FAISS index, and performing a search to find the nearest neighbors.

## Dependencies

To run the scripts in this directory, you need to install the following dependencies:

- `nltk`
- `textblob`
- `scikit-learn`
- `faiss`
- `transformers`
- `torch`

You can install these dependencies using `pip`:

```bash
pip install nltk textblob scikit-learn faiss-cpu transformers torch
```

## Instructions

1. Clone the repository:

```bash
git clone https://github.com/ewdlop/AI-ML-Computational-Physics-Note.git
cd AI-ML-Computational-Physics-Note/AIPyCharmProject
```

2. Install the dependencies:

```bash
pip install nltk textblob scikit-learn faiss-cpu transformers torch
```

3. Run the scripts:

- To run `analyze_title_text.py`:

```bash
python analyze_title_text.py
```

- To run `basic_vector_search.py`:

```bash
python basic_vector_search.py
```

- To run `text_embedding.py`:

```bash
python text_embedding.py
```

## NLTK Data

To download NLTK packages, use the following command:

```python
import nltk
nltk.download()
```

For more information, visit the [NLTK Data](https://www.nltk.org/data.html#) page.
