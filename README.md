# FinDocAI – Financial Document Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1t6wJ5SA7qiaZ813TWlcEg0te9DAe6lN4?usp=sharing)

A deep-learning pipeline that classifies financial documents (Balance Sheet,
Cash Flow, Income Statement, Notes, Others) by fine-tuning **FinBERT** on a
labeled HTML document dataset, plus a **Streamlit** app for classifying new
documents through a simple upload UI.

## What's in this repo

- **`Financial_Document_Classification.ipynb`** – the full training pipeline:
  downloads the dataset, parses HTML documents to plain text with
  BeautifulSoup, tokenizes with the FinBERT tokenizer, fine-tunes a
  `TFAutoModelForSequenceClassification` (FinBERT) on the 5 document classes,
  evaluates it, and pushes the trained model/tokenizer to the Hugging Face Hub.
- **`app.py`** – a Streamlit app that lets a user upload an HTML financial
  document, extracts its text, and sends it to a Hugging Face Inference API
  endpoint for classification, showing the predicted class and confidence score.
- **`samples/`** – example HTML documents, one per class, for trying the app.
- **`image/`** – training accuracy/loss plot and an app screenshot.

## Results

On the held-out test split, the fine-tuned model reached **95.84% test
accuracy** (verified from the notebook's evaluation cell output).

## Tech stack

- Python, TensorFlow, Hugging Face `transformers` / `huggingface_hub`
- BeautifulSoup for HTML text extraction
- scikit-learn (train/test split, confusion matrix)
- Pandas, NumPy, Matplotlib, Seaborn
- Streamlit for the demo app, Hugging Face Inference API for serving predictions

## Data and model provenance

The training notebook downloads the dataset via the Kaggle CLI
(`gopiashokan/financial-document-classification-dataset`) and, in the
"Model Deployment" step, pushes the fine-tuned model to the
`gopiashokan/Financial-Document-Classification-using-Deep-Learning` namespace
on the Hugging Face Hub. `app.py` calls that same hosted model through the
Inference API. To deploy the model under your own account, update the
`push_to_hub` calls in the notebook and the `API_URL` in `app.py` accordingly.

## Running the app

```bash
pip install -r requirements.txt
```

Create a `.env` file with a Hugging Face API token:

```
HUGGINGFACE_TOKEN=your_token_here
```

Then start the app:

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload one of the sample HTML files from
`samples/`, and view the predicted document class and confidence score.

## Running the notebook

Open `Financial_Document_Classification.ipynb` (locally or via the Colab
badge above) to reproduce data preprocessing, FinBERT fine-tuning, and
evaluation. It expects a Kaggle API key for the dataset download and a
Hugging Face token to push the trained model.

If you hit a TensorFlow DLL load error on Windows:

```bash
pip uninstall tensorflow
pip install tensorflow==2.12.0 --upgrade
```

## License

Released under the MIT License (see `LICENSE.md`).
