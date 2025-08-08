Got it — you want this **FinDocAI documentation rewritten so it looks different**, but still communicates all the key technical and project details.
Here’s a **reworded, condensed, and reorganized** version that makes it distinct while keeping your core achievements clear.

---

# **FinDocAI – Automated Financial Document Classification**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1t6wJ5SA7qiaZ813TWlcEg0te9DAe6lN4?usp=sharing)

## **Overview**

Manual sorting of financial statements is tedious and prone to mistakes. **FinDocAI** tackles this by applying **deep learning** and **domain-specific NLP** to automatically classify financial reports with high accuracy.
The system fine-tunes the **FinBERT** model using a curated dataset of financial documents and deploys it as an **interactive Streamlit application**, making classification accessible to technical and non-technical users alike.

---

## **Highlights**

* **Model:** FinBERT fine-tuned for 5-class financial document classification.
* **Accuracy:** Achieved **95.84%** on a custom dataset.
* **Deployment:** Fully functional **Streamlit app** hosted on Hugging Face Spaces with API integration.
* **Efficiency:** Handles HTML parsing, preprocessing, and prediction in seconds.

---

## **Tech Stack**

* **Languages & Libraries:** Python, NumPy, Pandas, scikit-learn, TensorFlow, Transformers (Hugging Face), BeautifulSoup
* **Visualization:** Matplotlib, Seaborn
* **App & Deployment:** Streamlit, Hugging Face Hub API
* **Data Handling:** HTML text extraction, tokenization, padding/truncation

---

## **Setup Instructions**

Install the dependencies:

```bash
pip install python-dotenv datasets tensorflow transformers sentencepiece numpy pandas beautifulsoup4 matplotlib seaborn streamlit streamlit_extras huggingface-hub
```

If TensorFlow DLL load error occurs:

```bash
pip uninstall tensorflow
pip install tensorflow==2.12.0 --upgrade
```

---

## **Running the Application**

1. Clone repo:

   ```bash
   git clone https://github.com/gopiashokan/Finance-Document-Classification-Using-Deep-Learning.git
   ```
2. Install packages:

   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:

   ```bash
   streamlit run app.py
   ```
4. Open browser at **[http://localhost:8501](http://localhost:8501)**

---

## **Workflow**

### **1. Data Collection**

* Dataset contains HTML financial reports categorized as **Balance Sheet, Cash Flow, Income Statement, Notes, Others**.
* Source: [Kaggle Dataset](https://www.kaggle.com/datasets/gopiashokan/financial-document-classification-dataset)

### **2. Preprocessing**

* **HTML Parsing:** Extracted clean text via BeautifulSoup.
* **Label Encoding:** Converted categories to numeric IDs.
* **Data Split:** Training & testing sets via scikit-learn.
* **Tokenization:** Used `yiyanghkust/finbert-pretrain` tokenizer.
* **Sequence Management:** Padded/truncated to length 512.

### **3. Model Training**

* **Transfer Learning:** Fine-tuned FinBERT on labeled dataset.
* **Optimizer & Loss:** Adam + SparseCategoricalCrossentropy.
* **Evaluation Metric:** Accuracy.
* **Result:** **95.84%** classification accuracy.

![Accuracy & Loss](https://github.com/gopiashokan/Finance-Document-Classification-Using-Deep-Learning/blob/main/image/Accuracy_Loss_Graph.jpg)

### **4. Deployment**

* **Hugging Face Hub:** Model + tokenizer uploaded for API access.
* **Streamlit App:** Allows document upload, prediction display, and confidence scoring.
* **Hosted App:** [Try on Hugging Face Spaces](https://huggingface.co/spaces/gopiashokan/Financial-Document-Classification-using-Deep-Learning)

![App Screenshot](https://github.com/gopiashokan/Finance-Document-Classification-Using-Deep-Learning/blob/main/image/Inference.png)

---

## **Key Outcomes**

* Fully automated classification pipeline from **raw HTML → label prediction**.
* Scalable architecture for other domain-specific NLP tasks.
* User-friendly deployment accessible via browser or API.

---

## **References**

* [scikit-learn](https://scikit-learn.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index)
* [Streamlit](https://docs.streamlit.io/)

---

## **License**

Released under the MIT License.

---

This new version is **more concise, avoids paragraph repetition**, and focuses on:

* Impact metrics
* Workflow clarity
* Quick-scan readability for recruiters or technical reviewers

If you want, I can also **adapt this into a compact resume-ready project bullet set** so it matches the style of your *Military Asset Detection* entry. That would make it consistent across your resume.
