
# Toxic Comment Classification using DistilBERT


This project is about finding toxic comments in text. It uses a machine learning model called DistilBERT. I trained it to say if a comment is toxic or not toxic.

---

## Files in This Project

```
📂 toxic-comment-detector/
├──  Assignment 3 (Major Project).ipynb  # Python code for training and testing
├── dataset(reduced).csv   # Small dataset for testing
├──  Assignment 3 (Major Project)report.pdf  # Project report
├── README.md                    # This file
├── requirements.txt             # Needed Python packages
```

---

## How to Use the Project
Please download the model from Google drive link:
https://drive.google.com/file/d/1FftEmnOhYef-fYStSQ3Qmk_YXiDdxK_f/view?usp=sharing

### Step 1: Install Python Libraries

```bash
pip install -r requirements.txt
```

### Step 2: Run the Python Code

Use Google Colab or Jupyter Notebook to open `Assignment 3 (Major Project).ipynb`.

---

## How to Load the Trained Model

After unzipping the model, you can load it like this:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("saved_model/")
tokenizer = AutoTokenizer.from_pretrained("saved_model/")
```

---

## Predict New Comment

```python
def predict_comment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=1).item()
    return "Toxic" if prediction == 1 else "Not Toxic"

# Try a comment
print(predict_comment("I hate you!"))
```

---

## What I Learned

- Using a pre-trained model
- Fine-tuning it for toxic comment detection
- Working with datasets
- Making predictions with Transformers library

---

## Author

**Sadman Sakib**  
Student ID: 48031275
Email: sadman.sakib2@students.mq.edu.au  
Macquarie University
