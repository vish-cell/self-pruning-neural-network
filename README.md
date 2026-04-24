## ⚡ Quick Start (Run Everything)

Follow these steps to run the project end-to-end.

---

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/vish-cell/self-pruning-neural-network.git
cd self-pruning-neural-network
```

---

### 🔹 2. Create Virtual Environment

```bash
python -m venv env
```

#### Activate:

**Windows**

```bash
env\Scripts\activate
```

**Mac/Linux**

```bash
source env/bin/activate
```

---

### 🔹 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 4. Train the Model

```bash
python main.py train
```

This will:

* Train models for different λ values
* Print accuracy & sparsity
* Generate plots:

  * `lambda_tradeoff.png`
  * `gate_distribution.png`

---

### 🔹 5. Evaluate Models

```bash
python main.py eval
```
<img width="383" height="160" alt="image" src="https://github.com/user-attachments/assets/240f12d5-e0ae-4747-9733-a0905b708b21" />

This will:

* Evaluate all trained models
* Print comparison results

---

### 🔹 6. Run API (Optional)

```bash
uvicorn api.app:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

### 🔹 7. Test Prediction
<img width="896" height="646" alt="Screenshot 2026-04-24 121928" src="https://github.com/user-attachments/assets/0d698391-8985-4c49-b67c-e5574c0370cb" />

* Go to `/predict`
* Click **Try it out**
* Upload an image (jpg/png)
  <img width="200" height="216" alt="cat_test_pic" src="https://github.com/user-attachments/assets/ff25ebc5-f589-4139-99a9-34b1c59ac1f7" />

* Click **Execute**

Example output:
<img width="847" height="299" alt="Screenshot 2026-04-24 122117" src="https://github.com/user-attachments/assets/c006a550-5acb-4f3b-bf0f-7428401266df" />

```json
{
  "class_id": 3,
  "class_name": "cat"
}
```

---

## 🧠 Notes

* Dataset (CIFAR-10) downloads automatically
* Models are saved during training
* API uses trained model for inference

---

## ⚠️ Troubleshooting

* If model not found → run training first
* If API fails → ensure dependencies installed
* Ignore CIFAR warnings (harmless)

---
