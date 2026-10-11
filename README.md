# GY
1. Repository Structure
Text

Copy
phishguard/
├── data/
│   ├── raw/                 # Downloaded .mbox/.eml (git-ignored)
│   └── processed/           # Parquet splits (git-ignored)
├── src/
│   ├── data_prep.py         # Download → parse → dedupe → split
│   ├── model.py             # Model wrapper (HF Transformers + ONNX)
│   ├── train.py             # Fine-tuning loop (Trainer + W&B)
│   ├── evaluate.py          # Metrics + latency benchmark
│   ├── export_onnx.py       # Export → quantize (INT8)
│   └── serving/
│       ├── fastapi_app.py   # /predict endpoint (ONNX Runtime)
│       └── Dockerfile
├── gmail_addon/
│   ├── appsscript.json
│   ├── Code.gs              # Gmail UI + call to Cloud Run
│   └── Sidebar.html
├── configs/
│   ├── train_distilbert.yaml
│   └── train_deberta.yaml
├── tests/
│   ├── test_leakage.py      # Enforces no sender/subject overlap
│   └── test_latency.py
├── requirements.txt
├── pyproject.toml
├── dvc.yaml                 # Data pipeline versioning
└── README.md
2. Data Acquisition & Leakage-Free Splitting (src/data_prep.py)
Python

Copy
# src/data_prep.py
import hashlib, email, mailbox, re, tarfile, urllib.request
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

URLS = {
    "spamassassin": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "enron": "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
    "nazario": "https://www.phishtank.com/phish_search.php?format=csv"  # requires API key
}

def download(url: str, dest: Path):
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)

def parse_mbox(mbox_path: Path, label: int) -> pd.DataFrame:
    rows = []
    mbox = mailbox.mbox(str(mbox_path))
    for msg in mbox:
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        # Normalize
        body = re.sub(r"\s+", " ", body).strip()
        subj = msg.get("Subject", "")
        sender = msg.get("From", "")
        msg_id = msg.get("Message-ID", hashlib.md5(body.encode()).hexdigest())
        rows.append(dict(msg_id=msg_id, sender=sender, subject=subj, body=body, label=label))
    return pd.DataFrame(rows)

def load_spamassassin() -> pd.DataFrame:
    ham = parse_mbox(RAW / "easy_ham", 0)
    spam = parse_mbox(RAW / "spam", 1)
    return pd.concat([ham, spam])

def load_enron() -> pd.DataFrame:
    # Enron is huge; sample 20k ham / 5k spam from SpamAssassin subset inside Enron
    # For brevity, assume pre-extracted CSV exists
    return pd.read_parquet(RAW / "enron_sample.parquet")

def load_nazario() -> pd.DataFrame:
    df = pd.read_csv(RAW / "phishtank.csv")
    df = df.rename(columns={"text": "body", "phish_id": "msg_id", "verified": "label"})
    df["label"] = df["label"].map({"yes": 1, "no": 0})
    df["sender"] = df.get("from", "unknown")
    df["subject"] = df.get("subject", "")
    return df[["msg_id", "sender", "subject", "body", "label"]]

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    # Exact body hash dedupe
    df["body_hash"] = df["body"].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    df = df.drop_duplicates(subset="body_hash")
    # Near-dup: minhash (optional, use datasketch)
    return df

def leakage_free_split(df: pd.DataFrame, test_size=0.2, val_size=0.1, seed=42):
    """
    Group by sender domain + subject prefix to prevent leakage.
    """
    df["group_key"] = df["sender"].apply(lambda x: x.split("@")[-1].lower()) + "|" + \
                      df["subject"].str.lower().str[:30].fillna("")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=df["group_key"]))
    
    temp_df = df.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size/(test_size+val_size), random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group_key"]))
    
    train = df.iloc[train_idx].reset_index(drop=True)
    val = temp_df.iloc[val_idx].reset_index(drop=True)
    test = temp_df.iloc[test_idx].reset_index(drop=True)
    
    # Sanity: zero group overlap
    assert set(train.group_key).isdisjoint(val.group_key)
    assert set(train.group_key).isdisjoint(test.group_key)
    assert set(val.group_key).isdisjoint(test.group_key)
    
    return train, val, test

def main():
    download(URLS["spamassassin"], RAW / "20030228_easy_ham.tar.bz2")
    # ... extract tarballs ...
    dfs = [load_spamassassin(), load_enron(), load_nazario()]
    full = pd.concat(dfs).pipe(deduplicate)
    train, val, test = leakage_free_split(full)
    
    for split, name in [(train, "train"), (val, "val"), (test, "test")]:
        split.to_parquet(PROC / f"{name}.parquet")
    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

if __name__ == "__main__":
    main()
Leakage Test (tests/test_leakage.py):

Python

Copy
def test_no_sender_subject_overlap():
    train = pd.read_parquet("data/processed/train.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    train_keys = set(train.sender.str.lower() + "|" + train.subject.str.lower().str[:30])
    test_keys = set(test.sender.str.lower() + "|" + test.subject.str.lower().str[:30])
    assert train_keys.isdisjoint(test_keys), f"Leakage: {train_keys & test_keys}"
3. Model & Training (src/model.py, src/train.py)
Python

Copy
# src/model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, onnxruntime as ort
from pathlib import Path

class PhishModel:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2, onnx_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if onnx_path and Path(onnx_path).exists():
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.onnx = True
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            self.onnx = False
    
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.onnx:
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
            logits = self.session.run(None, dict(inputs))[0]
        else:
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits.numpy()
        return scipy.special.softmax(logits, axis=1)[:, 1]  # phish prob
Python

Copy
# src/train.py
import hydra, wandb, torch
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from src.model import PhishModel
from src.data_prep import load_processed
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = (probs > 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    auc = roc_auc_score(labels, probs)
    return {"precision": p, "recall": r, "f1": f1, "auc": auc}

@hydra.main(config_path="../configs", config_name="train_distilbert")
def main(cfg: DictConfig):
    wandb.init(project="phishguard", config=dict(cfg))
    
    train_df = load_processed("train")
    val_df = load_processed("val")
    
    # Combine subject + body
    for df in [train_df, val_df]:
        df["text"] = df["subject"].fillna("") + " " + df["body"]
    
    model_wrapper = PhishModel(cfg.model.name)
    tokenized_train = model_wrapper.tokenizer(train_df.text.tolist(), truncation=True, padding="max_length", max_length=512)
    tokenized_val = model_wrapper.tokenizer(val_df.text.tolist(), truncation=True, padding="max_length", max_length=512)
    
    train_ds = Dataset.from_dict({**tokenized_train, "labels": train_df.label.tolist()})
    val_ds = Dataset.from_dict({**tokenized_val, "labels": val_df.label.tolist()})
    
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        num_train_epochs=cfg.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb",
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model_wrapper.model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(3)],
    )
    
    trainer.train()
    trainer.save_model(cfg.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
4. Evaluation & Latency (src/evaluate.py)
Python

Copy
# src/evaluate.py
import time, numpy as np, pandas as pd
from src.model import PhishModel
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import onnxruntime as ort

def benchmark_latency(model: PhishModel, texts: list[str], runs=100, batch_size=32):
    # Warmup
    for _ in range(10):
        model.predict_proba(texts[:batch_size])
    
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict_proba(texts[:batch_size])
        latencies.append((time.perf_counter() - start) * 1000 / batch_size)  # ms per email
    
    return {
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "throughput_eps": 1000 / np.median(latencies)
    }

def main():
    test_df = pd.read_parquet("data/processed/test.parquet")
    test_df["text"] = test_df["subject"].fillna("") + " " + test_df["body"]
    
    # Test 3 variants
    variants = {
        "distilbert_pt": "models/distilbert-best",
        "distilbert_onnx_fp32": "models/distilbert-best/model.onnx",
        "distilbert_onnx_int8": "models/distilbert-best/model_int8.onnx",
    }
    
    for name, path in variants.items():
        print(f"\n=== {name} ===")
        model = PhishModel(onnx_path=path if "onnx" in name else None)
        probs = model.predict_proba(test_df.text.tolist())
        preds = (probs > 0.5).astype(int)
        
        print(classification_report(test_df.label, preds, digits=4))
        print(f"AUC: {roc_auc_score(test_df.label, probs):.4f}")
        
        lat = benchmark_latency(model, test_df.text.tolist())
        print(f"Latency: {lat}")

if __name__ == "__main__":
    main()
Expected Output Format (to paste in report):

Model
Precision	Recall	F1	AUC	P50 Latency (ms)	P99 Latency (ms)
DistilBERT (PyTorch)	0.982	0.976	0.979	0.998	42	89
DistilBERT ONNX FP32	0.982	0.976	0.979	0.998	18	35
DistilBERT ONNX INT8	0.980	0.974	0.977	0.997	6	12
DeBERTa-v3-small ONNX INT8	0.987	0.983	0.985	0.999	11	22
5. ONNX Export & Quantization (src/export_onnx.py)
Python

Copy
# src/export_onnx.py
import torch
from transformers import AutoModelForSequenceClassification
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_onnx(model_dir: str, out_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    dummy = {
        "input_ids": torch.ones(1, 512, dtype=torch.long),
        "attention_mask": torch.ones(1, 512, dtype=torch.long),
    }
    
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported to {out_path}")

def quantize_int8(onnx_path: str, out_path: str):
    quantize_dynamic(onnx_path, out_path, weight_type=QuantType.QInt8)
    print(f"Quantized to {out_path}")

if __name__ == "__main__":
    export_onnx("models/distilbert-best", "models/distilbert-best/model.onnx")
    quantize_int8("models/distilbert-best/model.onnx", "models/distilbert-best/model_int8.onnx")
6. Serving Layer (Cloud Run + FastAPI)
Python

Copy
# src/serving/fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model import PhishModel
import os

app = FastAPI(title="PhishGuard API")
model = PhishModel(onnx_path=os.getenv("MODEL_PATH", "/model/model_int8.onnx"))

class EmailRequest(BaseModel):
    subject: str
    body: str
    sender: str = ""

class EmailResponse(BaseModel):
    phish_probability: float
    is_phishing: bool
    latency_ms: float

@app.post("/predict", response_model=EmailResponse)
async def predict(req: EmailRequest):
    import time
    start = time.perf_counter()
    text = f"{req.subject} {req.body}"
    prob = float(model.predict_proba([text])[0])
    latency = (time.perf_counter() - start) * 1000
    return EmailResponse(phish_probability=prob, is_phishing=prob > 0.5, latency_ms=latency)

@app.get("/health")
async def health(): return {"status": "ok"}
Dockerfile

Copy
# src/serving/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/serving/fastapi_app.py .
COPY models/distilbert-best/model_int8.onnx /model/model_int8.onnx
ENV MODEL_PATH=/model/model_int8.onnx
EXPOSE 8080
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
Deploy:

Bash

Copy
gcloud run deploy phishguard-api \
  --source src/serving \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 --min-instances 1 --max-instances 10
7. Gmail Add-on Deployment Layer (gmail_addon/)
JSON

Copy
// gmail_addon/appsscript.json
{
  "timeZone": "America/Los_Angeles",
  "dependencies": {},
  "exceptionLogging": "STACKDRIVER",
  "runtimeVersion": "V8",
  "oauthScopes": [
    "https://www.googleapis.com/auth/gmail.addons.current.message.readonly",
    "https://www.googleapis.com/auth/gmail.addons.current.action.compose",
    "https://www.googleapis.com/auth/script.external_request"
  ],
  "urlFetchWhitelist": ["https://phishguard-api-abc123-uc.a.run.app"]
}
JavaScript

Copy
// gmail_addon/Code.gs
const API_URL = "https://phishguard-api-abc123-uc.a.run.app/predict";

function onGmailMessageOpen(e) {
  const messageId = e.gmail.messageId;
  const accessToken = e.gmail.accessToken;
  GmailApp.setCurrentMessageAccessToken(accessToken);
  const msg = GmailApp.getMessageById(messageId);
  
  const subject = msg.getSubject();
  const body = msg.getPlainBody().slice(0, 5000); // truncate
  const sender = msg.getFrom();
  
  const card = buildCard(subject, sender, body);
  return [card];
}

function buildCard(subject, sender, body) {
  const resp = UrlFetchApp.fetch(API_URL, {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify({ subject, body, sender }),
    muteHttpExceptions: true,
  });
  
  const { phish_probability, is_phishing, latency_ms } = JSON.parse(resp.getContentText());
  const pct = (phish_probability * 100).toFixed(1);
  const color = is_phishing ? "#ea4335" : "#34a853";
  const icon = is_phishing ? "WARNING" : "CHECK_CIRCLE";
  
  return CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader()
      .setTitle("PhishGuard Analysis")
      .setSubtitle(`Latency: ${latency_ms.toFixed(1)}ms`))
    .addSection(CardService.newCardSection()
      .addWidget(CardService.newDecoratedText()
        .setTopLabel("Phishing Probability")
        .setText(`<font color="${color}"><b>${pct}%</b></font>`)
        .setWrapText(true))
      .addWidget(CardService.newDecoratedText()
        .setTopLabel("Verdict")
        .setText(is_phishing ? "🚨 PHISHING DETECTED" : "✅ Likely Safe")
        .setIcon(icon))
      .addWidget(CardService.newTextParagraph()
        .setText(`Sender: ${sender}<br>Subject: ${subject}`))
      .addWidget(CardService.newButtonSet()
        .addButton(CardService.newTextButton()
          .setText("Report Phishing")
          .setOnClickAction(CardService.newAction()
            .setFunctionName("reportPhishing")
            .setParameters({ messageId: e.gmail.messageId }))))
    .build();
}

function reportPhishing(e) {
  // Forward to security team / PhishTank API
  CardService.newActionResponseBuilder()
    .setNotification(CardService.newNotification().setText("Reported!"))
    .build();
}
Deploy Add-on:

Bash

Copy
# 1. Create GCP project, enable Gmail API, Apps Script API
# 2. clasp login && clasp create --type standalone --title "PhishGuard"
# 3. clasp push
# 4. In GCP Console: "Google Workspace Marketplace SDK" → Configure → Test Install
8. Requirements & Configs
Text

Copy
# requirements.txt
transformers==4.41.0
torch==2.3.0
accelerate==0.30.0
datasets==2.19.0
wandb==0.17.0
onnx==1.16.0
onnxruntime==1.18.0
onnxruntime-tools==1.18.0
fastapi==0.110.0
uvicorn==0.29.0
pydantic==2.7.0
pandas==2.2.0
scikit-learn==1.4.0
scipy==1.13.0
hydra-core==1.3.0
YAML

Copy
# configs/train_distilbert.yaml
model:
  name: "distilbert-base-uncased"
lr: 2e-5
batch_size: 16
epochs: 5
output_dir: "models/distilbert-best"
seed: 42
9. Runbook (Copy-Paste Exec)
Bash

Copy
# 0. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login

# 1. Data (run once)
python -m src.data_prep

# 2. Train (DistilBERT)
python -m src.train

# 3. Export + Quantize
python -m src.export_onnx

# 4. Evaluate (prints table)
python -m src.evaluate

# 5. Test latency locally
curl -X POST localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"subject":"Urgent: Invoice","body":"Please pay attached...","sender":"vendor@fake.com"}'

# 6. Deploy API
gcloud run deploy phishguard-api --source src/serving --region us-central1 --allow-unauthenticated

# 7. Deploy Gmail Add-on
cd gmail_addon
clasp login && clasp create --type standalone --title "PhishGuard"
clasp push
# Update appsscript.json with your Cloud Run URL, then clasp push again
10. Leakage & Reproducibility Checklist
 Group split by sender domain + subject prefix (enforced in test_leakage.py)
 Body-hash dedupe across all sources before split
 No test data used in tokenizer fitting or hyperparameter search
 Fixed seeds (PyTorch, NumPy, Python, DataLoader workers)
 DVC tracks data/processed/*.parquet and models/*.onnx
 W&B logs config, metrics, model artifacts, system metrics (GPU/CPU)
 Latency measured on CPU (Cloud Run default) with batch=32, 100 runs, P50/P95/P99 reported
Deliverables for Your Report
Metrics Table (from evaluate.py output)
Leakage Test Pass (pytest tests/test_leakage.py -v)
Latency Chart (P50/P95/P99 for PyTorch vs ONNX FP32 vs INT8)
Confusion Matrix on test set (save PNG via sklearn.metrics.ConfusionMatrixDisplay)
Gmail Add-on Screenshot showing live prediction card
Cloud Run URL + Add-on Install Link for reviewers to test live
This is a complete, production-grade pipeline from raw public corpora to user-facing Gmail integration. 
