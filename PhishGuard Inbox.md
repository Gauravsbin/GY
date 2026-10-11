# PhishGuard

Inbox risk dashboard for Gmail exports

The model runs inside this page. Emails you load are scored in your browser and are not uploaded anywhere.

Flag at 0.50

To use your own mail: open the Fetch Emails page of the PhishGuard Streamlit app, download the inbox or spam CSV, and drop it here. Columns named subject, sender, date, body and label are recognised.

Subject (optional) Message text

The text is scored by the same model as the inbox. The example below is written for this page and is not from any dataset.

## Measured test results

2,627 held-out emails, one stratified split (seed 42). Brackets are 95% bootstrap intervals. This dashboard uses the logistic regression model.

| Model | Precision | Recall | F1 | ROC-AUC | P50 ms |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + Logistic Regression | 0.988 | 0.969 | 0.978 | 0.998 | 0.50 |
| TF-IDF + Naive Bayes | 0.992 | 0.924 | 0.957 | 0.998 | 1.23 |
| TF-IDF + Linear SVM | 0.989 | 0.971 | 0.980 | 0.999 | 0.45 |

## F1 with 95% interval

## What these numbers do not show

- Only exact duplicates were removed. Near-duplicates and repeated campaigns can sit on both sides of the split, so the scores are probably optimistic.
- The training corpus appears to mix generic spam, advance-fee messages and ordinary business mail. A high score here does not mean the model separates credential phishing from everything else.
- The model reads message text only. It does not see headers, links, sender authentication or attachments.
- It has not been tested on your own mail or on a second corpus. Treat a verdict as a prompt to look, and keep using Gmail's own protection.