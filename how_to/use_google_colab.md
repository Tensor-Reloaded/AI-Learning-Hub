# Using Google Colab

Google Colab provides free access to hosted Jupyter notebooks with optional GPU/TPU acceleration.

---

## 1. Change Runtime Type

To enable GPU:

1. Go to `Runtime` > `Change runtime type`
2. Set **Hardware accelerator** to **GPU**


**Note**: GPU access may be limited. Runtime quotas apply.

---

## 2. Connect to Google Drive

To read/write files from your Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```


## 3. Load Datasets Efficiently

* Upload your dataset as a `.zip` to Drive
* Use Colab to download/unpack locally 

Avoid reading large datasets directly from Drive during training.

## 4. Save Model Checkpoints to Drive

```python
# Saving
torch.save(model.state_dict(), '/content/drive/MyDrive/models/model.pt')

# Loading
model.load_state_dict(torch.load('/content/drive/MyDrive/models/model.pt'))
model.eval()
```


## 5. Tips

* Colab sessions time out after inactivity.
* Use `torch.save()` frequently if training for long periods.