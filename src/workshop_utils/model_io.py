"""
model_io.py
-----------
Utility functions for saving and loading PyTorch model checkpoints.

Used by:
  - scripts/train_power_flow_surrogate.py   (saving)
  - notebooks/02_power_flow_surrogate.ipynb (loading)

Model file distribution strategy
----------------------------------
All .pt files are stored on Google Drive — regardless of size.
They are NOT committed to git.

Workflow:
  1. Train models locally (scripts/train_power_flow_surrogate.py).
  2. Upload the .pt files to the shared Google Drive folder.
  3. Copy the shareable file ID from Drive and paste into models/README.md.
  4. Notebooks call download_model_if_missing() at startup to fetch
     the file automatically — works in Colab and locally.

The .gitignore should exclude *.pt files:
  models/**/*.pt

Participants never need to train anything. They always start from
a downloaded checkpoint.
"""

from pathlib import Path
import torch


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(model, path, metadata=None):
    """
    Save a model's state_dict to a .pt file with optional metadata.

    After saving, upload the file to Google Drive and record the
    shareable link in models/README.md.

    Parameters
    ----------
    model    : nn.Module   — the trained PyTorch model
    path     : str | Path  — local destination path
    metadata : dict | None — extra info stored alongside weights, e.g.
                             {"epochs_trained": 200, "train_loss": 0.0012,
                              "input_features": [...], "perturbation_range": [0.8, 1.2]}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }

    torch.save(checkpoint, path)
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"Checkpoint saved → {path}  ({size_mb:.2f} MB)")
    if metadata:
        print(f"  Metadata: {metadata}")
    print("  Next step: upload this file to Google Drive and record the")
    print("  file ID in models/README.md.")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(model, path, device):
    """
    Load saved weights into an already-constructed model architecture.

    If the file does not exist locally, it must be downloaded first using
    download_model_if_missing(). The notebook startup cell handles this
    automatically when a gdrive_id is provided in models/README.md.

    Parameters
    ----------
    model  : nn.Module    — empty model with the correct architecture
    path   : str | Path   — local path to the .pt file
    device : torch.device — where to place the model (cuda or cpu)

    Returns
    -------
    model    : nn.Module — model with loaded weights, in eval mode, on device
    metadata : dict      — info stored at save time (empty dict if none)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nModel file not found: {path}\n"
            "Run download_model_if_missing() before load_checkpoint().\n"
            "See models/README.md for the Google Drive file ID."
        )

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = checkpoint.get("metadata", {})
    else:
        # Plain state_dict with no metadata wrapper
        model.load_state_dict(checkpoint)
        metadata = {}

    model.to(device)
    model.eval()

    print(f"Loaded model from: {path}")
    if metadata:
        print(f"  Trained for {metadata.get('epochs_trained', '?')} epochs  |  "
              f"val loss: {metadata.get('final_val_loss', '?')}")
    return model, metadata


# ---------------------------------------------------------------------------
# Google Drive download helper
# ---------------------------------------------------------------------------

def download_model_if_missing(path, gdrive_id: str):
    """
    Download a .pt file from Google Drive if it is not already present locally.

    Call this at the top of each notebook, before load_checkpoint().
    It is safe to call every time — if the file already exists it does nothing.

    Parameters
    ----------
    path       : str | Path — local destination path
                              e.g. "models/instructor/power_flow_pretrained.pt"
    gdrive_id  : str        — Google Drive file ID (the long string in the
                              shareable link after /d/ or ?id=)

    How to get the Google Drive file ID
    ------------------------------------
    1. Upload the .pt file to Google Drive.
    2. Right-click → Share → Change to "Anyone with the link".
    3. Copy the link. It looks like:
           https://drive.google.com/file/d/1A2B3C4D5E6F7G8H/view?usp=sharing
       The file ID is the part between /d/ and /view:
           1A2B3C4D5E6F7G8H
    4. Paste that ID into models/README.md and into this function call.

    Example
    -------
    download_model_if_missing(
        path      = MODEL_DIR / "power_flow_pretrained.pt",
        gdrive_id = "1A2B3C4D5E6F7G8H9I0J"
    )
    """
    path = Path(path)

    if path.exists():
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"Model already present: {path}  ({size_mb:.2f} MB)")
        return

    print(f"Downloading: {path.name}  (Google Drive ID: {gdrive_id})")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for Google Drive downloads.\n"
            "Install it with:  pip install gdown\n"
            "In Colab:         !pip install -q gdown"
        )

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, str(path), quiet=False)

    if not path.exists():
        raise RuntimeError(
            f"Download failed — {path.name} not found after download.\n"
            "Check that the Google Drive file is shared as "
            "'Anyone with the link' and the file ID is correct."
        )

    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"Download complete: {path}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn as nn
    import tempfile, os

    print("Running model_io self-test...")

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
        def forward(self, x):
            return self.fc(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TinyNet().to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_checkpoint.pt")

        # Save
        save_checkpoint(model, path, metadata={"epochs_trained": 5, "test": True})

        # Load
        fresh, meta = load_checkpoint(TinyNet(), path, device)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                       fresh.named_parameters()):
            assert torch.allclose(p1, p2), f"Weight mismatch in {n1}"

        print("  Weight comparison: PASSED")
        print(f"  Metadata recovered: {meta}")

        # Verify missing file raises clearly
        try:
            load_checkpoint(TinyNet(), path + "_missing.pt", device)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            print(f"  Missing file error: PASSED")

    print("Self-test complete.")