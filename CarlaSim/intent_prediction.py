"""Intent prediction (thesis §4.2.2): load PIE nine-feature RF classifier for early crossing intent."""

import os

def load_rf_payload(path):
    """Load joblib payload dict with keys: model, feature_dim, threshold, model_type."""
    try:
        import joblib
    except ImportError:
        print("[simulation] joblib not installed; RF intent disabled.")
        return None
    if not os.path.isfile(path):
        print(f"[simulation] RF model not found at {path} — intent always treated as absent.")
        return None
    try:
        payload = joblib.load(path)
        return payload
    except Exception as ex:
        print(f"[simulation] Failed to load RF model: {ex}")
        return None
