"""
app.py — Wafer Defect Classifier Demo
======================================
Gradio web app that takes a wafer map image and returns:
  1. Predicted defect class + confidence bar chart
  2. GradCAM heatmap showing which regions the model focused on
  3. Side-by-side overlay of wafer + heatmap

Run from the project root:
    conda activate wafer-ml
    pip install gradio          # one-time install
    python app.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend — required for Gradio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import pickle
import os
from pathlib import Path
import io
from PIL import Image as PILImage

# ── Paths — adjust if your folder structure differs ───────────────────────────
BASE_DIR    = Path(__file__).parent          # Directory where app.py lives
MODEL_PATH  = BASE_DIR / 'data/processed/best_focal.pt'
ENCODER_PATH = BASE_DIR / 'data/processed/label_encoder.pkl'
TEST_PATH   = BASE_DIR / 'data/processed/test.pkl'

IMAGE_SIZE  = 64
NUM_CLASSES = 9

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')


# ─────────────────────────────────────────────────────────────────────────────
# Model & GradCAM — loaded ONCE at startup, reused for every prediction
# Loading inside the predict function would reload weights on every click
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Rebuild ResNet18 with our 9-class head and load saved weights."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


class GradCAM:
    """
    Extracts a spatial attention heatmap using gradient-weighted
    class activation mapping on model.layer3 (8×8 feature maps).
    
    layer3 chosen over layer4 because our 64×64 input gives layer4
    only a 2×2 feature map — too coarse for spatial localization.
    layer3 gives 8×8 — enough resolution to localize defect regions.
    """

    def __init__(self, model):
        self.model        = model
        self.feature_maps = None
        self.gradients    = None
        self._register_hooks()

    def _register_hooks(self):
        def save_features(module, input, output):
            self.feature_maps = output.detach()

        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.model.layer3.register_forward_hook(save_features)
        self.model.layer3.register_full_backward_hook(save_gradients)

    def generate(self, input_tensor):
        """
        Run forward + backward pass and compute the weighted heatmap.
        
        Returns:
            heatmap     — numpy (64, 64) array in [0, 1]
            pred_idx    — integer class index
            probs       — numpy (9,) array of class probabilities
        """
        input_tensor = input_tensor.to(device)

        # Forward pass
        output = self.model(input_tensor)           # (1, 9) raw logits
        probs  = torch.softmax(output, dim=1)       # (1, 9) probabilities
        pred_idx = output.argmax(dim=1).item()

        # Backward pass on predicted class only
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_idx] = 1.0
        output.backward(gradient=one_hot)

        # Importance weight per channel = mean gradient over spatial dims
        weights = self.gradients.mean(dim=[2, 3])[0]    # (256,)

        # Weighted sum of feature maps
        feature_maps = self.feature_maps[0]             # (256, 8, 8)
        cam = torch.zeros(feature_maps.shape[1:], device=feature_maps.device)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]

        # ReLU + normalize + resize to input resolution
        cam = torch.relu(cam).cpu().numpy()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, pred_idx, probs[0].detach().cpu().numpy()


# ── Load everything once when the app starts ──────────────────────────────────
print('Loading model...')
model  = load_model()
gradcam = GradCAM(model)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)
CLASS_NAMES = list(le.classes_)

# Load test set for the "Try an example" gallery
test_df = pd.read_pickle(TEST_PATH)

print(f'Model ready. Classes: {CLASS_NAMES}')


# ─────────────────────────────────────────────────────────────────────────────
# Wafer preprocessing
# Must exactly match what was done during training (notebook 02)
# ─────────────────────────────────────────────────────────────────────────────

def wafer_array_to_tensor(wafer_array):
    """
    Convert a raw wafer map (2D numpy array, values 0/1/2) into
    a (1, 3, 64, 64) float tensor ready for the model.
    """
    # Resize to 64×64 using nearest-neighbor (preserves 0/1/2 values)
    resized = cv2.resize(
        wafer_array.astype(np.float32),
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )
    # Normalize: 0→0.0, 1→0.5, 2→1.0
    normalized = resized / 2.0
    # Shape: (1, 64, 64) → repeat to (3, 64, 64) → add batch → (1, 3, 64, 64)
    tensor = torch.FloatTensor(normalized).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    return tensor


def image_to_wafer_array(pil_image):
    """
    Convert an uploaded PNG/JPG image into a wafer map array (values 0/1/2).
    
    The app accepts two kinds of uploads:
      A) A real wafer map image (64×64 pixels, grayscale-ish, values near 0/128/255)
      B) A screenshot or visualization of a wafer map
    
    We convert to grayscale, resize to 64×64, and threshold into 3 bins.
    """
    img = np.array(pil_image.convert('L'))      # Convert to grayscale
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    # Threshold into 3 bins:
    #   0–85   → 0 (background / no die)
    #   86–170 → 1 (normal die)
    #   171+   → 2 (defective die)
    wafer = np.zeros_like(img, dtype=np.float32)
    wafer[img > 85]  = 1.0
    wafer[img > 170] = 2.0

    return wafer


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_results_figure(wafer_array, heatmap, pred_idx, probs):
    """
    Build a 3-panel matplotlib figure:
      Panel 1: Raw wafer map
      Panel 2: GradCAM heatmap
      Panel 3: Overlay (wafer + heatmap blended)
    
    Returns a numpy RGBA image that Gradio can display directly.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                              facecolor='#1a1a2e')      # Dark background

    # ── Shared style ──────────────────────────────────────────────────────────
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.axis('off')

    # ── Panel 1: Wafer map ────────────────────────────────────────────────────
    # Normalize wafer values (0/1/2) to [0, 1] for display
    wafer_display = wafer_array / 2.0
    axes[0].imshow(wafer_display, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('Wafer Map', color='white', fontsize=12, pad=8)

    # ── Panel 2: GradCAM heatmap ──────────────────────────────────────────────
    axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('GradCAM Attention', color='white', fontsize=12, pad=8)

    # ── Panel 3: Overlay ──────────────────────────────────────────────────────
    # Convert grayscale wafer to RGB for blending
    wafer_rgb    = np.stack([wafer_display] * 3, axis=-1)   # (64, 64, 3)
    heatmap_rgb  = cm.jet(heatmap)[:, :, :3]                # (64, 64, 3)

    # 55% wafer + 45% heatmap — keeps wafer structure visible
    overlay = 0.55 * wafer_rgb + 0.45 * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay)
    pred_name = CLASS_NAMES[pred_idx]
    conf      = probs[pred_idx]
    axes[2].set_title(
        f'Predicted: {pred_name}\n({conf:.1%} confidence)',
        color='#00ff88', fontsize=12, fontweight='bold', pad=8
    )

    plt.tight_layout(pad=1.5)

    # Render figure to numpy array so Gradio can display it
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    result = np.array(PILImage.open(buf).convert('RGB'))
    plt.close(fig)
    return result


def make_confidence_chart(probs):
    """
    Horizontal bar chart of class probabilities.
    Highlights the predicted class in bright green, others in steel blue.
    Returns numpy RGB image.
    """
    pred_idx = int(np.argmax(probs))

    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    colors = ['#00ff88' if i == pred_idx else '#4a90d9'
              for i in range(NUM_CLASSES)]

    # Sort by confidence descending for readability
    sorted_idx  = np.argsort(probs)[::-1]
    sorted_prob = probs[sorted_idx]
    sorted_names = [CLASS_NAMES[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(sorted_names, sorted_prob, color=sorted_colors,
                   edgecolor='none', height=0.6)

    # Add percentage labels on each bar
    for bar, p in zip(bars, sorted_prob):
        if p > 0.02:    # Only label bars wide enough to read
            ax.text(
                min(p + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                f'{p:.1%}',
                va='center', ha='left',
                color='white', fontsize=9
            )

    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Confidence', color='#aaaaaa', fontsize=10)
    ax.set_title('Class Probabilities', color='white', fontsize=12,
                 fontweight='bold', pad=10)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    result = np.array(PILImage.open(buf).convert('RGB'))
    plt.close(fig)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction function — called by Gradio on every input
# ─────────────────────────────────────────────────────────────────────────────

def predict(pil_image):
    """
    Takes a PIL image uploaded by the user.
    Returns:
        gradcam_fig     — numpy image: wafer + heatmap + overlay
        confidence_fig  — numpy image: probability bar chart
        summary_text    — markdown string: prediction summary
    """
    if pil_image is None:
        empty = np.zeros((100, 100, 3), dtype=np.uint8)
        return empty, empty, 'Please upload a wafer map image.'

    # ── Preprocess ────────────────────────────────────────────────────────────
    wafer_array = image_to_wafer_array(pil_image)
    tensor      = wafer_array_to_tensor(wafer_array)

    # ── Run GradCAM (includes forward + backward pass) ────────────────────────
    heatmap, pred_idx, probs = gradcam.generate(tensor)

    # ── Build output figures ──────────────────────────────────────────────────
    results_fig    = make_results_figure(wafer_array, heatmap, pred_idx, probs)
    confidence_fig = make_confidence_chart(probs)

    # ── Build text summary ────────────────────────────────────────────────────
    pred_name = CLASS_NAMES[pred_idx]
    conf      = probs[pred_idx]

    # Human-readable description of each defect type
    descriptions = {
        'Center':    'Defects clustered at the center of the wafer.',
        'Donut':     'Ring of defects around the center with a clean interior.',
        'Edge-Loc':  'Localized defect cluster near the wafer edge.',
        'Edge-Ring': 'Continuous ring of defects around the outer edge.',
        'Loc':       'Small localized defect cluster anywhere on the wafer.',
        'Near-full': 'Almost the entire wafer surface is defective.',
        'Random':    'Defects scattered with no clear spatial pattern.',
        'Scratch':   'Linear scratch defect running across the wafer surface.',
        'none':      'No defect pattern detected — wafer appears clean.',
    }

    summary = f"""## Prediction: **{pred_name}**
**Confidence:** {conf:.1%}

**What this means:** {descriptions.get(pred_name, '')}

**Top 3 candidates:**
"""
    top3 = np.argsort(probs)[::-1][:3]
    for rank, idx in enumerate(top3, 1):
        summary += f'{rank}. {CLASS_NAMES[idx]} — {probs[idx]:.1%}\n'

    summary += """
---
*GradCAM heatmap shows which regions of the wafer influenced this prediction.*
*Red = high attention, Blue = low attention.*
"""

    return results_fig, confidence_fig, summary


# ─────────────────────────────────────────────────────────────────────────────
# Example samples — pulled from test set, one per defect class
# Users can click these instead of uploading their own image
# ─────────────────────────────────────────────────────────────────────────────

def build_example_images():
    """
    Render one test sample per defect class as a PNG in memory.
    Gradio's Examples component needs file paths, so we save to /tmp.
    """
    examples = []
    os.makedirs('/tmp/wafer_examples', exist_ok=True)

    for class_name in CLASS_NAMES:
        # Find the first test sample of this class
        class_rows = test_df[test_df['failureType'] == class_name]
        if len(class_rows) == 0:
            continue

        wafer = class_rows.iloc[0]['waferMap']
        wafer_display = cv2.resize(
            wafer.astype(np.float32),
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # Save as greyscale PNG (values 0/1/2 scaled to 0/128/255)
        save_path = f'/tmp/wafer_examples/{class_name}.png'
        img_uint8 = (wafer_display / 2.0 * 255).astype(np.uint8)
        cv2.imwrite(save_path, img_uint8)
        examples.append([save_path])

    return examples


examples = build_example_images()
print(f'Built {len(examples)} example images')


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

# Custom CSS — dark semiconductor-themed styling
css = """
body { background-color: #0f0f1a; }

.gradio-container {
    max-width: 1100px !important;
    font-family: 'IBM Plex Mono', monospace;
}

.gr-button-primary {
    background: linear-gradient(135deg, #00ff88, #00ccff) !important;
    color: #0f0f1a !important;
    font-weight: bold !important;
    border: none !important;
}

.gr-button-primary:hover {
    opacity: 0.85 !important;
}

h1 { color: #00ff88 !important; }
h3 { color: #00ccff !important; }

.gr-panel {
    background-color: #1a1a2e !important;
    border: 1px solid #333355 !important;
}
"""

# App description shown below the title
description = """
**Wafer Defect Classification** using ResNet18 trained on WM-811K (811,457 wafers, 9 defect classes).

Upload a wafer map image — or click one of the examples below — to get:
- Predicted defect class with confidence score
- GradCAM heatmap showing which region of the wafer drove the prediction
- Full class probability breakdown

**Model:** ResNet18 with Focal Loss | **Test Accuracy:** 96.67% | **Macro F1:** 0.86
"""

with gr.Blocks(css=css, title='Wafer Defect Classifier') as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown('# 🔬 Wafer Defect Classifier')
    gr.Markdown(description)

    # ── Main layout: input left, summary right ────────────────────────────────
    with gr.Row():

        # Left column — upload + examples
        with gr.Column(scale=1):
            gr.Markdown('### Upload Wafer Map')
            image_input = gr.Image(
                type='pil',
                label='Wafer Map Image',
                height=280
            )
            predict_btn = gr.Button('🔍 Classify', variant='primary')

            gr.Markdown('### Example Wafers (click to load)')
            gr.Examples(
                examples=examples,
                inputs=image_input,
                label='Test set samples — one per defect class'
            )

        # Right column — text summary
        with gr.Column(scale=1):
            gr.Markdown('### Prediction Summary')
            summary_output = gr.Markdown(
                value='*Upload a wafer map or click an example to begin.*'
            )

    # ── GradCAM visualization ─────────────────────────────────────────────────
    gr.Markdown('### GradCAM Explainability')
    gradcam_output = gr.Image(
        label='Wafer Map | GradCAM Heatmap | Overlay',
        height=300
    )

    # ── Confidence chart ──────────────────────────────────────────────────────
    gr.Markdown('### Confidence Breakdown')
    confidence_output = gr.Image(
        label='Class Probabilities',
        height=320
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown("""
---
**Classes:** Center · Donut · Edge-Loc · Edge-Ring · Loc · Near-full · Random · Scratch · none

**Dataset:** WM-811K (MIR Lab, NTU) | **Architecture:** ResNet18 | **GradCAM Layer:** layer3
""")

    # ── Wire up the button ─────────────────────────────────────────────────────
    predict_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[gradcam_output, confidence_output, summary_output]
    )

    # Also trigger on image upload (no need to click the button)
    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=[gradcam_output, confidence_output, summary_output]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    demo.launch(
    server_name='0.0.0.0',
    server_port=7860,
    share=False,
    show_error=True,
    allowed_paths=['/tmp/wafer_examples', str(BASE_DIR / 'data/processed')]
)