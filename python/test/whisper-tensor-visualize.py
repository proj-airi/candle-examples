import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import WhisperProcessor
from os import path

# --- Configuration ---
# This must match the model used to generate the data
MODEL_ID = "onnx-community/lite-whisper-large-v3-ONNX"
# MODEL_ID = "onnx-community/whisper-large-v3-turbo"

# Directory where the .npy files are stored
INPUT_DIR = Path("verification_data")

# Number of top tokens to show in the logits plot
TOP_K_LOGITS = 20

def plot_mel_spectrogram(features, output_path):
    """Generates and saves a plot of the mel spectrogram."""
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0) # Remove batch dimension

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(features, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
    fig.colorbar(im, ax=ax, format='%+2.0f dB')
    ax.set_title("Input Mel Spectrogram")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Mel Bins")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved spectrogram plot to {output_path}")
    return fig

def plot_encoder_output(hidden_states, output_path):
    """Generates and saves a plot of the encoder hidden states."""
    if hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
        hidden_states = hidden_states.squeeze(0) # Remove batch dimension

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(hidden_states, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
    fig.colorbar(im, ax=ax)
    ax.set_title("Encoder Hidden States")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Hidden Dimension")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved encoder output plot to {output_path}")
    return fig

def plot_logits(logits, tokenizer, output_path):
    """Generates and saves a bar chart of the top K logits."""
    # Logits shape is (batch, sequence, vocab_size). We want the logits for the *next* token.
    # In the first step, the input sequence has 3 tokens, so we take the logits from the last position.
    last_token_logits = logits[0, -1, :]

    # Find the top K tokens and their corresponding logit values
    top_k_indices = np.argsort(last_token_logits)[-TOP_K_LOGITS:]
    top_k_values = last_token_logits[top_k_indices]

    # Decode the token IDs to human-readable strings
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]

    # Find the token that was actually chosen (the one with the highest logit)
    chosen_token_index = np.argmax(top_k_values)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(np.arange(TOP_K_LOGITS), top_k_values, color='skyblue')

    # Highlight the chosen token in a different color
    bars[chosen_token_index].set_color('salmon')

    ax.set_yticks(np.arange(TOP_K_LOGITS))
    ax.set_yticklabels(top_k_tokens)
    ax.invert_yaxis() # Display the highest value at the top
    ax.set_xlabel("Logit Value")
    ax.set_title(f"Top {TOP_K_LOGITS} Predicted Tokens (First Decoder Step)")

    # Add the logit values as text on the bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 1 # Position label correctly for negative logits
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f' {width:.2f}',
                va='center', ha='left')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved logits plot to {output_path}")
    return fig


def main():
    """Loads data and generates all visualizations."""
    # Ensure the input directory exists
    if not INPUT_DIR.is_dir():
        print(f"Error: Directory '{INPUT_DIR}' not found. Please run the data generation script first.")
        return

    # --- Load Data ---
    try:
        input_features = np.load(INPUT_DIR / f"{path.basename(MODEL_ID)}_input_features.npy")
        encoder_output = np.load(INPUT_DIR / f"{path.basename(MODEL_ID)}_encoder_output.npy")
        step_0_logits = np.load(INPUT_DIR / f"{path.basename(MODEL_ID)}_step_0_logits.npy")
    except FileNotFoundError as e:
        print(f"Error: Missing data file - {e}. Please ensure all .npy files exist in '{INPUT_DIR}'.")
        return

    print("Successfully loaded all .npy files.")

    # --- Load Tokenizer ---
    # The tokenizer is needed to decode the logit indices into text
    print(f"Loading tokenizer for {MODEL_ID}...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer
    print("Tokenizer loaded.")

    # --- Generate Plots ---
    # Create a directory to save the plots
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    plot_mel_spectrogram(input_features, plots_dir / "mel_spectrogram.png")
    plot_encoder_output(encoder_output, plots_dir / "encoder_output.png")
    plot_logits(step_0_logits, tokenizer, plots_dir / "step_0_logits.png")

    # --- Show Plots ---
    # This will open interactive windows for each plot.
    print("\nDisplaying plots. Close the plot windows to exit the script.")
    plt.show()


if __name__ == "__main__":
    # You will need matplotlib: pip install matplotlib
    main()
