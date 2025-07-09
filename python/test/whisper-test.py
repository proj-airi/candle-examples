import librosa
import numpy as np
import onnxruntime as ort
from transformers import WhisperProcessor
from huggingface_hub import hf_hub_download
import json
from pathlib import Path
from os import path

# --- Configuration ---
# Use the same model you are using in your Rust code
MODEL_ID = "onnx-community/lite-whisper-large-v3-ONNX"
# MODEL_ID = "onnx-community/whisper-large-v3-turbo"
REVISION = "main"

# Create a dummy WAV file for testing if you don't have one.
# This generates a 5-second 1kHz sine wave.
SAMPLE_RATE = 16000
DURATION = 5

AUDIO_FILE = "./.temp/recording.wav"

# Directory to save the verification tensors
OUTPUT_DIR = Path("verification_data")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    """
    Runs the Whisper ONNX model step-by-step and saves intermediate outputs
    for verification against another implementation.
    """
    print(f"Using model: {MODEL_ID}")

    # 1. Load Processor, Tokenizer, and Model Config
    # ------------------------------------------------
    # The processor handles audio resampling and conversion to a mel spectrogram.
    # Using the official processor from transformers ensures our input is correct.
    processor = WhisperProcessor.from_pretrained(MODEL_ID, local_files_only=True)

    # Download config to get model parameters
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json", revision=REVISION, local_files_only=True)
    with open(config_path) as f:
        config = json.load(f)

    # These parameters are crucial for understanding the model's architecture
    num_decoder_layers     = config["decoder_layers"]
    num_attention_heads    = config["decoder_attention_heads"]
    hidden_size            = config["d_model"]
    head_dim               = hidden_size // num_attention_heads
    eos_token_id           = config["eos_token_id"]
    decoder_start_token_id = config["decoder_start_token_id"]

    print(f"Model config loaded: {num_decoder_layers} layers, {num_attention_heads} heads, {hidden_size} hidden size.")

    # 2. Load ONNX Models
    # -------------------
    # Download the ONNX model files from the Hugging Face Hub
    encoder_path = hf_hub_download(repo_id=MODEL_ID, filename="onnx/encoder_model.onnx", revision=REVISION, local_files_only=True)
    decoder_path = hf_hub_download(repo_id=MODEL_ID, filename="onnx/decoder_model.onnx", revision=REVISION, local_files_only=True)

    print(f"Encoder path: {encoder_path}")
    print(f"Decoder path: {decoder_path}")

    # Create ONNX Runtime inference sessions
    # Use the CPU provider for simplicity and portability. You can add "CUDAExecutionProvider"
    # to the list if you have a compatible GPU and onnxruntime-gpu installed.
    # providers       = ["CPUExecutionProvider"]
    providers       = ["CoreMLExecutionProvider"]
    encoder_session = ort.InferenceSession(encoder_path, providers=providers)
    decoder_session = ort.InferenceSession(decoder_path, providers=providers)
    print("ONNX encoder and decoder sessions created.")


    # 3. Process Audio Input
    # ----------------------
    # Load audio file using librosa, which ensures it's a float array and at the correct sample rate.
    audio_data, _ = librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)

    # Use the processor to create the input_features (mel spectrogram)
    # This is the "golden" input to the model. Your Rust STFT implementation should produce this.
    input_features = processor(audio_data, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features

    print(f"Input features created with shape: {input_features.shape} and type: {input_features.dtype}")

    # --- SAVE FOR VERIFICATION 1 ---
    feature_path = OUTPUT_DIR / f"{path.basename(MODEL_ID)}_input_features.npy"
    np.save(feature_path, input_features)
    print(f"Saved input features to {feature_path}")


    # 4. Run Encoder
    # --------------
    # The encoder processes the audio features into a sequence of hidden states.
    encoder_inputs        = {"input_features": input_features}
    encoder_outputs       = encoder_session.run(None, encoder_inputs)
    encoder_hidden_states = encoder_outputs[0] # The name is 'last_hidden_state'

    print(f"Encoder hidden states created with shape: {encoder_hidden_states.shape}")

    # --- SAVE FOR VERIFICATION 2 ---
    encoder_output_path = OUTPUT_DIR / f"{path.basename(MODEL_ID)}_encoder_output.npy"
    np.save(encoder_output_path, encoder_hidden_states)
    print(f"Saved encoder hidden states to {encoder_output_path}")


    # 5. Autoregressive Generation (Decoder Loop)
    # -------------------------------------------
    print("\nStarting autoregressive generation...")

    # Define initial tokens for English transcription
    # <|startoftranscript|> <|en|> <|transcribe|>
    # You can find token IDs in the tokenizer.json or added_tokens.json files
    language_token_id = processor.tokenizer.convert_tokens_to_ids("<|en|>")
    task_token_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")

    decoder_input_ids = np.array([[decoder_start_token_id, language_token_id, task_token_id]], dtype=np.int64)

    generated_tokens = []

    # Initialize the KV cache. It's a list of numpy arrays, one for key and one for value per layer.
    # The shape is [batch_size, num_heads, sequence_length, head_dim].
    # For the first step, sequence_length is 0.
    past_key_values = []
    for i in range(num_decoder_layers):
        # Key and Value have the same shape
        shape = (1, num_attention_heads, 0, head_dim)
        past_key_values.append(np.zeros(shape, dtype=np.float32)) # past_key
        past_key_values.append(np.zeros(shape, dtype=np.float32)) # past_value

    max_new_tokens = 128
    for step in range(max_new_tokens):
        # Prepare inputs for the decoder
        decoder_inputs = {
            "encoder_hidden_states": encoder_hidden_states,
            "input_ids": decoder_input_ids,
        }

        # Run the decoder for one step
        decoder_outputs = decoder_session.run(None, decoder_inputs)

        # The outputs are [logits, present.0.key, present.0.value, present.1.key, ...]
        logits = decoder_outputs[0]
        present_key_values = decoder_outputs[1:]

        # Greedy search: get the token with the highest probability
        next_token_logits = logits[:, -1, :] # Shape: [batch_size, vocab_size]
        next_token = np.argmax(next_token_logits, axis=-1)[0]

        if step == 0:
            print("--- First Decoder Step Verification ---")
            print(f"Logits shape: {logits.shape}")
            print(f"First present key shape: {present_key_values[0].shape}")

            # --- SAVE FOR VERIFICATION 3 ---
            logits_path = OUTPUT_DIR / f"{path.basename(MODEL_ID)}_step_0_logits.npy"
            np.save(logits_path, logits)
            print(f"Saved first step logits to {logits_path}")

        # Check for End-Of-Sequence token
        if next_token == eos_token_id:
            print("End-of-sequence token generated. Stopping.")
            break

        generated_tokens.append(next_token)

        # Prepare for the next iteration
        # The new input is just the token we just generated
        decoder_input_ids = np.array([[next_token]], dtype=np.int64)
        # The new KV cache is the 'present' output from this step
        past_key_values = present_key_values

    # 6. Decode and Print Transcript
    # ------------------------------
    print("\n--- Final Transcript ---")
    transcript = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(transcript)


if __name__ == "__main__":
    # You might need to install soundfile: pip install soundfile
    main()
