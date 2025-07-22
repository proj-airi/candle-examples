import librosa
import torch
import time
from transformers import AutoProcessor, AutoModel

device = "mps"
dtype = torch.float16

# load the compressed Whisper model
model = AutoModel.from_pretrained("efficient-speech/lite-whisper-large-v3-turbo-acc", trust_remote_code=True)
# model = AutoModel.from_pretrained("efficient-speech/lite-whisper-large-v3-turbo", trust_remote_code=True)
model.to(dtype).to(device)

# we use the same processor as the original model
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

# set the path to your audio file
path = "./.temp/recording.wav"
audio, _ = librosa.load(path, sr=16000)

input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
input_features = input_features.to(dtype).to(device)

# Run transcription 3 times and measure timing
times = []
transcriptions = []

for i in range(3):
    print(f"Running transcription {i+1}/3...")

    start_time = time.time()
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]
    end_time = time.time()

    elapsed_time = end_time - start_time
    times.append(elapsed_time)
    transcriptions.append(transcription)

    print(f"Transcription {i+1}: {transcription}")
    print(f"Time taken: {elapsed_time:.3f} seconds")
    print()

# Calculate and print average time
average_time = sum(times) / len(times)
print(f"Individual times: {[f'{t:.3f}s' for t in times]}")
print(f"Average time: {average_time:.3f} seconds")
