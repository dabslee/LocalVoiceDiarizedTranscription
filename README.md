# LocalTranscription

<img src="https://img.shields.io/badge/Python-3.14-blue?style=flat-square&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/faster--whisper-1.2.1-orange?style=flat-square"> <img src="https://img.shields.io/badge/pyannote.audio-4.0.7-darkgreen?style=flat-square"> <img src="https://img.shields.io/badge/sherpa--onnx-2.0.1-purple?style=flat-square">

LocalTranscription is a command-line tool that turns an English audio file into a timestamped transcript labeled by speaker. Everything runs locally: models are downloaded once, and your audio is never uploaded.

There are two versions:

- `transcribe.py` uses pyannote for speaker detection. It's more accurate but slower, and needs a free Hugging Face account.
- `transcribe_sherpa.py` uses sherpa-onnx. It's faster and needs no account.

## Installation

From the project folder in cmd:

**pyannote version**

```bat
python -m venv .venv-pyannote
.venv-pyannote\Scripts\python.exe -m pip install -r requirements.txt
.venv-pyannote\Scripts\hf.exe auth login
```

Before logging in, accept the terms on the [model page](https://huggingface.co/pyannote/speaker-diarization-community-1) and create a Read token under [Access Tokens](https://huggingface.co/settings/tokens).

**sherpa version**

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-sherpa.txt
```

## Usage

```bat
.venv-pyannote\Scripts\python.exe transcribe.py "C:\path\to\audio.mp3"
.venv\Scripts\python.exe transcribe_sherpa.py "C:\path\to\audio.mp3"
```

| Option | Description |
|---|---|
| `--num-speakers N` | Number of speakers, if known (improves accuracy) |
| `--output-dir DIR` | Where to save transcripts (default: next to the audio) |
| `--model NAME` | Whisper model (default: `turbo`) |

Run with `--help` for all options.

## Output

Each run writes `<name>.transcript.txt` and `<name>.transcript.json`. The JSON adds per-word timestamps.

```
[00:00:00 - 00:00:09] SPEAKER_1: Good morning. Thanks for joining the call today.

[00:00:10 - 00:00:17] SPEAKER_2: Happy to help. I looked at the numbers last night.
```

## Troubleshooting

- **`Couldn't access the diarization model`:** accept the model terms and run `hf auth login` again.
- **Wrong number of speakers:** pass `--num-speakers`.
