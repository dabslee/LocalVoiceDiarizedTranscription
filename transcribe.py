#!/usr/bin/env python3
"""Transcribe an English audio file with speaker labels, entirely on this machine.

Models are downloaded once into ./models. Audio is only ever decoded and
processed locally; it is never uploaded anywhere.

Requires a free Hugging Face login to download the diarization model. Accept the
terms at https://huggingface.co/pyannote/speaker-diarization-community-1 and then
run `hf auth login`.

Usage:
    python transcribe.py "path/to/audio.mp3"
    python transcribe.py "path/to/audio.m4a" --num-speakers 2
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# pyannote reports usage metrics to its servers by default and reads this on import.
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
from faster_whisper import WhisperModel, decode_audio
from huggingface_hub.errors import LocalEntryNotFoundError
from pyannote.audio import Pipeline
from tqdm import tqdm

SAMPLE_RATE = 16000
DOWNLOAD_ATTEMPTS = 5
MODELS_DIR = Path(__file__).resolve().parent / "models"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def log(message):
    """Print a timestamped message without disturbing any active progress bar."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tqdm.write(f"[{stamp}] {message}", file=sys.stderr)


def format_time(seconds):
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def with_retries(load):
    """Call load, retrying when Hugging Face can't be reached and the model isn't cached yet."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return load()
        except LocalEntryNotFoundError:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            log(f"Couldn't reach Hugging Face (attempt {attempt}), retrying in 10s")
            time.sleep(10)


def load_diarizer(args):
    log(f"Loading diarization model {DIARIZATION_MODEL} (downloads on first use)")
    pipeline = with_retries(
        lambda: Pipeline.from_pretrained(DIARIZATION_MODEL, cache_dir=MODELS_DIR)
    )
    if pipeline is None:
        sys.exit(
            "Couldn't access the diarization model. Accept its terms at "
            f"https://huggingface.co/{DIARIZATION_MODEL} and run `hf auth login`."
        )
    return pipeline.to(torch.device(args.device))


def load_whisper(args):
    log(f"Loading Whisper model '{args.model}' (downloads on first use)")
    return with_retries(
        lambda: WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.threads,
            download_root=str(MODELS_DIR),
        )
    )


class DiarizationProgress:
    """Show a tqdm bar for each pyannote step that reports progress."""

    def __init__(self):
        self.step = None
        self.bar = None

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if step_name != self.step:
            self.close()
            self.step = step_name
            log(f"Diarization step: {step_name}")
            if total:
                self.bar = tqdm(total=total, unit="batch", desc=f"Diarizing ({step_name})")
        if self.bar and completed is not None:
            self.bar.update(completed - self.bar.n)

    def close(self):
        if self.bar:
            self.bar.close()
            self.bar = None


def diarize(pipeline, audio, args):
    """Return (start, end, speaker) turns sorted by start time."""
    # Passing decoded samples keeps pyannote from needing its own audio decoder.
    waveform = torch.from_numpy(audio).unsqueeze(0)
    progress = DiarizationProgress()
    try:
        output = pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            num_speakers=args.num_speakers,
            hook=progress,
        )
    finally:
        progress.close()

    # The exclusive output has one speaker at a time, which is what word labeling needs.
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True)
    ]
    return sorted(turns)


def transcribe(model, audio, args):
    """Return (words, info), where each word is a {start, end, text} dict."""
    segments, info = model.transcribe(
        audio,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=True,
        word_timestamps=True,
    )

    words = []
    next_milestone = 10
    total = int(info.duration)
    with tqdm(total=total, unit="s", desc="Transcribing") as bar:
        for segment in segments:
            if segment.words:
                words.extend(
                    {"start": w.start, "end": w.end, "text": w.word}
                    for w in segment.words
                )
            else:
                words.append(
                    {"start": segment.start, "end": segment.end, "text": segment.text}
                )

            position = min(int(segment.end), total)
            bar.update(max(position - bar.n, 0))
            percent = 100 * position / total if total else 100
            if percent >= next_milestone:
                log(f"Transcribed {format_time(position)} of {format_time(total)}")
                next_milestone = int(percent // 10 + 1) * 10
        bar.update(total - bar.n)

    return words, info


def assign_speakers(words, turns):
    """Label each word with the speaker it overlaps most, numbered by first appearance."""
    if not turns:
        return [{**word, "speaker": "UNKNOWN"} for word in words]

    starts = np.array([t[0] for t in turns])
    ends = np.array([t[1] for t in turns])
    names = {}
    labeled = []
    for word in words:
        # Negative overlap is the gap to a turn, so a word that falls between
        # turns goes to the nearest one.
        overlap = np.minimum(ends, word["end"]) - np.maximum(starts, word["start"])
        speaker = turns[int(np.argmax(overlap))][2]
        name = names.setdefault(speaker, f"SPEAKER_{len(names) + 1}")
        labeled.append({**word, "speaker": name})
    return labeled


def group_utterances(words):
    """Merge consecutive words from the same speaker into one utterance."""
    utterances = []
    for word in words:
        if utterances and utterances[-1]["speaker"] == word["speaker"]:
            current = utterances[-1]
            current["end"] = word["end"]
            current["text"] += word["text"]
            current["words"].append(word)
        else:
            utterances.append(
                {
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["text"],
                    "words": [word],
                }
            )
    for utterance in utterances:
        utterance["text"] = utterance["text"].strip()
    return utterances


def write_outputs(audio_path, utterances, info, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{audio_path.stem}.transcript.txt"
    json_path = output_dir / f"{audio_path.stem}.transcript.json"

    lines = [
        f"[{format_time(u['start'])} - {format_time(u['end'])}] {u['speaker']}: {u['text']}"
        for u in utterances
    ]
    txt_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    data = {
        "source": audio_path.name,
        "language": info.language,
        "duration": info.duration,
        "utterances": utterances,
    }
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return txt_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file locally with speaker diarization."
    )
    parser.add_argument("audio", type=Path, help="path to the audio file")
    parser.add_argument(
        "--model", default="turbo", help="Whisper model, e.g. small, medium.en, large-v3 (default: turbo)"
    )
    parser.add_argument("--language", default="en", help="language code (default: en)")
    parser.add_argument(
        "--num-speakers", type=int, help="exact speaker count, if known (improves accuracy)"
    )
    parser.add_argument("--output-dir", type=Path, help="default: next to the audio file")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 4, 10))
    return parser.parse_args()


def main():
    # Windows consoles default to a legacy code page that can't print every filename.
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if not args.audio.is_file():
        sys.exit(f"Audio file not found: {args.audio}")

    started = time.monotonic()
    torch.set_num_threads(args.threads)

    # Load models before the slow work so a failed download stops the run early.
    diarizer = load_diarizer(args)
    model = load_whisper(args)

    log(f"Decoding {args.audio.name}")
    audio = decode_audio(str(args.audio), sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    log(f"Audio length: {format_time(duration)}")

    processing_started = time.monotonic()
    log("Starting speaker diarization")
    turns = diarize(diarizer, audio, args)
    log(f"Found {len({t[2] for t in turns})} speaker(s) across {len(turns)} turns")

    log("Starting transcription")
    words, info = transcribe(model, audio, args)
    log(f"Transcribed {len(words)} words")

    utterances = group_utterances(assign_speakers(words, turns))
    output_dir = args.output_dir or args.audio.parent
    txt_path, json_path = write_outputs(args.audio, utterances, info, output_dir)

    log(f"Wrote {txt_path}")
    log(f"Wrote {json_path}")
    processing = time.monotonic() - processing_started
    if duration:
        log(f"Processing speed: {processing / duration:.2f} min per min of audio")
    log(f"Finished in {format_time(time.monotonic() - started)}")


if __name__ == "__main__":
    main()
