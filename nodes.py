"""ComfyUI nodes for audio annotation JSON generation."""

from __future__ import annotations

import json
import os
import pathlib
import re
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel


WORD_RE = re.compile(r"[A-Za-z0-9']+")
SECTION_LINE_RE = re.compile(r"^\[(.+?)\]$")
_MODEL_CACHE: Dict[Tuple[str, str, str, str], WhisperModel] = {}


def _get_input_directory() -> str:
    try:
        import folder_paths  # type: ignore

        return folder_paths.get_input_directory()
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "input"))


def _get_output_directory() -> str:
    try:
        import folder_paths  # type: ignore

        return folder_paths.get_output_directory()
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))


def _normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", word.lower())


def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text)


def _normalize_text(text: str) -> str:
    return " ".join(token for token in (_normalize_word(word) for word in _tokenize(text)) if token)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _round_time(value: float) -> float:
    return round(max(0.0, float(value)), 3)


def _resolve_audio_path(raw_path: str) -> str:
    candidate = pathlib.Path(str(raw_path).strip())
    if candidate.is_file():
        return str(candidate.resolve())

    input_dir = pathlib.Path(_get_input_directory())
    joined = (input_dir / candidate).resolve()
    if joined.is_file():
        return str(joined)

    raise FileNotFoundError(f"Audio file not found: {raw_path}")


def _run_ffmpeg_canonicalize(source_path: str) -> str:
    temp_dir = tempfile.mkdtemp(prefix="ktm_audio_annotation_")
    canonical_path = os.path.join(temp_dir, "canonical.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        source_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        canonical_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return canonical_path


def _resolve_device(device: str) -> Tuple[str, str]:
    requested = str(device or "auto").strip().lower()
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda":
        return "cuda", "float16"
    return ("cuda", "float16") if torch.cuda.is_available() else ("cpu", "int8")


def _load_whisper_model(model_name: str, device: str) -> WhisperModel:
    resolved_device, compute_type = _resolve_device(device)
    download_root = os.environ.get(
        "KTM_AUDIO_ANNOTATION_MODEL_CACHE",
        os.path.expanduser("~/.cache/ktm_audio_annotation"),
    )
    key = (model_name, resolved_device, compute_type, download_root)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = WhisperModel(
            model_name,
            device=resolved_device,
            compute_type=compute_type,
            download_root=download_root,
        )
        _MODEL_CACHE[key] = model
    return model


@dataclass
class ReferenceLine:
    text: str
    words: List[str]
    section_label: str
    section_label_raw: str


@dataclass
class ReferenceLyrics:
    sections: List[Dict[str, Any]]
    lines: List[ReferenceLine]
    words: List[Dict[str, Any]]


def _parse_reference_lyrics(text: str) -> ReferenceLyrics:
    sections: List[Dict[str, Any]] = []
    lines: List[ReferenceLine] = []
    words: List[Dict[str, Any]] = []

    current_label_raw = "verse"
    current_label = "verse"
    section_index = -1

    def ensure_section(label_raw: str, label: str) -> int:
        sections.append({"label_raw": label_raw, "label": label, "line_indices": []})
        return len(sections) - 1

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = SECTION_LINE_RE.match(line)
        if match:
            current_label_raw = match.group(1).strip()
            current_label = re.sub(r"\s+\d+$", "", current_label_raw.lower()).strip() or "section"
            section_index = ensure_section(current_label_raw, current_label)
            continue

        if section_index < 0:
            section_index = ensure_section(current_label_raw, current_label)

        line_words = _tokenize(line)
        line_index = len(lines)
        lines.append(
            ReferenceLine(
                text=line,
                words=line_words,
                section_label=current_label,
                section_label_raw=current_label_raw,
            )
        )
        sections[section_index]["line_indices"].append(line_index)

        for word in line_words:
            words.append(
                {
                    "word": word,
                    "normalized": _normalize_word(word),
                    "line_index": line_index,
                    "section_index": section_index,
                }
            )

    return ReferenceLyrics(sections=sections, lines=lines, words=words)


def _segment_words(text: str, start: float, end: float, probability: Optional[float]) -> List[Dict[str, Any]]:
    tokens = _tokenize(text)
    if not tokens:
        return []

    duration = max(_safe_float(end) - _safe_float(start), 0.05 * len(tokens))
    step = duration / max(len(tokens), 1)
    out: List[Dict[str, Any]] = []
    for index, token in enumerate(tokens):
        token_start = _safe_float(start) + (step * index)
        token_end = token_start + step
        out.append(
            {
                "word": token,
                "normalized": _normalize_word(token),
                "start": token_start,
                "end": token_end,
                "confidence": _safe_float(probability, 0.0),
            }
        )
    return out


def _transcribe_audio(canonical_wav_path: str, model_name: str, language: str, device: str) -> Dict[str, Any]:
    model = _load_whisper_model(model_name, device)
    segments_iter, info = model.transcribe(
        canonical_wav_path,
        language=(language or "").strip() or None,
        vad_filter=True,
        word_timestamps=True,
        beam_size=5,
        condition_on_previous_text=False,
        vad_parameters={"min_silence_duration_ms": 350},
    )

    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []

    for segment in segments_iter:
        segment_words: List[Dict[str, Any]] = []
        raw_words = list(segment.words or [])
        if raw_words:
            for raw_word in raw_words:
                segment_words.extend(
                    _segment_words(
                        getattr(raw_word, "word", "") or "",
                        getattr(raw_word, "start", None),
                        getattr(raw_word, "end", None),
                        getattr(raw_word, "probability", None),
                    )
                )
        else:
            segment_words.extend(
                _segment_words(
                    segment.text or "",
                    getattr(segment, "start", None),
                    getattr(segment, "end", None),
                    getattr(segment, "avg_logprob", None),
                )
            )

        if segment_words:
            words.extend(segment_words)

        segments.append(
            {
                "start": _safe_float(getattr(segment, "start", None)),
                "end": _safe_float(getattr(segment, "end", None)),
                "text": (segment.text or "").strip(),
                "confidence": _safe_float(getattr(segment, "avg_logprob", None)),
            }
        )

    detected_language = getattr(info, "language", None) or (language or "en")
    return {
        "language": detected_language,
        "segments": segments,
        "words": words,
    }


def _transcribe_audio_reference_mode(
    canonical_wav_path: str,
    model_name: str,
    language: str,
    device: str,
) -> Dict[str, Any]:
    model = _load_whisper_model(model_name, device)
    segments_iter, info = model.transcribe(
        canonical_wav_path,
        language=(language or "").strip() or None,
        vad_filter=False,
        word_timestamps=True,
        beam_size=5,
        condition_on_previous_text=False,
        chunk_length=20,
    )

    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []

    for segment in segments_iter:
        segment_words: List[Dict[str, Any]] = []
        raw_words = list(segment.words or [])
        if raw_words:
            for raw_word in raw_words:
                segment_words.extend(
                    _segment_words(
                        getattr(raw_word, "word", "") or "",
                        getattr(raw_word, "start", None),
                        getattr(raw_word, "end", None),
                        getattr(raw_word, "probability", None),
                    )
                )
        else:
            segment_words.extend(
                _segment_words(
                    segment.text or "",
                    getattr(segment, "start", None),
                    getattr(segment, "end", None),
                    getattr(segment, "avg_logprob", None),
                )
            )

        if segment_words:
            words.extend(segment_words)

        segments.append(
            {
                "start": _safe_float(getattr(segment, "start", None)),
                "end": _safe_float(getattr(segment, "end", None)),
                "text": (segment.text or "").strip(),
                "confidence": _safe_float(getattr(segment, "avg_logprob", None)),
            }
        )

    detected_language = getattr(info, "language", None) or (language or "en")
    return {
        "language": detected_language,
        "segments": segments,
        "words": words,
    }


def _build_reference_alignment(
    reference: ReferenceLyrics,
    transcript: Dict[str, Any],
    duration_seconds: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    ref_words = reference.words
    transcript_words = transcript["words"]
    ref_norms = [word["normalized"] for word in ref_words if word["normalized"]]
    transcript_norms = [word["normalized"] for word in transcript_words if word["normalized"]]

    if not ref_words:
        return [], [], [], {"alignment_coverage": 0.0, "text_similarity": 0.0, "matched_word_count": 0}

    ref_index_lookup = [index for index, word in enumerate(ref_words) if word["normalized"]]
    transcript_index_lookup = [index for index, word in enumerate(transcript_words) if word["normalized"]]

    matches: Dict[int, int] = {}
    matcher = SequenceMatcher(a=ref_norms, b=transcript_norms, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                matches[ref_index_lookup[i1 + offset]] = transcript_index_lookup[j1 + offset]
        elif tag == "replace":
            pairs = min(i2 - i1, j2 - j1)
            for offset in range(pairs):
                matches[ref_index_lookup[i1 + offset]] = transcript_index_lookup[j1 + offset]

    transcript_durations = [
        max(0.05, _safe_float(word.get("end")) - _safe_float(word.get("start")))
        for word in transcript_words
        if word.get("end") is not None and word.get("start") is not None
    ]
    default_word_duration = statistics.median(transcript_durations) if transcript_durations else max(0.18, duration_seconds / max(len(ref_words), 1))

    aligned_words: List[Optional[Dict[str, Any]]] = [None] * len(ref_words)
    for ref_index, transcript_index in matches.items():
        transcript_word = transcript_words[transcript_index]
        aligned_words[ref_index] = {
            "word": ref_words[ref_index]["word"],
            "start": _safe_float(transcript_word.get("start")),
            "end": _safe_float(transcript_word.get("end")),
            "confidence": _safe_float(transcript_word.get("confidence"), 0.0),
            "normalized": ref_words[ref_index]["normalized"],
            "line_index": ref_words[ref_index]["line_index"],
            "section_index": ref_words[ref_index]["section_index"],
        }

    matched_indices = sorted(matches.keys())
    if not matched_indices:
        evenly_spaced: List[Dict[str, Any]] = []
        cursor = 0.0
        step = max(duration_seconds / max(len(ref_words), 1), default_word_duration)
        for ref_word in ref_words:
            start = cursor
            end = min(duration_seconds, start + step)
            evenly_spaced.append(
                {
                    "word": ref_word["word"],
                    "start": _round_time(start),
                    "end": _round_time(end),
                    "confidence": 0.0,
                    "normalized": ref_word["normalized"],
                    "line_index": ref_word["line_index"],
                    "section_index": ref_word["section_index"],
                }
            )
            cursor = end
        return _group_reference_outputs(reference, evenly_spaced, 0, 0.0)

    first_matched = matched_indices[0]
    first_start = _safe_float(aligned_words[first_matched]["start"])
    if first_matched > 0:
        total_span = max(first_start, default_word_duration * first_matched)
        step = total_span / first_matched
        cursor = 0.0
        for index in range(first_matched):
            aligned_words[index] = {
                "word": ref_words[index]["word"],
                "start": cursor,
                "end": cursor + step,
                "confidence": 0.0,
                "normalized": ref_words[index]["normalized"],
                "line_index": ref_words[index]["line_index"],
                "section_index": ref_words[index]["section_index"],
            }
            cursor += step

    for prev_index, next_index in zip(matched_indices, matched_indices[1:]):
        gap_count = next_index - prev_index - 1
        if gap_count <= 0:
            continue
        prev_end = _safe_float(aligned_words[prev_index]["end"])
        next_start = _safe_float(aligned_words[next_index]["start"])
        total_span = max(next_start - prev_end, default_word_duration * gap_count)
        step = total_span / gap_count
        cursor = prev_end
        for offset in range(1, gap_count + 1):
            ref_index = prev_index + offset
            aligned_words[ref_index] = {
                "word": ref_words[ref_index]["word"],
                "start": cursor,
                "end": cursor + step,
                "confidence": 0.0,
                "normalized": ref_words[ref_index]["normalized"],
                "line_index": ref_words[ref_index]["line_index"],
                "section_index": ref_words[ref_index]["section_index"],
            }
            cursor += step

    last_matched = matched_indices[-1]
    last_end = _safe_float(aligned_words[last_matched]["end"])
    trailing_count = len(ref_words) - last_matched - 1
    if trailing_count > 0:
        total_span = max(duration_seconds - last_end, default_word_duration * trailing_count)
        step = total_span / trailing_count
        cursor = last_end
        for index in range(last_matched + 1, len(ref_words)):
            aligned_words[index] = {
                "word": ref_words[index]["word"],
                "start": cursor,
                "end": cursor + step,
                "confidence": 0.0,
                "normalized": ref_words[index]["normalized"],
                "line_index": ref_words[index]["line_index"],
                "section_index": ref_words[index]["section_index"],
            }
            cursor += step

    materialized = [
        {
            "word": entry["word"],
            "start": _round_time(entry["start"]),
            "end": _round_time(entry["end"]),
            "confidence": round(_safe_float(entry["confidence"]), 3),
            "line_index": entry["line_index"],
            "section_index": entry["section_index"],
        }
        for entry in aligned_words
        if entry is not None
    ]

    reference_text = " ".join(_normalize_text(line.text) for line in reference.lines)
    transcript_text = " ".join(_normalize_text(segment.get("text", "")) for segment in transcript["segments"])
    line_similarity = SequenceMatcher(None, reference_text, transcript_text, autojunk=False).ratio()
    return _group_reference_outputs(reference, materialized, len(matches), line_similarity)


def _group_reference_outputs(
    reference: ReferenceLyrics,
    aligned_words: List[Dict[str, Any]],
    matched_word_count: int,
    text_similarity: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    line_outputs: List[Dict[str, Any]] = []
    line_output_by_index: Dict[int, Dict[str, Any]] = {}
    for line_index, line in enumerate(reference.lines):
        line_words = [word for word in aligned_words if word["line_index"] == line_index]
        if not line_words:
            continue
        line_output = {
            "start": _round_time(line_words[0]["start"]),
            "end": _round_time(line_words[-1]["end"]),
            "text": line.text,
            "confidence": round(sum(word["confidence"] for word in line_words) / max(len(line_words), 1), 3),
            "section_label": line.section_label,
            "section_label_raw": line.section_label_raw,
        }
        line_outputs.append(line_output)
        line_output_by_index[line_index] = line_output

    section_outputs: List[Dict[str, Any]] = []
    for section in reference.sections:
        section_lines = [line_output_by_index[index] for index in section["line_indices"] if index in line_output_by_index]
        if not section_lines:
            continue
        section_outputs.append(
            {
                "start": _round_time(section_lines[0]["start"]),
                "end": _round_time(section_lines[-1]["end"]),
                "label": section["label"],
                "label_raw": section["label_raw"],
            }
        )

    qa = {
        "alignment_coverage": round(matched_word_count / max(len(reference.words), 1), 3),
        "text_similarity": round(text_similarity, 3),
        "matched_word_count": matched_word_count,
    }
    return line_outputs, aligned_words, section_outputs, qa


def _build_transcript_outputs(
    transcript: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    words = [
        {
            "word": word["word"],
            "start": _round_time(word["start"]),
            "end": _round_time(word["end"]),
            "confidence": round(_safe_float(word.get("confidence"), 0.0), 3),
        }
        for word in transcript["words"]
    ]
    segments = [
        {
            "start": _round_time(segment["start"]),
            "end": _round_time(segment["end"]),
            "text": segment["text"],
            "confidence": round(_safe_float(segment.get("confidence"), 0.0), 3),
        }
        for segment in transcript["segments"]
        if segment["text"]
    ]
    sections: List[Dict[str, Any]] = []
    qa = {
        "alignment_coverage": 1.0 if words else 0.0,
        "text_similarity": 1.0 if words else 0.0,
        "matched_word_count": len(words),
    }
    return segments, words, sections, qa


def _analyze_beats(canonical_wav_path: str) -> Dict[str, Any]:
    y, sr = librosa.load(canonical_wav_path, sr=22050, mono=True)
    duration_seconds = _safe_float(librosa.get_duration(y=y, sr=sr))
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beats = [_round_time(value) for value in beat_times.tolist()]
    downbeats = beats[::4] if beats else []

    tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size > 0 else 0.0
    return {
        "duration_seconds": round(duration_seconds, 3),
        "bpm": round(tempo_value, 2),
        "beats": beats,
        "downbeats": downbeats,
    }


def _build_annotation(
    source_path: str,
    canonical_wav_path: str,
    reference_lyrics_text: str,
    language: str,
    transcription_model: str,
    device: str,
) -> Dict[str, Any]:
    structure = _analyze_beats(canonical_wav_path)
    transcript = (
        _transcribe_audio_reference_mode(canonical_wav_path, transcription_model, language, device)
        if reference_lyrics_text.strip()
        else _transcribe_audio(canonical_wav_path, transcription_model, language, device)
    )

    if reference_lyrics_text.strip():
        reference = _parse_reference_lyrics(reference_lyrics_text)
        lyrics_segments, lyrics_words, lyric_sections, qa_extra = _build_reference_alignment(
            reference,
            transcript,
            structure["duration_seconds"],
        )
        lyrics_mode = "provided"
        section_outputs = lyric_sections
        lyrics_source_text = "\n".join(line.text for line in reference.lines)
    else:
        lyrics_segments, lyrics_words, lyric_sections, qa_extra = _build_transcript_outputs(transcript)
        lyrics_mode = "asr"
        section_outputs = lyric_sections
        lyrics_source_text = "\n".join(segment["text"] for segment in lyrics_segments)

    if not section_outputs and lyrics_segments:
        section_outputs = [
            {
                "start": lyrics_segments[0]["start"],
                "end": lyrics_segments[-1]["end"],
                "label": "verse",
                "label_raw": "Verse",
            }
        ]

    vocals_start = lyrics_words[0]["start"] if lyrics_words else 0.0
    vocals_end = lyrics_words[-1]["end"] if lyrics_words else structure["duration_seconds"]

    qa = {
        "asr_model": transcription_model,
        "alignment_model": "reference_anchor_v1" if reference_lyrics_text.strip() else "none",
        "reference_lyrics_used": bool(reference_lyrics_text.strip()),
        "separation_used": False,
        "needs_review": qa_extra["alignment_coverage"] < 0.2 or qa_extra["text_similarity"] < 0.35,
        "alignment_coverage": qa_extra["alignment_coverage"],
        "text_similarity": qa_extra["text_similarity"],
        "matched_word_count": qa_extra["matched_word_count"],
    }

    return {
        "source": {
            "audio_path": source_path,
            "canonical_wav": canonical_wav_path,
        },
        "lyrics": {
            "mode": lyrics_mode,
            "language": transcript["language"],
            "text": lyrics_source_text,
            "segments": lyrics_segments,
            "words": lyrics_words,
        },
        "structure": {
            "duration_seconds": structure["duration_seconds"],
            "bpm": structure["bpm"],
            "beats": structure["beats"],
            "downbeats": structure["downbeats"],
            "sections": section_outputs,
        },
        "instruments": [
            {
                "label": "vocals",
                "start": _round_time(vocals_start),
                "end": _round_time(vocals_end),
                "confidence": 0.99 if lyrics_words else 0.0,
                "source": "lyrics_alignment" if reference_lyrics_text.strip() else "asr",
            }
        ],
        "qa": qa,
    }


def _next_output_path(filename_prefix: str, extension: str) -> Tuple[str, str, str]:
    clean_prefix = str(filename_prefix or "annotations/ComfyUI").strip().replace("\\", "/").lstrip("/")
    if not clean_prefix:
        clean_prefix = "annotations/ComfyUI"

    subfolder = os.path.dirname(clean_prefix)
    stem = os.path.basename(clean_prefix) or "ComfyUI"
    output_dir = _get_output_directory()
    target_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(target_dir, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(stem)}_(\d{{5}})_\.{re.escape(extension)}$")
    counter = 1
    for name in os.listdir(target_dir):
        match = pattern.match(name)
        if match:
            counter = max(counter, int(match.group(1)) + 1)

    filename = f"{stem}_{counter:05d}_.{extension}"
    relative_path = f"{subfolder}/{filename}" if subfolder else filename
    absolute_path = os.path.join(target_dir, filename)
    return absolute_path, relative_path, subfolder


class KTMAudioReferenceAnnotation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "", "multiline": False}),
                "reference_lyrics": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "language": ("STRING", {"default": "en", "multiline": False}),
                "transcription_model": (["distil-large-v3", "large-v3"], {"default": "distil-large-v3"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("annotation_json", "summary")
    FUNCTION = "annotate"
    CATEGORY = "Killatamata/Audio"

    def annotate(
        self,
        audio_path: str,
        reference_lyrics: str,
        language: str,
        transcription_model: str,
        device: str,
    ):
        resolved_audio_path = _resolve_audio_path(audio_path)
        canonical_wav_path = _run_ffmpeg_canonicalize(resolved_audio_path)
        annotation = _build_annotation(
            resolved_audio_path,
            canonical_wav_path,
            reference_lyrics,
            language,
            transcription_model,
            device,
        )
        annotation_json = json.dumps(annotation, indent=2, ensure_ascii=True)
        summary = (
            f"duration={annotation['structure']['duration_seconds']}s "
            f"words={len(annotation['lyrics']['words'])} "
            f"coverage={annotation['qa']['alignment_coverage']}"
        )
        return (annotation_json, summary)


class KTMSaveAudioAnnotationJson:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "annotation_json": ("STRING", {"default": "{}", "multiline": True, "dynamicPrompts": False}),
                "filename_prefix": ("STRING", {"default": "annotations/ComfyUI", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "Killatamata/Audio"
    OUTPUT_NODE = True

    def save(self, annotation_json: str, filename_prefix: str):
        parsed = json.loads(annotation_json)
        absolute_path, relative_path, subfolder = _next_output_path(filename_prefix, "json")
        with open(absolute_path, "w", encoding="utf-8") as handle:
            json.dump(parsed, handle, indent=2, ensure_ascii=True)
            handle.write("\n")

        preview = {
            "filename": os.path.basename(absolute_path),
            "subfolder": subfolder,
            "type": "output",
            "fullpath": absolute_path,
        }
        return {"ui": {"json_files": [preview]}, "result": (relative_path,)}


NODE_CLASS_MAPPINGS = {
    "KTMAudioReferenceAnnotation": KTMAudioReferenceAnnotation,
    "KTMSaveAudioAnnotationJson": KTMSaveAudioAnnotationJson,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KTMAudioReferenceAnnotation": "KTM Audio Reference Annotation",
    "KTMSaveAudioAnnotationJson": "KTM Save Audio Annotation JSON",
}
