import os
import subprocess
import sys
import json
import wave
from faster_whisper import WhisperModel

def log(action, data):
    print(json.dumps({"action": action, "data": data}, ensure_ascii=False), flush=True)

def extract_audio(video_path, audio_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        "-vn",              # no video
        audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_audio_duration(audio_path):
    with wave.open(audio_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def main(video_id):
    video_path = f"../data/{video_id}/video.mp4"
    audio_path = f"../data/{video_id}/audio.wav"
    output_path = f"../data/{video_id}/transcription.json"
    if not os.path.exists(video_path):
        log("error", f"Arquivo não encontrado: {video_path}")
        return
    log("status", "Extraindo áudio...")
    extract_audio(video_path, audio_path)
    log("status", "Carregando modelo...")
    model_size = os.environ.get("WHISPER_MODEL", "base")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    log("status", "Calculando duração do áudio...")
    total_duration = get_audio_duration(audio_path)
    log("status", f"Duração total: {total_duration:.2f} segundos")
    log("status", "Transcrevendo áudio...")
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    results = {
        "language": info.language if info and info.language else "unknown",
        "text": "",
        "segments": []
    }
    last_progress = -1
    full_text_parts = []
    for i, segment in enumerate(segments):
        seg = {
            "id": i,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        }
        results["segments"].append(seg)
        full_text_parts.append(seg["text"])
        progress = round(min(1.0, max(0.0, segment.end / total_duration)), 2)
        if progress > last_progress:
            log("progress", progress)
            last_progress = progress
    results["text"] = " ".join(full_text_parts)
    log("status", "Salvando resultado...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log("done", {"msg": f"Transcrição salva em {output_path}"})
    if os.path.exists(audio_path):
        os.remove(audio_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        log("error", "Uso: python 02-transcribe.py <video_id>")
        sys.exit(1)
    main(sys.argv[1])