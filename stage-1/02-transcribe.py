import sys
import os
import json
import whisper

def print_json(action, data):
    print(json.dumps({"action": action, "data": data}), flush=True)

def main():
    try:
        if len(sys.argv) < 2:
            print_json("error", {"code": 1, "msg": "Parâmetro video_id não fornecido"})
            sys.exit(1)
        video_id = sys.argv[1]
        video_path = os.path.join("..", "data", video_id, "video.mp4")
        output_path = os.path.join("..", "data", video_id, "transcription.json")
        if not os.path.isfile(video_path):
            print_json("error", {"code": 2, "msg": f"Arquivo não encontrado: {video_path}"})
            sys.exit(1)
        model = whisper.load_model("base", device="cuda")
        print_json("progress", 0.0)
        result = model.transcribe(video_path, verbose=False)
        print_json("progress", 1.0)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print_json("done", {"msg": f"Transcrição salva em {output_path}"})
    except Exception as e:
        print_json("error", {"code": 99, "msg": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()