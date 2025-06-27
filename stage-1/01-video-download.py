import sys
import json
import os
import yt_dlp
import time

def log(action, data):
    print(json.dumps({"action": action, "data": data}), flush=True)

def progress_hook(d):
    if d['status'] == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate') or 1
        downloaded = d.get('downloaded_bytes', 0)
        progress = downloaded / total if total else 0
        log("progress", progress)
    elif d['status'] == 'finished':
        log("progress", 1.0)

def main():
    if len(sys.argv) < 2:
        log("error", {"code": 1, "msg": "Parâmetro video_id obrigatório"})
        sys.exit(1)
    video_id_input = sys.argv[1]
    ydl_opts_info = {'quiet': True, 'no_warnings': True, 'skip_download': True}
    info = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(video_id_input, download=False)
            video_id = info.get('id')
            if not video_id:
                log("error", {"code": 2, "msg": "Não foi possível obter info para o video_id fornecido"})
                sys.exit(1)
    except Exception as e:
        log("error", {"code": 3, "msg": f"Erro ao obter info do vídeo: {str(e)}"})
        sys.exit(1)
    output_dir = os.path.join("..", "data", video_id)
    if os.path.exists(output_dir):
        log("error", {"code": 4, "msg": f"Pasta {output_dir} já existe", "video_id": video_id})
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    upload_date_str = info.get('upload_date')
    formatted_date = None
    if upload_date_str:
        formatted_date = f"{upload_date_str[0:4]}-{upload_date_str[4:6]}-{upload_date_str[6:8]}"
    info_data = {
        "video_id": video_id,
        "url_original": info.get('webpage_url'),
        "titulo": info.get('title'),
        "autor": info.get('uploader'),
        "id_canal": info.get('channel_id'),
        "data_upload": formatted_date,
        "duracao_segundos": info.get('duration'),
        "url_thumbnail": info.get('thumbnail'),
        "timestamp_download": int(time.time())
    }
    json_filepath = os.path.join(output_dir, 'info.json')
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, ensure_ascii=False, indent=4)
        log("info", {"msg": f"Arquivo de metadados salvo em {json_filepath}"})
    except IOError as e:
        log("error", {"code": 5, "msg": f"Não foi possível salvar o arquivo info.json: {str(e)}"})
        sys.exit(1)
    ydl_opts_download = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'progress_hooks': [progress_hook],
        'outtmpl': os.path.join(output_dir, 'video.mp4'),
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
            ydl.download([video_id])
            log("done", {"video_id": video_id, "filename": os.path.join(output_dir, "video.mp4")})
    except Exception as e:
        log("error", {"code": 6, "msg": f"Erro no download: {str(e)}"})
        sys.exit(1)

if __name__ == "__main__":
    main()