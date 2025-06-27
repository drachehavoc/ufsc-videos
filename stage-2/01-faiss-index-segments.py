import sys
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import math

def log(action, data):
    print(json.dumps({"action": action, "data": data}), flush=True)

def main():
    if len(sys.argv) < 2:
        log("error", {"code": 1, "msg": "Parâmetro video_id obrigatório"})
        sys.exit(1)
    video_id = sys.argv[1]
    log("start", {"video_id": video_id})
    base_dir = os.path.join("..", "data", video_id)
    output_dir = os.path.join(base_dir, "faiss")
    synopsis_file = os.path.join(base_dir, "synopsis.txt")
    transcription_file = os.path.join(base_dir, "transcription.json")
    synopsis_npy_file = os.path.join(output_dir, "synopsis.npy")
    segments_npy_file = os.path.join(output_dir, "segments.npy")
    segments_faiss_file = os.path.join(output_dir, "segments.faiss")
    segments_map_file = os.path.join(output_dir, "segments_map.json")
    output_files = [synopsis_npy_file, segments_npy_file, segments_faiss_file, segments_map_file]
    if all(os.path.exists(f) for f in output_files):
        log("info", "Todos os arquivos de índice já existem para este vídeo. Processo ignorado.")
        sys.exit(0)
    if not os.path.exists(synopsis_file) or not os.path.exists(transcription_file):
        log("error", {"code": 3, "msg": f"Arquivos de entrada não encontrados em {base_dir}"})
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    log("info", "Carregando o modelo de sentence-transformer...")
    try:
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        model = SentenceTransformer(model_name, device='cuda')
        log("success", {"msg": f"Modelo '{model_name}' carregado com sucesso."})
    except Exception as e:
        log("error", {"code": 2, "msg": f"Falha ao carregar o modelo: {str(e)}"})
        sys.exit(1)
    try:
        log("info", "Processando a sinopse...")
        with open(synopsis_file, 'r', encoding='utf-8') as f:
            synopsis_text = f.read()
        if synopsis_text.strip():
            synopsis_embedding = model.encode(synopsis_text)
            np.save(synopsis_npy_file, synopsis_embedding)
            log("success", {"msg": "Embedding da sinopse salvo", "path": synopsis_npy_file})
        else:
            log("warning", "Arquivo de sinopse está vazio. Pulando.")
    except Exception as e:
        log("error", {"code": 4, "msg": f"Falha ao processar sinopse: {str(e)}"})
        sys.exit(1)
    try:
        log("info", "Processando a transcrição...")
        with open(transcription_file, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        segments = transcription_data.get("segments", [])
        if not segments:
            log("warning", "Nenhum segmento encontrado na transcrição. Finalizando.")
            sys.exit(0)
        texts = [seg['text'] for seg in segments]
        log("info", f"Gerando embeddings para {len(texts)} segmentos...")
        batch_size = 32
        all_embeddings = []
        num_texts = len(texts)
        num_batches = math.ceil(num_texts / batch_size)
        last_reported_progress = -1
        log("progress", 0.0)
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            batch_texts = texts[start_index:end_index]
            batch_embeddings = model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
            progress = round((i + 1) / num_batches, 2)
            if progress > last_reported_progress:
                log("progress", progress)
                last_reported_progress = progress
        segment_embeddings = np.vstack(all_embeddings)
        np.save(segments_npy_file, segment_embeddings)
        log("success", {"msg": "Embeddings dos segmentos salvos", "path": segments_npy_file})
        log("info", "Criando o índice FAISS...")
        d = segment_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(segment_embeddings)
        faiss.write_index(index, segments_faiss_file)
        log("success", {"msg": "Índice FAISS salvo", "path": segments_faiss_file})
        log("info", "Criando o mapa do índice...")
        segment_map = {i: seg['id'] for i, seg in enumerate(segments)}
        with open(segments_map_file, 'w', encoding='utf-8') as f:
            json.dump(segment_map, f, indent=2)
        log("success", {"msg": "Mapa dos segmentos salvo", "path": segments_map_file})
    except Exception as e:
        log("error", {"code": 5, "msg": f"Falha ao processar transcrição: {str(e)}"})
        sys.exit(1)
    log("done", f"Todos os arquivos de índice para o vídeo '{video_id}' foram gerados com sucesso.")
if __name__ == "__main__":
    main()