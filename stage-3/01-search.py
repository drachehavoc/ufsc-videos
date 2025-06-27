import sys
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def log(action, data):
    print(json.dumps({"action": action, "data": data}), flush=True)

def perform_search(query_text, model, index, video_map, k, data_root):
    try:
        log("info", {"query": query_text})
        query_embedding = model.encode(query_text, convert_to_numpy=True).astype('float32')
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        distances, indices = index.search(query_embedding, k)
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            
            if idx == -1:
                continue

            video_id = video_map.get(str(idx))
            if video_id:
                # --- LÓGICA DE ENRIQUECIMENTO ---
                title = "Título não encontrado"
                thumbnail_url = None
                try:
                    info_path = os.path.join(data_root, video_id, "info.json")
                    if os.path.exists(info_path):
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info_data = json.load(f)
                            title = info_data.get('titulo', title)
                            thumbnail_url = info_data.get('url_thumbnail')
                except Exception:
                    # Se houver erro ao ler o info.json, não quebra a busca
                    pass 
                results.append({
                    "video_id": video_id,
                    "title": title,
                    "thumbnail_url": thumbnail_url,
                    "distance": float(dist)
                })
        log("result", results)
    except Exception as e:
        log("error", {"code": 5, "msg": f"Falha durante a busca: {str(e)}"})

def main():
    data_root = os.path.join("..", "data")
    faiss_file = os.path.join(data_root, "videos.faiss")
    map_file = os.path.join(data_root, "videos_map.json")
    if not os.path.exists(faiss_file) or not os.path.exists(map_file):
        log("error", {"code": 2, "msg": f"Arquivos de índice global não encontrados em '{data_root}'. Execute o script da stage-2 primeiro."})
        sys.exit(1)
    try:
        log("info", "Carregando modelo de IA (isso pode levar um momento)...")
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        model = SentenceTransformer(model_name, device='cuda')
        log("info", "Carregando índice FAISS e mapa de vídeos...")
        index = faiss.read_index(faiss_file)
        with open(map_file, 'r', encoding='utf-8') as f:
            video_map = json.load(f)
        log("success", "Sistema de busca pronto.")
    except Exception as e:
        log("error", {"code": 3, "msg": f"Falha ao carregar modelo ou índices: {str(e)}"})
        sys.exit(1)
    if len(sys.argv) > 1:
        query_text = sys.argv[1]
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        log("start", {"mode": "single_run", "query": query_text, "k": k})
        perform_search(query_text, model, index, video_map, k, data_root)
    else:
        k = 5
        log("start", {"mode": "interactive", "k": k})
        print("\nModo interativo. Digite sua busca e pressione Enter.")
        print("Digite 'exit' ou 'quit' para sair.")
        while True:
            try:
                query_text = input("> ")
                if query_text.lower() in ['exit', 'quit']:
                    break
                if not query_text.strip():
                    continue
                perform_search(query_text, model, index, video_map, k, data_root)
            except (KeyboardInterrupt, EOFError):
                break
        
        log("done", "Sessão interativa encerrada.")

if __name__ == "__main__":
    main()