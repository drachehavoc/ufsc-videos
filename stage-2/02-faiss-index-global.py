import os
import sys
import json
import numpy as np
import faiss
import glob

def log(action, data):
    print(json.dumps({"action": action, "data": data}), flush=True)

def main():
    log("start", {"script": "02-faiss-index-global"})
    data_root = os.path.join("..", "data")
    output_dir = os.path.join(data_root)
    global_faiss_file = os.path.join(output_dir, "videos.faiss")
    global_map_file = os.path.join(output_dir, "videos_map.json")
    os.makedirs(output_dir, exist_ok=True)
    log("info", "Iniciando busca por arquivos de sinopse processados...")
    search_pattern = os.path.join(data_root, "*", "faiss", "synopsis.npy")
    embedding_files = glob.glob(search_pattern)
    if not embedding_files:
        log("warning", "Nenhum arquivo 'synopsis.npy' encontrado. Nenhum índice foi gerado.")
        sys.exit(0)
    log("info", f"Encontrados {len(embedding_files)} vídeos para indexar.")
    all_embeddings = []
    video_ids_map = []
    for file_path in embedding_files:
        try:
            embedding = np.load(file_path)
            video_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            all_embeddings.append(embedding)
            video_ids_map.append(video_id)
        except Exception as e:
            log("error", {"msg": f"Falha ao carregar ou processar o arquivo {file_path}", "error": str(e)})
            continue
    if not all_embeddings:
        log("error", {"msg": "Falha ao carregar todos os embeddings encontrados. O índice não será gerado."})
        sys.exit(1)
    log("info", f"Construindo índice com {len(all_embeddings)} vetores...")
    try:
        embeddings_matrix = np.vstack(all_embeddings).astype('float32')
        d = embeddings_matrix.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings_matrix)
        faiss.write_index(index, global_faiss_file)
        log("success", {"msg": "Índice FAISS global salvo", "path": global_faiss_file})
        final_map = {i: vid for i, vid in enumerate(video_ids_map)}
        with open(global_map_file, 'w', encoding='utf-8') as f:
            json.dump(final_map, f, indent=2)
        log("success", {"msg": "Mapa de vídeos global salvo", "path": global_map_file})
    except Exception as e:
        log("error", {"msg": "Falha ao construir ou salvar o índice FAISS global", "error": str(e)})
        sys.exit(1)
    log("done", f"Índice global para {len(video_ids_map)} vídeos gerado com sucesso.")
if __name__ == "__main__":
    main()