# stage-3/02-talk-marking.py

import sys
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# FUNÇÃO LOG CORRIGIDA, REVISADA E ABENÇOADA
def log(action, data):
    print(json.dumps({"action": action, "data": data}), flush=True)

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def generate_answer(llm, query, citations):
    if not citations:
        return "Desculpe, não encontrei nenhuma informação relevante sobre isso nos vídeos carregados."

    context = ""
    for i, cit in enumerate(citations):
        context += f"Trecho {i+1} (do vídeo '{cit['video_id']}' por {cit['author']} em {cit['timestamp']}):\n"
        context += f"\"{cit['text']}\"\n\n"

    system_prompt = "Você é um assistente prestativo. Sua tarefa é responder à pergunta do usuário baseando-se APENAS nos trechos de vídeo fornecidos. Seja conciso e direto. Mencione de qual vídeo ou autor você tirou a informação."
    
    user_prompt = f"Com base nos trechos abaixo, responda à seguinte pergunta: '{query}'\n\n--- TRECHOS ---\n{context}--- FIM DOS TRECHOS ---\n\nResposta:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        output = llm.create_chat_completion(messages=messages, temperature=0.5, max_tokens=150)
        return output['choices'][0]['message']['content'].strip()
    except Exception as e:
        log("error", {"msg": "Falha na geração de resposta do LLM", "error": str(e)})
        return "Ocorreu um erro ao tentar gerar a resposta."

def main():
    if len(sys.argv) < 2:
        log("error", {"code": 1, "msg": "Uso: python 02-talk-marking.py <video_id_1> ..."})
        sys.exit(1)

    video_ids_context = sys.argv[1:]
    log("start", {"mode": "qa_context", "context": video_ids_context})

    log("info", "Iniciando assistente de QA. Carregando todos os modelos...")
    try:
        retriever_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        retriever_model = SentenceTransformer(retriever_model_name, device='cuda')

        llm_repo_id = os.environ.get("LLM_HUGGINGFACE_REPO_ID")
        llm_filename = os.environ.get("LLM_HUGGINGFACE_FILE")
        llm_token = os.environ.get("LLM_HUGGINGFACE_TOKEN")
        llm_models_dir = os.environ.get("LLM_MODELS_DIR", "./llm-models")
        
        if not all([llm_repo_id, llm_filename, llm_token]):
             raise ValueError("Variáveis de ambiente do LLM (LLM_HUGGINGFACE_REPO_ID, LLM_HUGGINGFACE_FILE, LLM_HUGGINGFACE_TOKEN) não configuradas!")

        model_path = hf_hub_download(repo_id=llm_repo_id, filename=llm_filename, token=llm_token, cache_dir=llm_models_dir)
        llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)
        
    except Exception as e:
        log("error", {"code": 2, "msg": f"Falha crítica ao carregar modelos: {str(e)}"})
        sys.exit(1)

    log("info", "Construindo contexto de busca a partir dos vídeos fornecidos...")
    all_embeddings = []
    master_map = []
    data_root = os.path.join("..", "data")
    for video_id in video_ids_context:
        video_dir = os.path.join(data_root, video_id)
        segments_npy_file = os.path.join(video_dir, "faiss", "segments.npy")
        transcription_file = os.path.join(video_dir, "transcription.json")
        info_file = os.path.join(video_dir, "info.json")
        if not all(os.path.exists(f) for f in [segments_npy_file, transcription_file, info_file]):
            log("warning", f"Arquivos essenciais para o vídeo {video_id} não encontrados. Pulando.")
            continue
        try:
            embeddings = np.load(segments_npy_file)
            with open(transcription_file, 'r', encoding='utf-8') as f: transcription_data = json.load(f)
            with open(info_file, 'r', encoding='utf-8') as f: info_data = json.load(f)
            all_embeddings.append(embeddings)
            author = info_data.get("autor", "Autor desconhecido")
            segments_metadata = {seg['id']: seg for seg in transcription_data.get("segments", [])}
            for i in range(len(embeddings)):
                segment = segments_metadata.get(i)
                if segment:
                    master_map.append({
                        "video_id": video_id,
                        "author": author,
                        "timestamp": format_timestamp(segment.get("start", 0)),
                        "text": segment.get("text", "").strip()
                    })
        except Exception as e:
            log("error", {"msg": f"Falha ao carregar dados do vídeo {video_id}", "error": str(e)})


    if not all_embeddings:
        log("error", {"code": 4, "msg": "Nenhum vídeo válido carregado."})
        sys.exit(1)

    combined_embeddings = np.vstack(all_embeddings).astype('float32')
    faiss.normalize_L2(combined_embeddings)
    index = faiss.IndexFlatIP(combined_embeddings.shape[1])
    index.add(combined_embeddings)
    log("success", f"Assistente pronto. Contexto com {index.ntotal} trechos carregado.")

    k = 3
    similarity_threshold = 0.5
    print("\nFaça sua pergunta sobre os vídeos carregados.")
    print("Digite 'exit' ou 'quit' para sair.")
    
    while True:
        try:
            query_text = input("> ")
            if query_text.lower() in ['exit', 'quit']: break
            if not query_text.strip(): continue

            query_embedding = retriever_model.encode(query_text, convert_to_numpy=True).astype('float32')
            query_embedding = np.expand_dims(query_embedding, axis=0)
            faiss.normalize_L2(query_embedding)
            similarities, indices = index.search(query_embedding, k)
            
            citations = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                sim = similarities[0][i]
                if idx != -1 and sim >= similarity_threshold:
                    citation_data = master_map[idx].copy()
                    citation_data['similarity_score'] = float(sim)
                    citations.append(citation_data)
            
            generated_message = generate_answer(llm, query_text, citations)
            log("message", generated_message)
            
            if citations:
                citations.sort(key=lambda x: x['similarity_score'], reverse=True)
                log("result", citations)

        except (KeyboardInterrupt, EOFError):
            break

    log("done", "Sessão encerrada.")

if __name__ == "__main__":
    main()