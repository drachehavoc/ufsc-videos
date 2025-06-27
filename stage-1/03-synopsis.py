import sys
import json
import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

def log(action, data):
    try:
        payload = {"action": action, "data": data}
        print(json.dumps(payload))
        sys.stdout.flush()
    except TypeError:
        payload = {"action": action, "data": str(data)}
        print(json.dumps(payload))
        sys.stdout.flush()

def download_model(config):
    try:
        repo_id = config["repo_id"]
        filename = config["filename"]
        token = config["token"]
        revision = config["revision"]
        models_dir = config["models_dir"]
        log("info", f"Iniciando download do modelo '{filename}' para a pasta '{models_dir}'.")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
            cache_dir=models_dir
        )
        return model_path        
    except Exception as e:
        log("error", {"code": 6, "msg": f"Falha no download do modelo: {str(e)}"})
        sys.exit(1)

def summarize_chunk(llm, text_chunk, config):
    messages = [
        {"role": "system", "content": "Você é um assistente que resume partes da transcrição de um vídeo. Extraia os pontos-chave de forma concisa."},
        {"role": "user", "content": f"Resuma em português do Brasil a seguinte parte da transcrição de um vídeo:\n\n---\n{text_chunk}\n---"}
    ]
    output = llm.create_chat_completion(
        messages=messages,
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=256
    )
    return output['choices'][0]['message']['content'].strip()

def main():
    config = {
        "repo_id": os.environ.get("SINOPSIS_HUGGINGFACE_REPO_ID"),
        "filename": os.environ.get("SINOPSIS_HUGGINGFACE_FILE"),
        "revision": os.environ.get("SINOPSIS_HUGGINGFACE_REVISION", "main"),
        "token": os.environ.get("SINOPSIS_HUGGINGFACE_TOKEN"),
        "models_dir": os.environ.get("SINOPSIS_MODELS_DIR", "./llm-models"),
        "context_window": int(os.environ.get("SINOPSIS_LLAMA_CONTEXT_WINDOW", 8192)),
        "gpu_layers": int(os.environ.get("SINOPSIS_LLAMA_GPU_LAYERS", -1)),
        "threads": int(os.environ.get("SINOPSIS_LLAMA_THREADS", 4)),
        "verbose": os.environ.get("SINOPSIS_LLAMA_VERBOSE", "False").lower() == "true",
        "chunk_size": int(os.environ.get("SINOPSIS_CHUNK_SIZE_TOKENS", 7000)),
        "temperature": float(os.environ.get("SINOPSIS_LLAMA_TEMPERATURE", 0.7)),
        "top_p": float(os.environ.get("SINOPSIS_LLAMA_TOP_P", 0.9)),
    }
    if not all([config["repo_id"], config["filename"], config["token"]]):
        log("error", {"code": 10, "msg": "Variáveis de ambiente essenciais (REPO_ID, FILE, TOKEN) não estão definidas."})
        sys.exit(1)
    if len(sys.argv) != 2:
        log("error", {"code": 1, "msg": f"Uso: {sys.argv[0]} <video_id>"})
        sys.exit(1)
    video_id = sys.argv[1]
    data_dir = os.path.join("..", "data", video_id)
    input_file = os.path.join(data_dir, "transcription.json")
    output_file = os.path.join(data_dir, "synopsis.txt")
    if os.path.isfile(output_file):
        log("info", f"O arquivo de sinopse '{output_file}' já existe. Processo ignorado.")
        sys.exit(0) 
    if not os.path.isfile(input_file):
        log("error", {"code": 2, "msg": f"Arquivo de entrada não encontrado: {input_file}"})
        sys.exit(1)
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        text = " ".join(seg.get("text", "") for seg in input_data.get("segments", []) if seg.get("text"))
        if not text.strip():
            log("error", {"code": 4, "msg": "Nenhum texto encontrado no arquivo para resumir."})
            sys.exit(1)
    except Exception as e:
        log("error", {"code": 3, "msg": f"Falha ao ler o arquivo de entrada: {str(e)}"})
        sys.exit(1)
    log("progress", 0.0)
    try:
        log("progress", 0.1)
        model_path = download_model(config)
        log("info", f"Caminho do modelo: {model_path}")
        log("progress", 0.2)
        log("info", "Carregando o modelo na memória...")
        llm = Llama(
            model_path=model_path,
            n_ctx=config["context_window"],
            n_gpu_layers=config["gpu_layers"],
            n_threads=config["threads"],
            verbose=config["verbose"]
        )
        log("info", "Modelo carregado com sucesso.")
        log("progress", 0.3)
        text_tokens = llm.tokenize(text.encode("utf-8", errors="ignore"))
        token_count = len(text_tokens)
        synopsis = ""
        if token_count > config["chunk_size"]:
            log("info", f"Texto longo ({token_count} tokens). Usando estratégia Map-Reduce.")
            summaries = []
            num_chunks = (token_count + config["chunk_size"] - 1) // config["chunk_size"]
            for i in range(num_chunks):
                start = i * config["chunk_size"]
                end = start + config["chunk_size"]
                chunk_tokens = text_tokens[start:end]
                chunk_text = llm.detokenize(chunk_tokens).decode("utf-8", errors="ignore")
                log("info", f"Resumindo pedaço {i+1} de {num_chunks}...")
                partial_summary = summarize_chunk(llm, chunk_text, config)
                summaries.append(partial_summary)
                log("progress", 0.3 + (0.6 * (i + 1) / num_chunks))
            log("info", "Combinando resumos parciais para criar a sinopse final.")
            combined_summaries = "\n\n".join(summaries)
            messages_reduce = [
                {"role": "system", "content": "Você é um mestre em síntese. Crie uma sinopse final e coerente para um vídeo a partir de vários resumos parciais de sua transcrição."},
                {"role": "user", "content": f"Crie uma sinopse final em Português do Brasil a partir destes resumos:\n\n---\n{combined_summaries}\n---\n\nSinopse Final:"}
            ]
            final_output = llm.create_chat_completion(messages=messages_reduce, temperature=config["temperature"], top_p=config["top_p"], max_tokens=350)
            synopsis = final_output['choices'][0]['message']['content'].strip()
        else:
            log("info", "Texto curto. Gerando resumo direto.")
            messages_direct = [
                {"role": "system", "content": "Você é um assistente prestativo que cria sinopses curtas e concisas de vídeos."},
                {"role": "user", "content": f"Gere uma sinopse breve para o vídeo com a seguinte transcrição:\n\n---\n{text}\n---\n\nSinopse:"}
            ]
            output = llm.create_chat_completion(messages=messages_direct, temperature=config["temperature"], top_p=config["top_p"], max_tokens=350)
            synopsis = output['choices'][0]['message']['content'].strip()
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(synopsis)
            log("info", f"Sinopse salva com sucesso em: {output_file}")
        except Exception as e:
            log("error", {"code": 7, "msg": f"Falha ao salvar o arquivo de sinopse: {str(e)}"})
            sys.exit(1)
        log("progress", 1.0)
        log("result", {"path": output_file})
    except Exception as e:
        log("error", {"code": 5, "msg": f"Falha na inferência do modelo: {str(e)}"})
        sys.exit(1)

if __name__ == "__main__":
    main()