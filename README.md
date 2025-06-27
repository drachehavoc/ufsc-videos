# UFSC Vídeos

Anteriormente chamado de Cérebro UFLisC, é um sistema de busca, pesquisa e referenciação acadêmica de vídeos com conteúdo científico.

## Anotações

### Se não funcionar o CUDA

dentro do venv execute:

```bash
pip uninstall llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```