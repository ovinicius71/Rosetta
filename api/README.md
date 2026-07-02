# api/ — serviço FastAPI (Rosetta)

Ponte entre o canvas (web) e o modelo (ml).

## Endpoints
| Método | Rota         | Entrada                          | Saída                          |
|--------|--------------|----------------------------------|--------------------------------|
| POST   | `/recognize` | tinta (esquema compartilhado)    | `{ "latex": "x^2" }`           |
| POST   | `/evaluate`  | `{ "latex": "1+1" }` (opcional)  | `{ "result": "2", ... }` SymPy |
| GET    | `/health`    | —                                | `{ "status": "ok" }`           |

O corpo de `/recognize` segue **`schemas/ink.schema.json`** (espelhado em `schemas.py`).

## Rodar (da RAIZ do repo — caminhos do config são relativos a ela)

```powershell
$env:PYTHONPATH = "api/src;ml/src"
$env:HMER_CKPT  = "checkpoints/crohme/last.ckpt"     # ou overfit_crohme p/ teste
$env:HMER_CONFIG = "ml/configs/crohme.yaml"          # config do TREINO do checkpoint
C:\HMER\venv\Scripts\python.exe -m uvicorn hmer_api.main:app --port 8000
```

Variáveis: `HMER_CKPT` (sem ela → 501 stub), `HMER_CONFIG` (arquitetura + vocab do
checkpoint), `HMER_DEVICE` (`cpu` default — não disputa a GPU com treino em andamento).

O modelo carrega no primeiro request (lazy). `/evaluate` usa SymPy (`parse_latex`):
aritmética exata, frações, raízes, resolve equações de 1 incógnita.
