# TrustGate

TrustGate is a local full-stack demo of a security gateway for AI chat. It evaluates each prompt before sending it to a language model, tracks risk across a conversation, and scans model responses for personally identifiable information (PII) before showing them to the user. A monitoring interface shows decisions, session risk, and pipeline records.

## What it does

Each message passes through these stages:

1. **Prompt Guard** scores prompts for benign content and four attack classes: prompt injection, jailbreak, data exfiltration, and model extraction. It also measures semantic similarity and deviation from known examples.
2. **Supervisor** combines the prompt signals with a session risk tracker and decides whether to allow, flag, or block the prompt.
3. **LLM client** sends allowed or flagged prompts to a configured model provider. Groq, Google Gemini, Ollama, and a deterministic mock backend are supported.
4. **Privacy Sieve** scans generated responses for PII and masks detected values before returning the response.
5. **SQLite** stores pipeline records for the monitoring dashboard and session analytics.

The frontend provides a chat page and a monitor page with session activity, risk information, decisions, records, and analytics.

> TrustGate is an educational prototype. Its detection and redaction are best-effort and must not be treated as a complete security or privacy control for production systems.

## Project layout

```text
backend/        FastAPI application and SQLite persistence
dataset/        Prompt Guard training data and generation script
frontend/       React + Vite chat and monitoring UI
llm/            Groq, Gemini, Ollama, and mock model clients
models/         Trained Prompt Guard model artifacts
privacy_sieve/  PII detection and response masking
prompt_guard/   Prompt scoring, semantic analysis, training code
supervisor/     Policy decisions and per-session risk tracking
```

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer and npm
- Git LFS to download the tracked embedding model (see [Git LFS](https://git-lfs.com/))
- Optional: an API key for Groq or Gemini, or a local [Ollama](https://ollama.com/) installation

## Get the code

```bash
git clone https://github.com/Aditya-dott/TrustGate.git
cd TrustGate
git lfs pull
```

Git LFS is required because `models/embedding_model.pkl` is larger than GitHub's normal 100 MB file limit. Without the LFS client, Git may check out only a small pointer file where the model is expected.

## Run locally

### 1. Install Python dependencies

From the repository root, create and activate a virtual environment, then install the backend requirements:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Then install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure a language model (optional)

Without a provider configured, TrustGate uses its mock backend, which is useful for trying the interface without a model API key.

For Groq, set the key in the same terminal before starting the backend:

```powershell
$env:GROQ_API_KEY="your-groq-api-key"
```

For Gemini:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
```

To explicitly select a backend, set `TRUSTGATE_LLM_BACKEND` to `groq`, `gemini`, `ollama`, or `mock`. For Ollama, install and start Ollama, pull a model (for example, `ollama pull llama3.2`), then set `TRUSTGATE_LLM_BACKEND=ollama`. `OLLAMA_HOST` can override the default `http://localhost:11434` host.

Do not commit API keys or put them in source files.

### 3. Start the backend

In a terminal at the repository root, with the virtual environment active:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`; interactive API documentation is at `http://localhost:8000/docs`. The first startup loads the local classifier and embedding model. If the optional spaCy English model is not installed, PII detection continues in regex-only mode.

### 4. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (by default, `http://localhost:5173`). The Vite development server proxies `/api` requests to the backend on port 8000.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `TRUSTGATE_LLM_BACKEND` | Force `groq`, `gemini`, `ollama`, or `mock` | Auto-select from configured credentials; mock if none |
| `GROQ_API_KEY` | Groq API credential | Unset |
| `GEMINI_API_KEY` | Google Gemini API credential | Unset |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `TRUSTGATE_DB_PATH` | SQLite database file path; use `:memory:` for ephemeral storage | `backend/trustgate.db` |
| `TRUSTGATE_PORT` | Port used by `python backend/main.py` | `8000` |

## API overview

The FastAPI backend exposes:

- `GET /api/health` — backend health and selected LLM information
- `POST /api/session/new` — create a session
- `POST /api/chat` — evaluate and process a prompt
- `GET /api/session/{session_id}/state` — current session risk state
- `GET /api/session/{session_id}/records` — session pipeline records
- `GET /api/session/{session_id}/analytics` — session analytics
- `GET /api/records/{record_id}` — retrieve one record
- `DELETE /api/session/{session_id}` — clear a session and its stored records

## Model and data files

The pre-trained Prompt Guard artifacts are in `models/`. The embedding artifact is tracked with Git LFS; clone with Git LFS installed to retrieve it. The training dataset is in `dataset/prompts_dataset.csv`. Retraining is optional and can take substantial time; the script documents its command-line options:

```bash
python -m prompt_guard.train_model --help
```

## Development checks

The frontend production bundle can be built with:

```bash
cd frontend
npm run build
```

The repository also contains focused Python validation scripts named `test_step*.py` and a Prompt Guard smoke test. Run the backend validation script from the repository root with:

```bash
python backend/test_step7.py
```

## Data and privacy

Conversation records are stored locally in SQLite by default. Prompts and model responses may therefore be written to `backend/trustgate.db`. Choose a different `TRUSTGATE_DB_PATH` or use `:memory:` if you do not want the default database file. When using Groq or Gemini, approved prompts are sent to that provider and subject to its terms and privacy policy. The mock backend makes no external model request.

## License

No license has been added yet. Until a license is included, standard copyright applies and reuse permissions are not granted by default.
