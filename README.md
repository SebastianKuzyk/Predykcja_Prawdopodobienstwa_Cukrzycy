# System predykcji zachorowania na cukrzycę

Aplikacja internetowa ma na podstawie podstawowych parametrów zdrowotnych oszacować prawdopodobieństwo cukrzycy i pokazać wynik w przeglądarce.

W repozytorium jest serwer FastAPI, wspólny układ strony oraz endpoint sprawdzający, czy usługa działa.

## Technologie

- Python 3.11+
- FastAPI
- HTML5 i Bootstrap 5

## Uruchomienie

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --reload-dir app
```

Aplikacja jest pod adresem http://localhost:8000. Stan serwisu zwraca http://localhost:8000/api/health.
