# System predykcji zachorowania na cukrzycę

Aplikacja webowa szacuje prawdopodobieństwo cukrzycy na podstawie parametrów zdrowotnych, pokazuje wskaźnik ryzyka i zapisuje zapytania w SQLite. Projekt powstaje w pięciu etapach.

## Etapy

1. Szkielet FastAPI, układ strony i endpoint zdrowia.
2. Formularz parametrów zdrowotnych.
3. Zapis zapytań w SQLite i historia.
4. Model uczenia maszynowego i wskaźnik ryzyka.
5. Testy oraz wdrożenie w Dockerze.

W tym repozytorium jest etap 1.

## Stack etapu 1

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
