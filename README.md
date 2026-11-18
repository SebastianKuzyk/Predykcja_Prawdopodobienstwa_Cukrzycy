# System predykcji zachorowania na cukrzycę

Aplikacja liczy prawdopodobieństwo cukrzycy na podstawie parametrów zdrowotnych, pokazuje wskaźnik ryzyka i zapisuje każde zapytanie w SQLite.

Dane wejściowe modelu to plik `diabetes.csv` (8 cech klinicznych i kolumna `Outcome`).

## Stack

- Python 3.11 (obraz Dockera; lokalnie wystarczy 3.11+)
- FastAPI
- HTML5 i Bootstrap 5
- SQLite
- scikit-learn
- Docker

## Uruchomienie lokalne

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --reload-dir app
```

Aplikacja jest pod adresem http://localhost:8000.

Przy pierwszym starcie, gdy brakuje pliku modelu, trening uruchamia się sam. Żeby przeliczyć model od nowa:

```powershell
python -m app.ml
```

Testy:

```powershell
pytest
```

## Docker

```powershell
docker compose up --build
```

Historia zapytań zostaje w katalogu `data/` na hoście. Model powstaje w trakcie budowania obrazu.

## Struktura

```text
app/
  main.py         trasy stron i API
  ml.py           trening i predykcja
  service.py      zapis i odczyt historii
  db_models.py    tabela predictions
  database.py     połączenie SQLite
  schemas.py      walidacja danych
  forms.py        pola formularza
  risk.py         progi wskaźnika ryzyka
  templates/      widoki HTML
  static/         style
diabetes.csv      dane treningowe
Dockerfile
docker-compose.yml
```

## Model

Potok: zamiana zer oznaczających brak pomiaru na braki, imputacja medianą, standaryzacja, regresja logistyczna.

Zero oznacza brak pomiaru dla glukozy, ciśnienia, fałdu skórnego, insuliny i BMI. Liczba ciąż równa zero jest prawdziwą wartością.

Metryki (dokładność, czułość, precyzja, F1, ROC AUC) są liczone na 20% wierszy odłożonych przed oceną (`random_state=42`). Do predykcji służy model nauczony ponownie na całym pliku. Wyniki widać na stronie głównej i w `models/metrics.json`.

## API

- `POST /api/predict` — predykcja i zapis
- `GET /api/predictions` — historia
- `GET /api/health` — stan aplikacji
- `GET /docs` — dokumentacja interaktywna

Przykład:

```powershell
Invoke-RestMethod http://localhost:8000/api/predict -Method Post -ContentType "application/json" -Body '{"pregnancies":1,"glucose":89,"blood_pressure":66,"skin_thickness":23,"insulin":94,"bmi":28.1,"diabetes_pedigree":0.167,"age":21}'
```

Wszystkie zapytania trafiają do jednej lokalnej bazy. Ta wersja nie ma kont użytkowników.
