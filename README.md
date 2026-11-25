# System predykcji zachorowania na cukrzycę

Aplikacja liczy prawdopodobieństwo cukrzycy na podstawie parametrów zdrowotnych, pokazuje wskaźnik ryzyka i zapisuje każde zapytanie w SQLite.

Dane wejściowe modelu to plik `diabetes.csv` (8 cech klinicznych i kolumna `Outcome`).

## Stack

- Python 3.11 (obraz Dockera; lokalnie wystarczy 3.11+)
- FastAPI
- HTML5 i Bootstrap 5
- SQLite
- scikit-learn
- SHAP
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

Obraz jest na Docker Hubie: [skuzyk/predykcja-cukrzycy](https://hub.docker.com/r/skuzyk/predykcja-cukrzycy).

Pobranie i uruchomienie:

```powershell
docker pull skuzyk/predykcja-cukrzycy
docker run --rm -p 8000:8000 -v ${PWD}/data:/app/data skuzyk/predykcja-cukrzycy
```

Aplikacja jest pod adresem http://localhost:8000. Katalog `data` na hoście trzyma historię zapytań.

Żeby zbudować obraz z tego repozytorium zamiast go pobierać:

```powershell
docker compose up --build
```

## Struktura

```text
app/
  main.py         trasy stron i API
  ml.py           trening, porównanie modeli, SHAP
  csv_io.py       wczytywanie i zapis CSV
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

Potok: zamiana zer oznaczających brak pomiaru na braki, imputacja medianą, standaryzacja, a potem regresja logistyczna albo las losowy.

Zero oznacza brak pomiaru dla glukozy, ciśnienia, fałdu skórnego, insuliny i BMI. Liczba ciąż równa zero jest prawdziwą wartością.

Oba algorytmy są oceniane na podziale 80/20 (`random_state=42`) i w 5-krotnej walidacji krzyżowej. Metryki to dokładność, precyzja, czułość, F1 i ROC AUC. Do predykcji służy model z wyższym ROC AUC z walidacji krzyżowej, nauczony ponownie na całym pliku. Hiperparametry (C, liczba drzew, maksymalna głębokość) ustawia się na stronie głównej. Wyniki są w `models/metrics.json`.

Pojedyncze zapytanie z formularza dostaje wyjaśnienie SHAP: które cechy podniosły, a które obniżyły wynik modelu.

Nowe przypadki można dodać formularzem albo plikiem CSV. Historia i podzbiór wysokiego ryzyka (prawdopodobieństwo co najmniej 50%) pobierają się jako CSV.

## API

- `POST /api/predict` — predykcja i zapis
- `GET /api/predictions` — historia
- `GET /api/predictions.csv` — historia jako CSV
- `GET /api/predictions.csv?subset=wysokie` — podzbiór wysokiego ryzyka
- `GET /api/health` — stan aplikacji
- `GET /docs` — dokumentacja interaktywna

Przykład:

```powershell
Invoke-RestMethod http://localhost:8000/api/predict -Method Post -ContentType "application/json" -Body '{"pregnancies":1,"glucose":89,"blood_pressure":66,"skin_thickness":23,"insulin":94,"bmi":28.1,"diabetes_pedigree":0.167,"age":21}'
```

Wszystkie zapytania trafiają do jednej lokalnej bazy. Ta wersja nie ma kont użytkowników.
