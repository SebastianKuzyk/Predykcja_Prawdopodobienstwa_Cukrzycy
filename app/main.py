import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.csv_io import parse_patients_csv, predictions_to_csv
from app.database import get_db, init_db
from app.forms import EXAMPLES, FORM_FIELDS, FormError, parse_form, parse_hyperparameters
from app.ml import ensure_model, explain_patient, load_hyperparameters, load_metrics, train_and_save
from app.schemas import PatientInput, PredictionRead
from app.service import create_prediction, list_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def pretty_number(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def format_dt(value: datetime | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("T", " ")[:16]
    return value.strftime("%d.%m.%Y %H:%M")


templates.env.filters["pretty"] = pretty_number
templates.env.filters["format_dt"] = format_dt


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_model()
    yield


app = FastAPI(
    title="Predykcja cukrzycy",
    description=(
        "Szacowanie prawdopodobieństwa cukrzycy na podstawie parametrów medycznych. "
        "Wynik ma charakter edukacyjny i nie jest diagnozą."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def render(
    request: Request,
    template_name: str,
    context: dict | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    payload = {
        "active": "home",
        "fields": FORM_FIELDS,
        "examples": EXAMPLES,
        "metrics": load_metrics(),
        "hyperparameters": load_hyperparameters(),
        "form": {},
        "result": None,
        "explanations": None,
        "error": None,
        "notice": None,
        "warnings": [],
        "predictions": [],
    }
    if context:
        payload.update(context)
    return templates.TemplateResponse(
        request,
        template_name,
        payload,
        status_code=status_code,
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return render(request, "index.html")


@app.post("/", response_class=HTMLResponse)
async def predict_form(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    form = await request.form()
    raw = {field["name"]: str(form.get(field["name"], "")) for field in FORM_FIELDS}
    try:
        patient = parse_form(raw)
    except FormError as exc:
        return render(
            request,
            "index.html",
            {"error": exc.message, "form": raw},
            status_code=400,
        )
    result = create_prediction(db, patient)
    explanations = explain_patient(patient)
    return render(
        request,
        "index.html",
        {"result": result, "form": raw, "explanations": explanations},
    )


@app.post("/model", response_class=HTMLResponse)
async def retrain_model(request: Request) -> HTMLResponse:
    form = await request.form()
    raw = {key: str(form.get(key, "")) for key in ("logreg_c", "forest_n_estimators", "forest_max_depth")}
    try:
        settings = parse_hyperparameters(raw)
    except FormError as exc:
        return render(request, "index.html", {"error": exc.message}, status_code=400)
    metrics = train_and_save(settings)
    logger.info("Przeliczono modele. Wybrano %s.", metrics["selected"])
    return render(
        request,
        "index.html",
        {"notice": f"Modele przeliczono. Do predykcji wybrano: {metrics['selected_label']}."},
    )


@app.get("/historia", response_class=HTMLResponse)
def history(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return render(
        request,
        "history.html",
        {"active": "history", "predictions": list_predictions(db)},
    )


@app.post("/csv", response_class=HTMLResponse)
async def upload_csv(
    request: Request,
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    next_page: str = Form("home", alias="next"),
) -> HTMLResponse:
    destination = "history" if next_page == "history" else "home"
    template_name = "history.html" if destination == "history" else "index.html"
    payload = await file.read()
    if len(payload) > 1_000_000:
        return render(
            request,
            template_name,
            {
                "active": destination,
                "error": "Plik CSV jest za duży.",
                "predictions": list_predictions(db) if destination == "history" else [],
            },
            status_code=400,
        )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.decode("cp1250")
    patients, messages = parse_patients_csv(text)
    for patient in patients:
        create_prediction(db, patient)
    notice = None
    if patients:
        count = len(patients)
        if count == 1:
            noun = "wiersz"
        elif 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
            noun = "wiersze"
        else:
            noun = "wierszy"
        notice = f"Zapisano {count} {noun} z pliku CSV."
        logger.info("Wczytano %s wierszy z CSV.", len(patients))
    elif not messages:
        messages = ["Plik CSV nie zawiera wierszy do zapisania."]
    return render(
        request,
        template_name,
        {
            "active": destination,
            "notice": notice,
            "warnings": messages,
            "predictions": list_predictions(db) if destination == "history" else [],
        },
        status_code=200 if patients else 400,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictionRead)
def api_predict(patient: PatientInput, db: Session = Depends(get_db)) -> PredictionRead:
    return create_prediction(db, patient)


@app.get("/api/predictions", response_model=list[PredictionRead])
def api_predictions(
    limit: int = Query(200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[PredictionRead]:
    return list_predictions(db, limit)


@app.get("/api/predictions.csv")
def api_predictions_csv(
    subset: str | None = Query(None),
    db: Session = Depends(get_db),
) -> Response:
    high_risk_only = subset == "wysokie"
    content = predictions_to_csv(list_predictions(db, limit=500), high_risk_only=high_risk_only)
    filename = "predykcje-wysokie.csv" if high_risk_only else "predykcje.csv"
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
