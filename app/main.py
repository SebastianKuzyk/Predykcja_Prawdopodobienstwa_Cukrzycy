from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.forms import EXAMPLES, FORM_FIELDS, FormError, parse_form
from app.ml import ensure_model, load_metrics
from app.schemas import PatientInput, PredictionRead
from app.service import create_prediction, list_predictions

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
        "form": {},
        "result": None,
        "error": None,
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
    return render(request, "index.html", {"result": result, "form": raw})


@app.get("/historia", response_class=HTMLResponse)
def history(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return render(
        request,
        "history.html",
        {"active": "history", "predictions": list_predictions(db)},
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
