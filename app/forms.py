from pydantic import ValidationError

from app.schemas import PatientInput

PEDIGREE_LEVELS = [
    {
        "level": 0,
        "weight": 0.0,
        "weight_label": "0.000",
        "title": "Brak historii w rodzinie",
        "who": "Brak znanych przypadków cukrzycy wśród bliskich krewnych.",
    },
    {
        "level": 1,
        "weight": 0.125,
        "weight_label": "0.125",
        "title": "Krewni trzeciego stopnia",
        "who": "Kuzyni pierwszego stopnia.",
    },
    {
        "level": 2,
        "weight": 0.25,
        "weight_label": "0.250",
        "title": "Krewni drugiego stopnia",
        "who": "Dziadkowie, wujkowie, ciotki.",
    },
    {
        "level": 3,
        "weight": 0.5,
        "weight_label": "0.500",
        "title": "Krewni pierwszego stopnia",
        "who": "Rodzice, rodzeństwo, własne dzieci.",
    },
]

FORM_FIELDS = [
    {
        "name": "pregnancies",
        "label": "Liczba ciąż",
        "unit": "",
        "hint": "Ile razy osoba była w ciąży.",
        "min": 0,
        "max": 20,
        "step": "1",
        "integer": True,
    },
    {
        "name": "glucose",
        "label": "Glukoza",
        "unit": "mg/dL",
        "hint": "Stężenie glukozy w osoczu 2 godziny po podaniu glukozy.",
        "min": 0,
        "max": 400,
        "step": "any",
        "integer": False,
    },
    {
        "name": "blood_pressure",
        "label": "Ciśnienie krwi",
        "unit": "mm Hg",
        "hint": "Rozkurczowe ciśnienie krwi.",
        "min": 0,
        "max": 250,
        "step": "any",
        "integer": False,
    },
    {
        "name": "skin_thickness",
        "label": "Grubość fałdu skórnego",
        "unit": "mm",
        "hint": "Fałd skórny na tricepsie.",
        "min": 0,
        "max": 100,
        "step": "any",
        "integer": False,
    },
    {
        "name": "insulin",
        "label": "Insulina",
        "unit": "μU/ml",
        "hint": "Insulina w surowicy 2 godziny po teście.",
        "min": 0,
        "max": 1000,
        "step": "any",
        "integer": False,
    },
    {
        "name": "bmi",
        "label": "BMI",
        "unit": "kg/m²",
        "hint": "Wskaźnik masy ciała.",
        "min": 0,
        "max": 80,
        "step": "any",
        "integer": False,
    },
    {
        "name": "diabetes_pedigree",
        "label": "Wskaźnik rodowodu cukrzycy",
        "unit": "",
        "hint": "Wybierz poziom 0, 1, 2 albo 3. Ikona obok opisuje, kogo dotyczy każdy poziom.",
        "min": 0,
        "max": 3,
        "step": "any",
        "integer": False,
        "choices": PEDIGREE_LEVELS,
    },
    {
        "name": "age",
        "label": "Wiek",
        "unit": "lata",
        "hint": "Wiek w latach.",
        "min": 1,
        "max": 120,
        "step": "1",
        "integer": True,
    },
]

EXAMPLES = {
    "niskie": {
        "pregnancies": 1,
        "glucose": 89,
        "blood_pressure": 66,
        "skin_thickness": 23,
        "insulin": 94,
        "bmi": 28.1,
        "diabetes_pedigree": 0.167,
        "age": 21,
    },
    "wysokie": {
        "pregnancies": 1,
        "glucose": 189,
        "blood_pressure": 60,
        "skin_thickness": 23,
        "insulin": 846,
        "bmi": 30.1,
        "diabetes_pedigree": 0.398,
        "age": 59,
    },
}

LABELS = {field["name"]: field["label"] for field in FORM_FIELDS}


class FormError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def format_validation_error(exc: ValidationError) -> str:
    messages: list[str] = []
    for error in exc.errors():
        field_name = str(error["loc"][-1])
        label = LABELS.get(field_name, field_name)
        error_type = error.get("type")
        ctx = error.get("ctx") or {}
        if error_type == "greater_than_equal":
            messages.append(f"{label}: minimalna wartość to {ctx.get('ge')}.")
        elif error_type == "less_than_equal":
            messages.append(f"{label}: maksymalna wartość to {ctx.get('le')}.")
        else:
            messages.append(f"{label}: niepoprawna wartość.")
    return " ".join(messages)


def parse_hyperparameters(raw: dict[str, str]) -> dict[str, float | int]:
    specs = [
        ("logreg_c", "C regresji logistycznej", 0.01, 100, False),
        ("forest_n_estimators", "Liczba drzew", 50, 400, True),
        ("forest_max_depth", "Maksymalna głębokość lasu", 2, 20, True),
    ]
    parsed: dict[str, float | int] = {}
    errors: list[str] = []
    for name, label, minimum, maximum, integer in specs:
        text = str(raw.get(name, "")).strip().replace(",", ".")
        if text == "":
            errors.append(f"Uzupełnij pole: {label}.")
            continue
        try:
            number = float(text)
        except ValueError:
            errors.append(f"Pole „{label}” musi być liczbą.")
            continue
        if integer and not number.is_integer():
            errors.append(f"Pole „{label}” musi być liczbą całkowitą.")
            continue
        if number < minimum or number > maximum:
            errors.append(f"{label}: dozwolony zakres to {minimum}–{maximum}.")
            continue
        parsed[name] = int(number) if integer else number
    if errors:
        raise FormError(" ".join(errors))
    return parsed


def parse_form(raw: dict[str, str], pedigree_as_level: bool = True) -> PatientInput:
    parsed: dict[str, int | float] = {}
    errors: list[str] = []

    for field in FORM_FIELDS:
        name = field["name"]
        label = field["label"]
        text = str(raw.get(name, "")).strip().replace(",", ".")
        if text == "":
            errors.append(f"Uzupełnij pole: {label}.")
            continue
        try:
            number = float(text)
        except ValueError:
            errors.append(f"Pole „{label}” musi być liczbą.")
            continue
        if field.get("choices") and pedigree_as_level:
            if not number.is_integer():
                errors.append(f"Pole „{label}” wymaga poziomu 0, 1, 2 albo 3.")
                continue
            level = int(number)
            choice = next((item for item in field["choices"] if item["level"] == level), None)
            if choice is None:
                errors.append(f"Pole „{label}” wymaga poziomu 0, 1, 2 albo 3.")
                continue
            parsed[name] = choice["weight"]
            continue
        if field.get("choices"):
            parsed[name] = number
            continue
        if field["integer"] and not number.is_integer():
            errors.append(f"Pole „{label}” musi być liczbą całkowitą.")
            continue
        parsed[name] = int(number) if field["integer"] else number

    if errors:
        raise FormError(" ".join(errors))

    try:
        return PatientInput(**parsed)
    except ValidationError as exc:
        raise FormError(format_validation_error(exc)) from exc
