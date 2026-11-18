from dataclasses import dataclass


@dataclass(frozen=True)
class Risk:
    level: str
    label: str
    color: str
    badge: str
    bar: str


def probability_percent(probability: float) -> float:
    return round(probability * 100, 1)


def describe_risk(probability: float) -> Risk:
    """Progi są liczone od wartości pokazanej na wskaźniku, zaokrąglonej do 0,1%."""
    percent = probability_percent(probability)
    if percent < 33:
        return Risk("niskie", "Niskie ryzyko", "#198754", "text-bg-success", "bg-success")
    if percent < 66:
        return Risk(
            "umiarkowane",
            "Umiarkowane ryzyko",
            "#b58105",
            "text-bg-warning",
            "bg-warning",
        )
    return Risk("wysokie", "Wysokie ryzyko", "#dc3545", "text-bg-danger", "bg-danger")


def classification_label(prediction: int) -> str:
    if prediction == 1:
        return "Klasyfikacja przy progu 50%: wynik dodatni (cukrzyca)."
    return "Klasyfikacja przy progu 50%: wynik ujemny (brak cukrzycy)."
