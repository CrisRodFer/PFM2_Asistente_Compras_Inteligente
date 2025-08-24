
# scripts/utils/generar_calendario_estacional.py
# =============================================================================
# NOMBRE: generar_calendario_estacional.py
# DESCRIPCIÓN:
#   Función genérica y "pura" que construye un calendario estacional diario para
#   un año dado con pesos normalizados (suma = 1.0). No escribe a disco.
#
#   Combina:
#     - ciclo mensual de ingresos,
#     - estacionalidad semanal,
#     - eventos del ecommerce (fechas fijas y reglas dinámicas).
#
#   Incluye placeholders paramétricos para:
#     - Puentes nacionales,
#     - Vuelta al cole,
#     - Amazon Prime Day.
#
# FLUJO:
#   1) Generación del calendario base (365/366 días).
#   2) Cálculo de pesos parciales: mensual, semanal y eventos.
#   3) Combinación y normalización (tratamiento explícito del 29/02 si aplica).
#   4) Devuelve DataFrame con columnas auxiliares y 'Peso Normalizado'.
#
# OUTPUT (columnas):
#   ['Date','Month','Day','Weekday','Evento',
#    'w_monthly','w_weekly','w_event','Peso Final',
#    'Peso Normalizado','LeapNote']
#
# DEPENDENCIAS:
#   - pandas (>=1.5)
#   pip install pandas
#
# NOTA:
#   La validación estructural (365/366, suma=1, sin nulos/negativos) se hace
#   en un script aparte (p.ej., validar_calendario_estacional.py).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
import calendar as _cal
import datetime as _dt
import pandas as pd


# =============================================================================
# 1. UTILIDADES DE FECHA
# =============================================================================

def _is_leap(year: int) -> bool:
    """True si el año es bisiesto."""
    return _cal.isleap(year)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> _dt.date:
    """
    Devuelve la fecha del n-ésimo 'weekday' (0=lunes..6=domingo) del mes.
    Ej.: tercer lunes de enero → weekday=0, n=3.
    """
    count = 0
    last_day = _cal.monthrange(year, month)[1]
    for day in range(1, last_day + 1):
        d = _dt.date(year, month, day)
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
    raise ValueError("No existe ese n-ésimo weekday en el mes.")


def _last_weekday_of_month(year: int, month: int, weekday: int) -> _dt.date:
    """Último 'weekday' (0=lunes..6=domingo) del mes."""
    last_day = _cal.monthrange(year, month)[1]
    for day in range(last_day, 0, -1):
        d = _dt.date(year, month, day)
        if d.weekday() == weekday:
            return d
    raise RuntimeError("No se pudo calcular el último weekday del mes.")


def _easter_sunday(year: int) -> _dt.date:
    """
    Domingo de Pascua (algoritmo gregoriano anónimo).
    Útil para estimar Semana Santa (jueves/viernes santo, etc.).
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = 1 + ((h + l - 7*m + 114) % 31)
    return _dt.date(year, month, day)


# =============================================================================
# 2. EVENTOS
# =============================================================================

@dataclass
class EventRule:
    """
    Regla de evento:
      - name: nombre del evento (se almacena en 'Evento'; si hay múltiples en el mismo día, se concatenan)
      - multiplier: multiplicador del peso para las fechas afectadas (>1 sube, <1 baja)
      - dates: lista opcional de fechas fijas (date o str 'YYYY-MM-DD')
      - rule: callable(year)-> iterable[date] para fechas dinámicas (e.g., último viernes de noviembre)
    """
    name: str
    multiplier: float
    dates: list | None = None
    rule: callable | None = None


def _default_monthly_cycle(day: int) -> float:
    """
    Ciclo mensual por día del mes (1..31):
      - 1-10: mayor peso (cobro)
      - 11-20: neutro
      - 21-fin: menor peso
    """
    if day <= 10:
        return 1.2
    elif day <= 20:
        return 1.0
    return 0.8


def _default_weekly_weights() -> dict:
    """Pesos por día de la semana (en inglés como dt.day_name())."""
    return {
        "Friday": 1.10,
        "Saturday": 1.10,
        "Sunday": 1.10,
        "Monday": 0.95,
        "Tuesday": 0.95,
        # Wednesday/Thursday → 1.0 por defecto
    }


# --- Placeholders activables por parámetro -----------------------------------

def _puentes_nacionales(year: int) -> list[_dt.date]:
    """
    PLACEHOLDER Puentes nacionales:
    Devuelve las fechas de días puente relevantes (cuando haya un festivo que genera puente).
    Por defecto devuelve lista vacía para no afectar el patrón.
    TODO: poblar con reglas/festivos reales y devolver fechas concretas.
    """
    return []  # ← sin efecto hasta que se alimente


def _back_to_school_period(year: int) -> list[_dt.date]:
    """
    PLACEHOLDER Vuelta al cole:
    Por defecto (placeholder) devuelve 1–15 de septiembre como rango ligero.
    Ajusta según necesidad o reemplaza por lógica real.
    """
    return [_dt.date(year, 9, d) for d in range(1, 16)]


def _prime_day_dates(year: int) -> list[_dt.date]:
    """
    PLACEHOLDER Amazon Prime Day:
    Aproximación habitual: 2 días a mediados de julio.
    Aquí se usa 2º martes y 2º miércoles de julio como placeholder.
    Ajusta si conoces las fechas exactas.
    """
    tuesday = _nth_weekday_of_month(year, 7, weekday=1, n=2)   # 1=Tuesday
    wednesday = tuesday + _dt.timedelta(days=1)
    return [tuesday, wednesday]


def _materialize_events(year: int, event_rules: list[EventRule]) -> pd.DataFrame:
    """
    Construye un DataFrame con columnas:
      - Date (datetime64[ns])
      - Evento (string con nombres concatenados por '; ')
      - w_event (producto de multiplicadores cuando coinciden varios eventos en el mismo día)
    """
    records: dict[_dt.date, float] = {}
    labels: dict[_dt.date, list[str]] = {}

    for er in event_rules:
        # Fechas directas
        dates: list[_dt.date] = []
        if er.dates:
            for d in er.dates:
                if isinstance(d, str):
                    d = _dt.date.fromisoformat(d)
                dates.append(d)

        # Regla dinámica
        if er.rule:
            for d in er.rule(year):
                dates.append(d)

        for d in dates:
            if d.year != year:
                continue
            records.setdefault(d, 1.0)
            records[d] *= er.multiplier
            labels.setdefault(d, [])
            labels[d].append(er.name)

    if not records:
        return pd.DataFrame(columns=["Date", "Evento", "w_event"])

    df = pd.DataFrame({
        "Date": pd.to_datetime(list(records.keys())),
        "w_event": list(records.values()),
        "Evento": ["; ".join(labels[d]) for d in records.keys()],
    }).sort_values("Date")

    return df


def _default_event_rules(
    year: int,
    *,
    enable_puentes: bool = False,
    enable_back_to_school: bool = False,
    enable_prime_day: bool = False,
) -> list[EventRule]:
    """
    Conjunto base de eventos + placeholders opcionales (desactivados por defecto).
    """
    # Fechas dinámicas básicas
    blue_monday = _nth_weekday_of_month(year, 1, 0, 3)               # 3er lunes de enero
    black_friday = _last_weekday_of_month(year, 11, 4)               # último viernes de noviembre (4=viernes)
    cyber_monday = black_friday + _dt.timedelta(days=3)

    # Semana Santa (aprox): jueves-lunes alrededor de Pascua
    easter = _easter_sunday(year)
    maundy_thursday = easter - _dt.timedelta(days=3)  # Jueves Santo
    good_friday = easter - _dt.timedelta(days=2)      # Viernes Santo
    holy_saturday = easter - _dt.timedelta(days=1)    # Sábado Santo
    easter_monday = easter + _dt.timedelta(days=1)    # Lunes de Pascua

    rules: list[EventRule] = [
        # Rebajas (rangos)
        EventRule(
            name="Rebajas Invierno", multiplier=1.50,
            dates=[_dt.date(year, 1, d) for d in range(1, 16)]
        ),
        EventRule(
            name="Rebajas Verano", multiplier=1.50,
            dates=[_dt.date(year, 7, d) for d in range(1, 16)]
        ),

        # Fechas fijas
        EventRule(name="San Valentín", multiplier=1.80, dates=[_dt.date(year, 2, 14)]),
        EventRule(name="Navidad", multiplier=1.80, dates=[_dt.date(year, 12, 25)]),
        EventRule(name="Día del Padre", multiplier=1.20, dates=[_dt.date(year, 3, 19)]),

        # Dinámicas core
        EventRule(name="Blue Monday", multiplier=1.20, dates=[blue_monday]),
        EventRule(name="Black Friday", multiplier=3.00, dates=[black_friday]),
        EventRule(name="Cyber Monday", multiplier=2.00, dates=[cyber_monday]),

        # Semana Santa (multiplicadores suaves)
        EventRule(name="Jueves Santo", multiplier=1.20, dates=[maundy_thursday]),
        EventRule(name="Viernes Santo", multiplier=1.30, dates=[good_friday]),
        EventRule(name="Sábado Santo", multiplier=1.15, dates=[holy_saturday]),
        EventRule(name="Lunes de Pascua", multiplier=1.10, dates=[easter_monday]),
    ]

    # --- Placeholders activables por parámetro (desactivados por defecto) -----
    if enable_puentes:
        puente_dates = _puentes_nacionales(year)  # ← por defecto [], sin efecto
        if puente_dates:
            rules.append(EventRule(name="Puentes nacionales", multiplier=0.90, dates=puente_dates))

    if enable_back_to_school:
        bts_dates = _back_to_school_period(year)  # 1–15 sep (placeholder)
        rules.append(EventRule(name="Vuelta al cole", multiplier=1.05, dates=bts_dates))

    if enable_prime_day:
        prime_dates = _prime_day_dates(year)      # 2º mar y mié de julio (placeholder)
        rules.append(EventRule(name="Amazon Prime Day", multiplier=1.20, dates=prime_dates))

    return rules


# =============================================================================
# 3. FUNCIÓN GENÉRICA (PURA)
# =============================================================================

def generar_calendario_estacional(
    anio: int = 2024,
    *,
    monthly_cycle: callable | None = None,
    weekly_weights: dict | None = None,
    event_rules: list[EventRule] | None = None,
    leap_strategy: str = "explicit",
    normalize: bool = True,
    # Activadores de placeholders:
    enable_puentes: bool = False,
    enable_back_to_school: bool = False,
    enable_prime_day: bool = False,
) -> pd.DataFrame:
    """
    Genera un calendario estacional diario para 'anio' y devuelve un DataFrame.
    No realiza I/O (no escribe a disco ni imprime).

    Parámetros:
      - monthly_cycle: callable(day:int)->float. Si None, usa ciclo por defecto.
      - weekly_weights: dict con claves como dt.day_name() ('Monday'..'Sunday').
      - event_rules: lista de EventRule. Si None, usa reglas por defecto (+ placeholders activables).
      - leap_strategy: 'explicit' (29/02 con peso) o 'redistribute' (29/02 peso 0 y renormaliza el resto).
      - normalize: True → garantiza suma total de 'Peso Normalizado' = 1.0.
      - enable_puentes / enable_back_to_school / enable_prime_day: activan placeholders (por defecto desactivados).
    """
    # 1) Calendario base
    dates = pd.date_range(f"{anio}-01-01", f"{anio}-12-31", freq="D")
    df = pd.DataFrame({"Date": dates})
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.day_name()

    # 2) Pesos parciales
    if monthly_cycle is None:
        monthly_cycle = _default_monthly_cycle
    df["w_monthly"] = df["Day"].apply(lambda d: float(monthly_cycle(int(d))))

    if weekly_weights is None:
        weekly_weights = _default_weekly_weights()
    df["w_weekly"] = df["Weekday"].map(weekly_weights).fillna(1.0).astype(float)

    if event_rules is None:
        event_rules = _default_event_rules(
            anio,
            enable_puentes=enable_puentes,
            enable_back_to_school=enable_back_to_school,
            enable_prime_day=enable_prime_day,
        )
    events_df = _materialize_events(anio, event_rules)
    df = df.merge(events_df, on="Date", how="left")
    df["w_event"] = df["w_event"].fillna(1.0).astype(float)
    df["Evento"] = df["Evento"].fillna("")

    # 3) Combinación y tratamiento del 29/02 (si bisiesto)
    df["Peso Final"] = df["w_monthly"] * df["w_weekly"] * df["w_event"]

    leap_note = "n/a"
    if _is_leap(anio):
        feb29_mask = (df["Date"].dt.month == 2) & (df["Date"].dt.day == 29)
        if leap_strategy == "redistribute":
            # Asignar peso 0 al 29/02 y renormalizar el resto (manteniendo 366 filas)
            df.loc[feb29_mask, "Peso Final"] = 0.0
            leap_note = "redistribute"
        else:
            leap_note = "explicit"
    df["LeapNote"] = leap_note

    # 4) Normalización final
    if normalize:
        if _is_leap(anio) and leap_strategy == "redistribute":
            mask_others = ~((df["Date"].dt.month == 2) & (df["Date"].dt.day == 29))
            total_others = df.loc[mask_others, "Peso Final"].sum()
            df.loc[mask_others, "Peso Normalizado"] = df.loc[mask_others, "Peso Final"] / total_others
            df.loc[~mask_others, "Peso Normalizado"] = 0.0
        else:
            total = df["Peso Final"].sum()
            df["Peso Normalizado"] = df["Peso Final"] / total

    # Orden final de columnas
    cols = [
        "Date", "Month", "Day", "Weekday", "Evento",
        "w_monthly", "w_weekly", "w_event", "Peso Final",
        "Peso Normalizado", "LeapNote"
    ]
    return df[cols]
