"""Label-free OCR primitives extracted from the development experiments."""

from datetime import date, timedelta
from statistics import median
import re

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
NAME_FIELDS = FIELDS[:3]
LONG_FIELDS = ("Birth_place_ru", "Issue_organization_ru")
DATE_FIELDS = ("Birth_date", "Issue_date")
EVIDENCE_FIELDS = (*NAME_FIELDS, "Sex_ru", *DATE_FIELDS)
# Modal VIZ/MRZ alignments on development; E/YO is deliberately ambiguous.
TRANSLIT = dict(zip("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ -",
                    "ABVGDEEJZIQKLMNOPRSTUFHC34WXY9678<<"))


def norm(s):
    return " ".join(str(s or "").split()).upper()


def encode_name(value):
    return "".join(TRANSLIT.get(c, c) for c in norm(value))


def encode_names(fields):
    text = "PNRUS" + encode_name(fields[NAME_FIELDS[0]]) + "<<"
    text += "<".join(encode_name(fields[f]) for f in NAME_FIELDS[1:])
    return text.ljust(44, "<")[:44]


def check_digit(text, weights=(7, 3, 1)):
    if not re.fullmatch(r"[A-Z0-9<]+", text):
        return None
    values = [0 if c == "<" else int(c) if c.isdigit() else ord(c) - 55 for c in text]
    return str(sum(v * weights[i % len(weights)] for i, v in enumerate(values)) % 10)


def checksum_results(line, weights=(7, 3, 1)):
    blocks = {"number": (line[:9], line[9]), "birth": (line[13:19], line[19]),
              "optional": (line[28:42], line[42]),
              "composite": (line[:10] + line[13:20] + line[21:43], line[43])}
    return {name: check_digit(text, weights) == digit for name, (text, digit) in blocks.items()}


def parse_numeric(line):
    if not re.fullmatch(r"[0-9]{10}RUS[0-9]{7}[MF]<{7}[0-9]{13}<[0-9]{2}", line):
        return None
    digits = line[:3] + line[28] + line[3:9]
    return {"Licence_number": f"{digits[:2]} {digits[2:4]} {digits[4:]}",
            "Birth_date": line[13:19], "Sex_ru": {"M": "МУЖ.", "F": "ЖЕН."}[line[20]],
            "Issue_date": line[29:35], "Issue_organisation_code": line[35:38] + "-" + line[38:41]}


def date_key(value):
    value = norm(value)
    return value[8:10] + value[3:5] + value[:2] if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", value) else value


def normalize_short(field, value):
    text = norm(value)
    if field == "Sex_ru":
        return text.rstrip(".") + "." if text.rstrip(".") in ("МУЖ", "ЖЕН") else text
    if not text or not re.fullmatch(r"[0-9 ./-]+", text):
        return text
    digits = re.sub(r"\D", "", text)
    if field == "Licence_number" and len(digits) == 10:
        return digits[:2] + " " + digits[2:4] + " " + digits[4:]
    if field == "Issue_organisation_code" and len(digits) == 6:
        return digits[:3] + "-" + digits[3:]
    if field in DATE_FIELDS and len(digits) == 8:
        candidate = digits[:2] + "." + digits[2:4] + "." + digits[4:]
        if valid_date(candidate):
            return candidate
    return text


def valid_date(text):
    if not re.fullmatch(r"[0-9]{2}\.[0-9]{2}\.[0-9]{4}", text):
        return False
    try:
        date(int(text[6:]), int(text[3:5]), int(text[:2]))
    except ValueError:
        return False
    return True


def strict_mrz(prediction):
    lines = (prediction.get("mrz") or {}).get("lines") or []
    if len(lines) != 2:
        return None
    line = lines[1].get("text") or ""
    numeric = parse_numeric(line)
    return numeric if numeric is not None and all(checksum_results(line).values()) else None


def names_agree(prediction, fields):
    lines = (prediction.get("mrz") or {}).get("lines") or []
    if len(lines) != 2 or not fields["Last_name_ru"] or not fields["First_name_ru"]:
        return False
    if 8 + sum(len(norm(fields[f])) for f in FIELDS[:3]) >= 44:
        return False
    # A terminal filler avoids treating a truncated patronymic as fully verified.
    first = lines[0].get("text") or ""
    return len(first) == 44 and first.endswith("<") and first == encode_names(fields)


def patronymic_sex(value):
    text = norm(value)
    if text.endswith("ИЧ"):
        return "МУЖ."
    if text.endswith(("ОВНА", "ЕВНА", "ИЧНА")):
        return "ЖЕН."
    return None


def correct_fields(prediction, sex_policy="keep"):
    """No labels: formatting, then checksum-protected number/code; sex is a separate arm."""
    fields = viz_fields(prediction)
    for field in (*DATE_FIELDS, "Licence_number", "Issue_organisation_code", "Sex_ru"):
        fields[field] = normalize_short(field, fields[field])
    numeric = strict_mrz(prediction)
    if numeric:
        for field in ("Licence_number", "Issue_organisation_code"):
            fields[field] = numeric[field]
        supported = (fields["Sex_ru"] not in ("МУЖ.", "ЖЕН.")
                     and patronymic_sex(fields["Middle_name_ru"]) == numeric["Sex_ru"]
                     and names_agree(prediction, fields)
                     and ((prediction["viz"].get("Middle_name_ru") or {}).get("conf") or 0) >= 0.99)
        if sex_policy == "mrz" or (sex_policy == "supported" and supported):
            fields["Sex_ru"] = numeric["Sex_ru"]
    return fields


def at_least(feature, threshold):
    return feature is not None and feature >= threshold


def viz_fields(prediction):
    return {f: " ".join(line.get("text") or "" for line in
                       (prediction["viz"].get(f) or {}).get("lines", [])) for f in FIELDS}


def box_height_ratio(value):
    value = value or {}
    lines, box = value.get("lines") or [], value.get("box")
    if not lines or not box or len(box) != 4 or box[3] <= 0:
        return None
    boxes = [line.get("box") for line in lines]
    if any(not b or len(b) != 4 or b[3] <= 0 for b in boxes):
        return None
    return box[3] / (len(lines) * median(b[3] for b in boxes))


def long_field_passes(value, char_min, box_ratio):
    value = value or {}
    lines = value.get("lines") or []
    if not lines or not norm(" ".join(line.get("text") or "" for line in lines)):
        return False
    chars = [c for line in lines for c in (line.get("char_conf") or [])]
    if not chars or any(len(line.get("char_conf") or []) != len(line.get("text") or "") for line in lines):
        return False
    if any(c is None or not char_min <= c <= 1 for c in chars):
        return False
    ratio = box_height_ratio(value) if box_ratio is not None else None
    return box_ratio is None or (ratio is not None and 0 < ratio <= box_ratio)


def quality_accepts(prediction, fields, char_min, box_ratio, conf_min=0.99):
    """Experimental B + fixed YO rejection + long-field character/box gates."""
    if any("Ё" in norm(fields[f]) for f in NAME_FIELDS):
        return False
    numeric = strict_mrz(prediction)
    if numeric is None or not names_agree(prediction, fields):
        return False
    if fields["Sex_ru"] != numeric["Sex_ru"] or any(
            not valid_date(fields[f]) or date_key(fields[f]) != numeric[f] for f in DATE_FIELDS):
        return False
    suffix = patronymic_sex(fields["Middle_name_ru"])
    if suffix is not None and suffix != fields["Sex_ru"]:
        return False
    evidence = [f for f in EVIDENCE_FIELDS if f != "Middle_name_ru" or norm(fields[f])]
    if not all(at_least((prediction["viz"].get(f) or {}).get("conf"), conf_min)
               for f in (*evidence, *LONG_FIELDS)):
        return False
    return all(long_field_passes(prediction["viz"].get(f), char_min, box_ratio) for f in LONG_FIELDS)


VALID_STATUSES = {"valid_until_milestone", "valid_indefinite"}


def validity_status(birth_text, issue_text, as_of=date(2026, 10, 1)):
    """The date/milestone algorithm specified in the task, not an authenticity check."""
    if not valid_date(birth_text) or not valid_date(issue_text):
        return "invalid_dates"
    birth = date(int(birth_text[6:]), int(birth_text[3:5]), int(birth_text[:2]))
    issued = date(int(issue_text[6:]), int(issue_text[3:5]), int(issue_text[:2]))
    if issued > as_of:
        return "issued_future"
    if issued < birth:
        return "issued_before_14"

    def classify(leap_day):
        birthdays = []
        for years in (14, 20, 45):
            try:
                birthdays.append(birth.replace(year=birth.year + years))
            except ValueError:  # February 29 in a non-leap anniversary year.
                birthdays.append(date(birth.year + years, *leap_day))
        m14, m20, m45 = birthdays
        if issued < m14:
            return "issued_before_14"
        if issued >= m45:
            return "valid_indefinite"
        milestone = m20 if issued < m20 else m45
        if as_of <= milestone:
            return "valid_until_milestone"
        if as_of <= milestone + timedelta(days=90):
            return "uncertain_90_days"
        return "expired"

    early, late = classify((2, 28)), classify((3, 1))
    return early if early == late else "uncertain_leap_anniversary"
