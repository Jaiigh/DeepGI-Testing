import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle


INPUT_JSON = "input.json"
OUTPUT_PDF = "endoscopy_report.pdf"


LOCATION_ROWS = [
    "Terminal ileum",
    "Ileocecal valve",
    "Cecum",
    "Ascending colon",
    "Transverse colon",
    "Descending colon",
    "Sigmoid colon",
    "Rectum",
    "Anus",
    "Other",
]


def normalize(value):
    """Normalize a location string for case-insensitive comparison."""
    if value is None:
        return ""
    return str(value).strip().lower()


def is_positive(value):
    """True for values such as True, 1, '1', 'true', etc."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def has_intervention(result):
    """Determine whether this lesion gets an intervention number."""
    procedure = result.get("procedure")
    pathology = result.get("pathology")
    biopsy_pieces = result.get("biopsy_pieces")

    procedure_present = procedure is not None and str(procedure).strip() != ""

    pieces_present = False
    if biopsy_pieces is not None and str(biopsy_pieces).strip() != "":
        try:
            pieces_present = float(biopsy_pieces) > 0
        except (TypeError, ValueError):
            pieces_present = False

    intervention_present = procedure_present or pieces_present or is_positive(pathology)

    return (intervention_present, procedure_present, pieces_present, is_positive(pathology))


def make_intervention_text(result, intervention_number, intervention_flags):
    """Build one intervention row."""
    parts = []

    procedure = result.get("procedure")
    if intervention_flags[1]:
        parts.append(
            f"Polypectomy Method : {str(procedure).strip().capitalize()} Result : Success"
        )

    biopsy_pieces = result.get("biopsy_pieces")
    if intervention_flags[2]:
        parts.append(f"Clipping by {biopsy_pieces}")

    if intervention_flags[3]:
        parts.append("Pathology")

    return f"({intervention_number}) " + ", ".join(parts)


def build_location_text(location, records):
    """Build the finding text for one anatomical location."""
    location_records = [
        r for r in records if normalize(r.get("location")) == normalize(location)
    ]

    if not location_records:
        return "Normal"

    fragments = []
    for result in location_records:
        size = result.get("size_mm")
        lesion_type = str(result.get("lesion_type", "")).strip().capitalize()
        size_text = str(size) if size is not None else ""

        # Assign intervention numbers in first-come-first-served order.
        intervention_number = result.get("_intervention_number")
        number_text = (
            f"{intervention_number}" if intervention_number is not None else ""
        )

        if len(location_records) == 1:
            fragments.append(
                f"A {size_text} -cm. {lesion_type} was seen at {location}({number_text})"
            )
        else:
            fragments.append(
                f"A {size_text} -cm. {lesion_type} at {location}({number_text})"
            )

    if len(fragments) == 1:
        return fragments[0]

    return ", ".join(fragments) + " were seen."


def build_diagnosis_text(records):
    """Build the Diagnosis row."""
    if not records:
        return "Normal"

    values = [
        f"{str(r.get('lesion_type', '')).strip().capitalize()} at {str(r.get('location', '')).strip().capitalize()}"
        for r in records
    ]
    return ", ".join(values)


def build_intervention_text(records):
    """Build the Intervention row."""
    interventions = []

    for result in records:
        number = result.get("_intervention_number")
        if number is not None:
            interventions.append(
                make_intervention_text(result, number, has_intervention(result))
            )

    return interventions


def prepare_records(result):
    """prepare the result records."""
    if isinstance(result, list):
        records = result
    else:
        records = [result]

    intervention_number = 1
    prepared = []

    for original in records:
        record = dict(original)
        if record.get("lesion_type") is None:
            continue  # Skip records without a lesion type.

        if has_intervention(record):
            record["_intervention_number"] = intervention_number
            intervention_number += 1
        else:
            record["_intervention_number"] = None

        prepared.append(record)

    return prepared


def make_pdf(result, output_pdf):
    records = prepare_records(result)

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()

    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        spaceAfter=0,
    )

    value_style = ParagraphStyle(
        "Value",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        spaceAfter=0,
    )

    title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
    )

    rows = []

    # Section heading
    rows.append([
        Paragraph("Endoscopic findings", title_style),
        ""
    ])

    # Location rows
    for display_name in LOCATION_ROWS:
        value = Paragraph(build_location_text(display_name, records), value_style)
        rows.append([
            Paragraph(display_name, label_style),
            value,
        ])

    # Diagnosis
    diagnosis = build_diagnosis_text(records)
    rows.append([
        Paragraph("Diagnosis", label_style),
        Paragraph(diagnosis, value_style),
    ])

    # Intervention
    intervention_lines = build_intervention_text(records)
    if intervention_lines:
        intervention_html = "<br/>".join(intervention_lines)
        intervention_value = Paragraph(intervention_html, value_style)
    else:
        intervention_value = Paragraph("None", value_style)

    rows.append([
        Paragraph("Intervention", label_style),
        intervention_value,
    ])

    # Fixed rows
    rows.append([
        Paragraph("Complications", label_style),
        Paragraph("No", value_style),
    ])

    rows.append([
        Paragraph("Comment", label_style),
        Paragraph("None", value_style),
    ])

    table = Table(
        rows,
        colWidths=[40 * mm, 135 * mm],
        hAlign="LEFT",
    )

    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),

        # Extra space after the section heading.
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),

        # Space before Diagnosis / Intervention / Complications / Comment.
        ("TOPPADDING", (0, 11), (-1, 11), 6),
        ("TOPPADDING", (0, 12), (-1, 12), 6),
        ("TOPPADDING", (0, 13), (-1, 13), 6),
        ("TOPPADDING", (0, 14), (-1, 14), 6),
    ]))

    doc.build([table])


def load_findings(path):
    """Read the JSON file and return all result objects from findings."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    findings = data.get("findings", [])

    if not isinstance(findings, list):
        raise ValueError('"findings" must be a list.')

    results = []

    for i, finding in enumerate(findings, start=1):
        if not isinstance(finding, dict):
            raise ValueError(f"findings[{i - 1}] must be an object.")

        if "result" not in finding:
            continue

        result = finding["result"]

        if not isinstance(result, dict):
            raise ValueError(f'findings[{i - 1}]["result"] must be an object.')

        results.append(result)

    return results


def generate_pdf_from_json(input_json, output_pdf):
    """Read a DeepGI case JSON file and generate its endoscopy PDF."""
    results = load_findings(input_json)

    if not results:
        raise ValueError("No result objects were found inside the findings array.")

    make_pdf(results, output_pdf)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / INPUT_JSON
    output_path = script_dir / OUTPUT_PDF

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input JSON not found: {input_path}\n"
            f"Put {INPUT_JSON!r} in the same directory as this Python file."
        )

    generate_pdf_from_json(input_path, output_path)
    print(f"Created: {output_path}")
