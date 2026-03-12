import os
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import PermissionDenied
from google.cloud import documentai
from json_repair import repair_json


def build_llm() -> LLM:
    model = os.getenv("MODEL", "gemini/gemini-2.5-flash")
    return LLM(model=model)


def process_with_document_ai(pdf_path: str) -> tuple[str, list[dict[str, Any]]]:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = os.getenv("DOCUMENTAI_LOCATION", "us")
    processor_id = os.getenv("DOCUMENTAI_PROCESSOR_ID")

    if not project_id or not processor_id:
        raise ValueError(
            "Faltan GOOGLE_CLOUD_PROJECT_ID o DOCUMENTAI_PROCESSOR_ID en .env"
        )

    endpoint = f"{location}-documentai.googleapis.com"
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=endpoint)
    )

    name = client.processor_path(project_id, location, processor_id)
    pdf_bytes = Path(pdf_path).read_bytes()
    raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    try:
        result = client.process_document(request=request)
    except PermissionDenied as exc:
        raise RuntimeError(
            "Document AI devolvio 403. Verifica que la identidad de ADC "
            "(service account o usuario autenticado) tenga permisos sobre el "
            "processor y que documentai.googleapis.com este habilitado en el "
            "proyecto del processor."
        ) from exc

    document = result.document
    entities = []
    for entity in document.entities:
        entities.append(
            {
                "type": entity.type_,
                "mention_text": entity.mention_text,
                "confidence": float(entity.confidence) if entity.confidence is not None else None,
                "normalized_value": (
                    entity.normalized_value.text
                    if entity.normalized_value and entity.normalized_value.text
                    else None
                ),
            }
        )

    return (document.text or ""), entities


def entities_to_field_map(entities: list[dict[str, Any]]) -> dict[str, Any]:
    field_map: dict[str, Any] = {}
    for entity in entities:
        key = entity.get("type")
        if not key:
            continue
        value = entity.get("normalized_value") or entity.get("mention_text")
        if key not in field_map and value not in (None, ""):
            field_map[key] = value
    return field_map


def parse_json_output(raw_text: str, label: str) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        repaired = repair_json(raw_text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(f"No se pudo parsear JSON de {label}: {raw_text}") from exc


def build_health_extraction_crew(llm: LLM) -> Crew:
    health_agent = Agent(
        role="Analista de Certificados de Salud",
        goal="Extraer campos estructurados de certificados de descanso medico",
        backstory="Especialista en lectura de certificados medicos en espanol.",
        llm=llm,
        verbose=True,
    )

    health_task = Task(
        description=(
            "Extrae del certificado de salud usando primero las entidades detectadas "
            "(detected_entities_json) y, de ser necesario, el texto completo. "
            "Devuelve SOLO JSON valido con: "
            "devuelve SOLO JSON valido: "
            "autogenerado, citt, contingencia, dias_acumulados_consecutivos, "
            "dias_acumulados_no_consecutivos, documento_identidad, fecha_fin, "
            "fecha_inicio, fecha_otorgamiento, medico, medico_cmp, medico_cop, "
            "nombre_asegurado, total_dias, total_dias_incapacidad_acumulado. "
            "Usa formato YYYY-MM-DD para fechas cuando sea posible. "
            "Si no encuentras un campo usa null. "
            "Entidades: {detected_entities_json}. Texto: {document_text}"
        ),
        expected_output="Un objeto JSON valido y sin texto adicional.",
        agent=health_agent,
    )

    return Crew(
        agents=[health_agent],
        tasks=[health_task],
        process=Process.sequential,
        verbose=True,
    )


def build_dni_extraction_crew(llm: LLM) -> Crew:
    dni_agent = Agent(
        role="Analista de DNI",
        goal="Extraer campos clave de documentos nacionales de identidad",
        backstory="Especialista en lectura de DNI y validacion de datos personales.",
        llm=llm,
        verbose=True,
    )

    dni_task = Task(
        description=(
            "Extrae del DNI usando primero las entidades detectadas "
            "(detected_entities_json) y, de ser necesario, el texto completo. "
            "Devuelve SOLO JSON "
            "valido: full_name, dni_number, birth_date(YYYY-MM-DD o null), "
            "expiration_date(YYYY-MM-DD o null), document_number, nationality. "
            "Si no encuentras un campo usa null. "
            "Entidades: {detected_entities_json}. Texto: {document_text}"
        ),
        expected_output="Un objeto JSON valido y sin texto adicional.",
        agent=dni_agent,
    )

    return Crew(
        agents=[dni_agent],
        tasks=[dni_task],
        process=Process.sequential,
        verbose=True,
    )


def safe_date(value: Any) -> date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        try:
            return datetime.fromisoformat(value).date()
        except (TypeError, ValueError):
            return None


def first_non_empty(source: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = source.get(key)
        if value not in (None, "", "null"):
            return value
    return None


def evaluate_business_rules(health_data: dict[str, Any], dni_data: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    max_rest_days = int(os.getenv("RULE_MAX_REST_DAYS", "30"))
    max_certificate_age_days = int(os.getenv("RULE_MAX_CERTIFICATE_AGE_DAYS", "15"))

    required_health_fields = [
        "nombre_asegurado",
        "documento_identidad",
        "fecha_otorgamiento",
        "total_dias",
        "medico",
    ]
    for field in required_health_fields:
        if health_data.get(field) in (None, "", "null"):
            reasons.append(f"Falta campo obligatorio en certificado: {field}")

    required_dni_fields = ["full_name", "dni_number"]
    for field in required_dni_fields:
        if dni_data.get(field) in (None, "", "null"):
            reasons.append(f"Falta campo obligatorio en DNI: {field}")

    cert_dni = str(health_data.get("documento_identidad", "")).strip()
    dni_number = str(dni_data.get("dni_number", "")).strip()
    if cert_dni and dni_number and cert_dni != dni_number:
        reasons.append("El DNI del certificado no coincide con el DNI presentado")

    rest_days_raw = first_non_empty(health_data, ["total_dias", "dias_acumulados_consecutivos"])
    try:
        rest_days = int(rest_days_raw)
    except (ValueError, TypeError):
        rest_days = -1

    if rest_days <= 0:
        reasons.append("Los dias de descanso medico deben ser mayores a 0")
    if rest_days > max_rest_days:
        reasons.append(
            f"Los dias de descanso ({rest_days}) superan el maximo permitido ({max_rest_days})"
        )

    issue_date = safe_date(health_data.get("fecha_otorgamiento"))
    if not issue_date:
        reasons.append("La fecha de emision del certificado es invalida o faltante")
    else:
        age_days = (date.today() - issue_date).days
        if age_days < 0:
            reasons.append("La fecha de otorgamiento no puede ser futura")
        if age_days > max_certificate_age_days:
            reasons.append(
                "El certificado esta fuera de vigencia para evaluacion "
                f"({age_days} dias de antiguedad)"
            )

    start_date = safe_date(health_data.get("fecha_inicio"))
    end_date = safe_date(health_data.get("fecha_fin"))
    if start_date and end_date and end_date < start_date:
        reasons.append("La fecha_fin no puede ser menor que fecha_inicio")

    expiration_date = safe_date(dni_data.get("expiration_date"))
    if expiration_date and expiration_date < date.today():
        reasons.append("El DNI esta vencido")

    approved = len(reasons) == 0
    return {
        "approved": approved,
        "decision": "APROBADO" if approved else "RECHAZADO",
        "reasons": reasons,
        "rules": {
            "max_rest_days": max_rest_days,
            "max_certificate_age_days": max_certificate_age_days,
        },
    }


def build_decision_crew(llm: LLM) -> Crew:
    decision_agent = Agent(
        role="Oficial de Cumplimiento",
        goal="Emitir dictamen final claro para el expediente",
        backstory="Evalua resultados y redacta una conclusion operativa y auditables.",
        llm=llm,
        verbose=True,
    )

    decision_task = Task(
        description=(
            "Con base en health_data, dni_data y business_result, redacta SOLO JSON "
            "valido con campos: final_decision, executive_summary, checklist. "
            "Respeta estrictamente business_result.decision como decision final. "
            "health_data={health_data}; dni_data={dni_data}; business_result={business_result}"
        ),
        expected_output="Un objeto JSON valido y sin texto adicional.",
        agent=decision_agent,
    )

    return Crew(
        agents=[decision_agent],
        tasks=[decision_task],
        process=Process.sequential,
        verbose=True,
    )


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Falta GOOGLE_API_KEY. Configurala en .env antes de ejecutar.")
        return

    health_pdf_path = os.getenv("HEALTH_PDF_PATH", "descanso.pdf")
    dni_pdf_path = os.getenv("DNI_PDF_PATH", "dni.pdf")

    if not Path(health_pdf_path).exists():
        print(f"No existe el PDF del certificado: {health_pdf_path}")
        return
    if not Path(dni_pdf_path).exists():
        print(f"No existe el PDF del DNI: {dni_pdf_path}")
        return

    try:
        health_text, health_entities = process_with_document_ai(health_pdf_path)
        dni_text, dni_entities = process_with_document_ai(dni_pdf_path)
    except Exception as exc:
        print(f"Error al procesar PDF con Document AI: {exc}")
        print("Este proyecto esta en modo solo Document AI (sin lectura local).")
        return

    if not health_text.strip() or not dni_text.strip():
        print(
            "Document AI no devolvio texto en uno de los documentos. Verifica PDF "
            "y configuracion del processor."
        )
        return

    llm = build_llm()
    health_crew = build_health_extraction_crew(llm=llm)
    dni_crew = build_dni_extraction_crew(llm=llm)

    if health_entities:
        print("\nEntidades detectadas en certificado:")
        for entity in health_entities:
            print(
                f"  Type: {entity['type']} | Mention: {entity['mention_text']} | "
                f"Confidence: {entity['confidence']}"
            )

    if dni_entities:
        print("\nEntidades detectadas en DNI:")
        for entity in dni_entities:
            print(
                f"  Type: {entity['type']} | Mention: {entity['mention_text']} | "
                f"Confidence: {entity['confidence']}"
            )

    health_raw = str(
        health_crew.kickoff(
            inputs={
                "document_text": health_text,
                "detected_entities_json": json.dumps(health_entities, ensure_ascii=False),
            }
        )
    )
    dni_raw = str(
        dni_crew.kickoff(
            inputs={
                "document_text": dni_text,
                "detected_entities_json": json.dumps(dni_entities, ensure_ascii=False),
            }
        )
    )

    try:
        health_data = parse_json_output(health_raw, "certificado de salud")
        dni_data = parse_json_output(dni_raw, "DNI")
    except ValueError as exc:
        print(f"Error al estructurar salida de agentes: {exc}")
        return

    business_result = evaluate_business_rules(health_data, dni_data)
    decision_crew = build_decision_crew(llm=llm)
    decision_raw = str(
        decision_crew.kickoff(
            inputs={
                "health_data": json.dumps(health_data, ensure_ascii=False),
                "dni_data": json.dumps(dni_data, ensure_ascii=False),
                "business_result": json.dumps(business_result, ensure_ascii=False),
            }
        )
    )

    try:
        decision_data = parse_json_output(decision_raw, "dictamen final")
    except ValueError:
        decision_data = {
            "final_decision": business_result["decision"],
            "executive_summary": "No se pudo parsear JSON del agente de dictamen.",
            "checklist": business_result["reasons"],
        }

    print("\n=== RESULTADO FINAL ===")
    print(json.dumps({
        "document_ai": {
            "health_entities": health_entities,
            "health_entity_fields": entities_to_field_map(health_entities),
            "dni_entities": dni_entities,
            "dni_entity_fields": entities_to_field_map(dni_entities),
        },
        "health_data": health_data,
        "dni_data": dni_data,
        "business_result": business_result,
        "decision_report": decision_data,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
