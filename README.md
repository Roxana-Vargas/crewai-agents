# Proyecto simple con CrewAI

Este proyecto crea un flujo multiagente para evaluacion de descanso medico:
- Agente extractor de certificado de salud
- Agente extractor de DNI
- Agente de dictamen final

El sistema extrae datos con Document AI y aplica reglas de negocio para aprobar o rechazar.
El proveedor configurado por defecto es Gemini via Google API Key.
La extraccion de texto de PDF se hace con Google Document AI.

## 1) Crear entorno virtual (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Instalar dependencias

```powershell
pip install -r requirements.txt
```

## 3) Configurar variables de entorno

Copia `.env.example` a `.env`:

```powershell
Copy-Item .env.example .env
```

Luego edita `.env` con tu configuracion:

```txt
MODEL=gemini/gemini-2.5-flash
GOOGLE_API_KEY=tu_google_api_key
GOOGLE_CLOUD_PROJECT_ID=tu-proyecto-gcp
DOCUMENTAI_LOCATION=us
DOCUMENTAI_PROCESSOR_ID=tu-processor-id
HEALTH_PDF_PATH=health_certificate.pdf
DNI_PDF_PATH=dni.pdf
RULE_MAX_REST_DAYS=30
RULE_MAX_CERTIFICATE_AGE_DAYS=15
```

Necesitas crear un processor en Document AI (por ejemplo, OCR Processor) y copiar su ID.

## 4) Ejecutar

```powershell
python main.py
```

## 5) Colocar los PDFs de entrada

Pon los archivos en las rutas definidas por `HEALTH_PDF_PATH` y `DNI_PDF_PATH`.

## Notas

- Si falta `GOOGLE_API_KEY`, el script te lo indicara.
- Si falta el PDF o falla Document AI, el script mostrara el error.
- El resultado final sale en JSON con: `health_data`, `dni_data`, `business_result`, `decision_report`.
