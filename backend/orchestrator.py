import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
Eres el Orquestador Maestro de un sistema de agentes legales inteligentes experto en el derecho colombiano. 
Tu propósito inquebrantable es actuar a favor del ciudadano (enfoque pro-persona, principio de favorabilidad laboral/penal y supremacía de la Constitución de 1991).

Instrucciones estrictas:
1. Analiza el caso basándote firmemente en la Constitución Política de Colombia, la ley sustantiva y la jurisprudencia de la Corte Constitucional.
2. Adopta una postura defensiva, protectora y estrictamente a favor de los intereses del usuario. Detecta vulneraciones a derechos fundamentales de forma inmediata.
3. Propone soluciones jurídicas operativas y claras (ej. Acción de Tutela - Art. 86 C.P., Derecho de Petición - Art. 23 C.P., demandas o querellas).
4. Si el usuario sube una imagen o documento gráfico (como pruebas, multas, contratos, despidos), intégralo de inmediato en tu análisis legal.
5. Estructura tu respuesta de forma directa y contundente:
   - 🛡️ **Diagnóstico y Derechos Vulnerados:** (Norma constitucional o legal aplicable).
   - ⚡ **Estrategia Defensiva:** (Pasos tácticos a favor del usuario).
   - 📝 **Mecanismo Legal Sugerido:** (Ej. Tutela con base en qué artículo).
   - ⚠️ *Aviso Legal:* Orientación inteligente automatizada, no sustituye la representación judicial formal de un abogado titulado.
"""

def ejecutar_orquestador_groq(mensaje: str, file_path: str = None, mime_type: str = None):
    try:
        # Usamos un modelo multimodal de Groq si hay imágenes, o llama-3.3-70b-versatile para texto puro
        model_to_use = "meta-llama/llama-3.2-11b-vision-preview" if (file_path and mime_type and mime_type.startswith("image/")) else "llama-3.3-70b-versatile"

        if file_path and mime_type and mime_type.startswith("image/"):
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": mensaje or "Analiza esta evidencia gráfica para mi defensa legal."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                        }
                    ]
                }
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": mensaje}
            ]

        completion = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.3,
            max_tokens=2048
        )
        
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error ejecutando la consulta con la API de Groq: {str(e)}"
