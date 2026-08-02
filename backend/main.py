import os
import shutil
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from orchestrator import ejecutar_orquestador_groq

app = FastAPI(title="Asesor Legal IA - Groq Edition (Colombia)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/chat")
async def chat_legal_groq(
    mensaje: str = Form(""),
    archivo: UploadFile = File(None)
):
    file_path = None
    mime_type = None
    
    if archivo:
        file_path = os.path.join(UPLOAD_DIR, archivo.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(archivo.file, buffer)
        mime_type = archivo.content_type

    respuesta_ia = ejecutar_orquestador_groq(mensaje, file_path, mime_type)
    
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    return {
        "status": "success",
        "respuesta": respuesta_ia,
        "archivo_procesado": archivo.filename if archivo else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
