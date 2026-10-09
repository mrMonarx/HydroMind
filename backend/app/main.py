from fastapi import FastAPI

app = FastAPI(
    title="HydroMind Agro API",
    description="Sug'orish ma'lumotlari va tavsiyalar uchun backend",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Server ishlayotganini tekshirish uchun."""
    return {"status": "ok", "project": "HydroMind Agro", "version": app.version}
