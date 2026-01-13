from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from typing import Optional
import openai
import uuid
import json
import os
import io
import re
import tempfile
import pdfplumber
import docx
from dotenv import load_dotenv

PDF_STORE = {} 

# ------------------------
# INIT
# ------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# SYSTEM PROMPTS
# ------------------------
SYSTEM_PROMPT = """
Tu es SenJobCoach, un coach carrière senior, humain, chaleureux et expert.

Tu sais accueillir, rassurer, guider et analyser intelligemment.

IMPORTANT :
- Raisonne en interne mais ne révèle jamais ton raisonnement.
- Adapte ton ton au message de l’utilisateur.
- Si le message est une simple discussion (salut, merci, doute), répond naturellement.
- Ne force jamais l’analyse.

Quand les conditions sont réunies (poste + CV complet), réalise une analyse professionnelle structurée.

Agis comme un expert RH et recruteur senior IT intervenant dans des tout type de contexte.

Analyse de manière approfondie l’adéquation entre le CV ci-dessous et
l’offre d’emploi ci-dessous.



Livrables attendus :
1. Score d’adéquation global (%) avec justification.
2. Analyse détaillée par dimension :
   - Responsabilités opérationnelles
   - Compétences techniques (réseaux, systèmes, sécurité, outils)
   - Expérience terrain / environnements critiques
   - Collaboration transverse et gouvernance
   - Soft skills et culture HSE
3. Tableaux comparatifs clairs pour chaque dimension.
4. Identification explicite :
   - des points forts différenciants
   - des écarts ou risques perçus par un recruteur
5. Recommandations concrètes :
   - ajustements du CV (phrases exactes à ajouter)
   - éléments à mettre en avant en entretien
6. Conclusion sous forme de note recruteur (go / no-go / go avec ajustements).

Quand tu dois afficher des données structurées :
- Utilise uniquement du HTML valide
- N’utilise PAS de Markdown
- Ajoute des titres <h4> 
- Reste clair et lisible




Contraintes de forme :
- Réponse très structurée
- Titres numérotés
- Tableaux lisibles
- Ton neutre, professionnel, orienté décision.


Utilise un ton :
- Humain
- Bienveillant
- Professionnel
- Clair

Règles :
- Réponds naturellement aux messages simples.
- Si un CV est fourni, analyse-le sérieusement.
- Structure clairement les réponses longues.
- Ne révèle jamais ton raisonnement interne.
"""

CV_ANALYSIS_PROMPT = """
Analyse le CV Si lui  seul est present.

Produis une réponse structurée avec :

1. Résumé du profil
2. Niveau de séniorité estimé
3. Expériences clés
4. Compétences techniques
5. Compétences comportementales
6. Points forts
7. Axes d’amélioration
8. Score global du CV sur 100 (avec justification courte)
9. Un tableau d’évaluation avec :
   - les grandes dimensions du poste
   - le poids de chaque dimension (%)
   - mon niveau d’adéquation
   - un score chiffré par dimension sur 10
   - Un score global sur 100
   - Un court verdict recruteur (shortlist / risque / points forts)
   - Des recommandations concrètes pour améliorer mon CV et atteindre +90/100
   - Présente le récapitulatif sous forme de table clair et lisible.
   - Adopte un ton professionnel, direct et orienté décision.
10. Pour chaque score, représenter visuellement la valeur à l’aide
d’une barre ASCII de longueur fixe (ex : 12 ou 20 caractères),
avec :
- █ en couleur verte pour la partie remplie
- ░ pour la partie vide
- le pourcentage affiché à droite.

"""

# ------------------------
# HELPERS
# ------------------------
def extract_cv_text(file: UploadFile) -> str:
    """
    Lecture réelle PDF & DOCX
    """
    text = ""

    filename = file.filename.lower()
    content = file.file.read()

    if filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text.strip()


def is_gibberish(text: str) -> bool:
    text = text.strip()
    if len(text) < 5:
        return True
    vowels = re.findall(r"[aeiouyAEIOUY]", text)
    return len(vowels) < 2

def generate_pdf(analysis_text: str, session_id: str) -> str:
    """
    Génère un PDF simple à partir du texte IA
    Retourne le chemin du fichier
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp_file.name

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Analyse de CV – SenJobCoach</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Session :</b> {session_id}", styles["Normal"]))
    story.append(Spacer(1, 12))

    for line in analysis_text.split("\n"):
        story.append(Paragraph(line.replace("&", "&amp;"), styles["Normal"]))
        story.append(Spacer(1, 6))

    doc.build(story)

    return pdf_path


# ------------------------
# ENDPOINT PRINCIPAL
# ------------------------
@app.post("/analyze")
async def analyze(
    query: str = Form(""),
    history: str = Form("[]"),
    session_id: Optional[str] = Form(None),
    cv: Optional[UploadFile] = File(None)
):
    session_id = session_id or str(uuid.uuid4())
    analysis_ready = False

    try:
        history_messages = json.loads(history)
    except Exception:
        history_messages = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history_messages:
        if "role" in msg and "content" in msg:
            messages.append(msg)

    # =========================
    # CV UPLOAD
    # =========================
    if cv:
        cv_text = extract_cv_text(cv)

        if not cv_text or len(cv_text) < 200:
            return {
                "response": (
                    "⚠️ Je n’ai pas réussi à lire correctement le CV.\n\n"
                    "Merci d’essayer avec un fichier PDF ou Word bien lisible."
                ),
                "session_id": session_id,
                "analysis_ready": False
            }
            
        analysis_ready = True
        messages.append({
            "role": "system",
            "content": CV_ANALYSIS_PROMPT + cv_text
        })

        messages.append({
            "role": "user",
            "content": query or "Analyse complète de ce CV"
        })

    # =========================
    # MESSAGE TEXTE
    # =========================
    else:
        if is_gibberish(query):
            return {
                "response": (
                    "🙂 Je n’ai pas bien compris.\n\n"
                    "Vous pouvez :\n"
                    "• poser une question\n"
                    "• uploader votre CV\n"
                    "• parler de votre projet professionnel"
                ),
                "session_id": session_id
            }

        messages.append({
            "role": "user",
            "content": query or "Bonjour"
        })

    # =========================
    # OPENAI
    # =========================
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4
        )

        response_text = completion.choices[0].message.content
       
        

    except Exception:
        response_text = (
            "😕 Désolé, un problème technique est survenu.\n"
            "Merci de réessayer."
        )

    # =========================
    # PDF EXPORT (SI ANALYSE CV)
    # =========================
    pdf_path = None
    if cv:
        pdf_path = generate_pdf(response_text, session_id)
        PDF_STORE[session_id] = pdf_path

    return {
        "response": response_text,
        "session_id": session_id,
        "analysis_ready": analysis_ready,
        "pdf_available": bool(pdf_path)
    }




@app.get("/download-pdf/{session_id}")
async def download_pdf(session_id: str):
    pdf_path = PDF_STORE.get(session_id)

    if not pdf_path or not os.path.exists(pdf_path):
        return {"error": "PDF non trouvé"}

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"Analyse_CV_{session_id}.pdf"
    )

