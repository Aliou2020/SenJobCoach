from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import openai
import uuid
import re
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS pour autoriser ton site PHP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------
# MODELS
# ------------------------

class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str

class AnalyzeRequest(BaseModel):
    query: str
    history: List[Message] = []
    session_id: Optional[str] = None

class AnalyzeResponse(BaseModel):
    response: str
    session_id: str
# ------------------------
# SYSTEM PROMPT
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

Structure obligatoire de l’analyse :
1. Résumé du profil
2. Niveau de séniorité estimé
3. Score global du CV (0–100)
4. Score de compatibilité avec le poste (0–100)
5. Compétences techniques
6. Compétences comportementales
7. Points forts
8. Axes d’amélioration
9. Recommandations personnalisées

Utilise un ton :
- Humain
- Bienveillant
- Professionnel
- Clair
"""

# ------------------------
# DETECTION LOGIC
# ------------------------

POSTE_KEYWORDS = [
    # Termes generaux
    "poste", "job", "emploi", "position", "intitule", "fonction", "role",

    # Description de poste
    "description de poste", "fiche de poste", "jd", "job description",
    "offre", "offre d emploi", "annonce", "annonce d emploi",

    # Intention de candidature
    "je vise", "je postule", "je candidate", "je veux postuler",
    "je souhaite postuler", "je cherche un poste",
    "je cherche un emploi", "je recherche un emploi",

    # Actions liees au poste
    "candidature", "postuler", "postulation", "deposer ma candidature",
    "soumettre ma candidature", "envoyer ma candidature",

    # Ciblage du poste
    "poste vise", "job vise", "poste cible", "position cible",
    "poste souhaite", "job souhaite",

    # Expressions conversationnelles
    "pour ce poste", "pour ce job", "pour cette position",
    "par rapport au poste", "par rapport au job",

    # Contexte recrutement
    "recrutement", "processus de recrutement",
    "selection", "shortlist", "profil recherche",

    # Variantes / fautes courantes
    "post", "jobe", "emplois", "jobb",
    "je postule a", "je vise un poste", "jd poste"
]

HELLO_KEYWORDS = [
    # Salutations simples
    "salut", "bonjour", "bonsoir", "coucou", "hello", "hi", "hey",
    "allo", "allô", "bon matin", "bon apres midi", "bonne journee",

    # Salutations polies / professionnelles
    "bonjour monsieur", "bonjour madame", "bonjour a vous", "bonsoir a vous",
    "enchante", "ravi de vous rencontrer", "au plaisir de vous lire",
    "cordialement",

    # Salutations informelles / amicales
    "salut tout le monde", "hey salut", "coucou toi", "wesh", "yo",
    "ca dit quoi", "quoi de neuf",

    # Démarrage de discussion
    "comment ca va", "comment allez vous", "comment vas tu",
    "ca va", "tu vas bien", "tout va bien",
    "comment se passe ta journee",

    # Réponses courantes
    "ca va bien", "tres bien merci", "pas mal", "comme ci comme ca",
    "tranquillement", "on fait aller", "ca va et toi",

    # Relances
    "et toi", "et vous", "des nouvelles", "quoi de nouveau",
    "tu fais quoi", "que puis je faire pour toi",
    "comment puis je aider", "de quoi veux tu parler",

    # Politesse / interaction
    "merci", "merci beaucoup", "je vous remercie",
    "s il te plait", "s il vous plait", "avec plaisir",
    "pas de souci", "aucun probleme", "d accord", "tres bien",

    # Clôture
    "au revoir", "a bientot", "a plus tard", "bonne soiree",
    "a tout a l heure", "a la prochaine", "merci et a bientot",

    # Expressions courtes fréquentes
    "peut etre", "bien sur",
    "je comprends", "compris", "interessant", "c est clair",

    # Variantes / fautes courantes (chat)
    "bonjourr", "slt", "bjr", "bsr", "salu",
     "sava", "commen sa va", "comen tu va"
]


NO_POSTE_KEYWORDS = [
    # Absence explicite de poste
    "pas de poste", "aucun poste", "pas encore de poste",
    "je n ai pas de poste", "je n ai pas encore de poste",
    "je ne vise aucun poste",

    # Incertitude / hésitation
    "je ne sais pas quel poste", "je ne sais pas quel job",
    "je ne sais pas quoi viser", "je ne sais pas encore",
    "je ne sais pas quel emploi",

    # Recherche generale
    "je cherche un job", "je cherche un emploi",
    "je cherche du travail", "je suis en recherche d emploi",
    "je suis a la recherche d un emploi",

    # Ouverture / exploration
    "je suis ouvert", "je suis ouvert a tout",
    "tous types de postes", "tout type de job",
    "peu importe le poste",

    # Expressions conversationnelles
    "pas pour le moment", "pas encore decide",
    "pas defini", "non defini",

    # Variantes / fautes courantes
    "pas poste", "aucun job", "no job",
    "je sai pas", "je sais pa encore"
]

NO_CV_KEYWORDS = [
    # Absence explicite de CV
    "pas de cv", "je n ai pas de cv", "je n ai aucun cv",
    "pas encore de cv", "je n ai pas encore de cv",

    # CV non pret / en cours
    "cv pas pret", "mon cv n est pas pret",
    "cv en cours", "cv en preparation",
    "je travaille sur mon cv",

    # Oubli / indisponibilite
    "je n ai pas mon cv", "je n ai pas mon cv sur moi",
    "je ne retrouve pas mon cv", "cv indisponible",

    # Incertitude / hesitation
    "je ne sais pas si mon cv est pret",
    "mon cv n est pas a jour", "cv pas a jour",

    # Ouverture / alternatives
    "je peux le faire plus tard",
    "plus tard pour le cv",
    "je ferai le cv apres",

    # Expressions conversationnelles
    "pas maintenant", "pas pour le moment",

    # Variantes / fautes courantes (chat)
    "pa de cv", "pas cv", "cv pa pret",
    "j ai pa de cv", "g pa de cv"
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def is_greeting(text: str) -> bool:
    return any(text.startswith(k) for k in HELLO_KEYWORDS)

    
def detect_intent(text: str) -> str:
    if detect_no_cv(text):
        return "NO_CV"
    if detect_no_poste(text):
        return "NO_POSTE"
    if detect_poste(text):
        return "POSTE"
    if is_greeting(text):
        return "HELLO"
    return "UNKNOWN"


def is_gibberish(text: str) -> bool:
    # Trop court
    if len(text) < 5:
        return True

    # Un seul "mot" bizarre
    if len(text.split()) == 1 and len(text) < 6:
        return True

    # Peu de voyelles → souvent du bruit
    vowels = re.findall(r"[aeiouy]", text)
    if len(vowels) < 2:
        return True

    # Trop peu de diversité de caractères
    if len(set(text)) < 4:
        return True

    return False


def detect_poste(text: str) -> bool:
    return any(k in text for k in POSTE_KEYWORDS)

def detect_no_poste(text: str) -> bool:
    return any(k in text for k in NO_POSTE_KEYWORDS)

def detect_no_cv(text: str) -> bool:
    return any(k in text for k in NO_CV_KEYWORDS)

def detect_cv(text: str) -> bool:
    return (
        ("experience" in text or "expérience" in text)
        and ("education" in text or "formation" in text)
        and len(text) > 800
    )

def merge_text(history: List[Message], latest: str) -> str:
    return " ".join([m.content for m in history] + [latest])

# ------------------------
# ENDPOINT
# ------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):

    session_id = payload.session_id or str(uuid.uuid4())
    user_text = normalize(payload.query)

    # =====================================================
    # 1️⃣ BRUIT / NON-SENS — PRIORITÉ ABSOLUE
    # =====================================================
    if is_gibberish(user_text):
        return AnalyzeResponse(
            response=(
                "Désolé, je n’ai pas compris votre message 🙂\n\n"
                "👉 Vous pouvez par exemple écrire :\n"
                "• « Je veux analyser mon CV »\n"
                "• « Je cherche un emploi »\n"
                "• « Je ne sais pas encore quel poste viser »"
            ),
            session_id=session_id
        )

    # =====================================================
    # 2️⃣ DÉTECTION D’INTENTION (MESSAGE ACTUEL)
    # =====================================================
    intent = detect_intent(user_text)

    # =====================================================
    # 3️⃣ PAS DE CV
    # =====================================================
    if intent == "NO_CV":
        return AnalyzeResponse(
            response=(
                "Pas de souci 👍\n\n"
                "Nous pouvons avancer même sans CV finalisé.\n\n"
                "👉 Vous pouvez :\n"
                "• décrire vos expériences\n"
                "• partager un CV brouillon\n"
                "• ou me dire ce que vous souhaitez améliorer\n\n"
                "Que préférez-vous faire ?"
            ),
            session_id=session_id
        )

    # =====================================================
    # 4️⃣ PAS DE POSTE
    # =====================================================
    if intent == "NO_POSTE":
        return AnalyzeResponse(
            response=(
                "C’est tout à fait normal de ne pas avoir encore un poste précis 🙏\n\n"
                "Pour vous orienter efficacement, dites-moi :\n"
                "• votre domaine (data, IT, finance, humanitaire…)\n"
                "• votre niveau (junior, confirmé, senior)\n"
                "• vos objectifs professionnels\n\n"
                "Expliquez-moi simplement votre situation."
            ),
            session_id=session_id
        )

    # =====================================================
    # 5️⃣ SALUTATION (APRÈS INTENTIONS)
    # =====================================================
    if intent == "HELLO":
        return AnalyzeResponse(
            response=(
                "Bonjour 👋\n\n"
                "Je suis **SenJobCoach**, votre coach carrière.\n\n"
                "Je peux vous aider à :\n"
                "• analyser votre CV\n"
                "• l’adapter à un poste précis\n"
                "• clarifier votre positionnement professionnel\n\n"
                "Que souhaitez-vous faire ? 🙂"
            ),
            session_id=session_id
        )

    # =====================================================
    # 6️⃣ DÉTECTION POSTE / CV
    # =====================================================
    has_poste_now = detect_poste(user_text)
    has_cv_now = detect_cv(user_text)

    if has_poste_now and len(user_text.split()) < 4:
        has_poste_now = False

    # =====================================================
    # 7️⃣ RIEN DE CLAIR → GUIDAGE
    # =====================================================
    if not has_poste_now and not has_cv_now:
        return AnalyzeResponse(
            response=(
                "Pour bien commencer 🎯\n\n"
                "Quel est **le poste ou le domaine professionnel que vous visez** ?\n\n"
                "Vous pouvez aussi coller une fiche de poste (Job Description)."
            ),
            session_id=session_id
        )

    # =====================================================
    # 8️⃣ POSTE OK MAIS PAS DE CV
    # =====================================================
    if has_poste_now and not has_cv_now:
        return AnalyzeResponse(
            response=(
                "Parfait, merci pour le poste ciblé ✅\n\n"
                "Pour continuer, j’ai maintenant besoin de **votre CV complet**.\n\n"
                "👉 Copiez-collez l’ensemble du CV (expérience, formation, compétences)."
            ),
            session_id=session_id
        )

    # =====================================================
    # 9️⃣ ANALYSE IA (SEULEMENT ICI)
    # =====================================================
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in payload.history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": payload.query})

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )

    return AnalyzeResponse(
        response=completion.choices[0].message.content,
        session_id=session_id
    )
