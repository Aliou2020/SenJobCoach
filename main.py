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
    text = text.strip()

    # Trop court
    if len(text) < 6:
        return True

    words = text.split()

    # 1–2 mots non informatifs
    if len(words) < 2:
        return True

    # Peu de voyelles → bruit clavier
    vowels = re.findall(r"[aeiouyAEIOUY]", text)
    if len(vowels) < 2:
        return True

    # Trop peu de diversité
    if len(set(text)) < 6:
        return True

    return False



def detect_poste_strict(text: str) -> bool:
    if is_gibberish(text):
        return False

    # Minimum 4 mots
    if len(text.split()) < 4:
        return False

    # Doit contenir au moins UN mot clé fort
    return any(k in text for k in POSTE_KEYWORDS)



def detect_no_poste(text: str) -> bool:
    return any(k in text for k in NO_POSTE_KEYWORDS)

def detect_no_cv(text: str) -> bool:
    return any(k in text for k in NO_CV_KEYWORDS)

def detect_cv_strict(text: str) -> bool:
    return (
        len(text) > 800
        and ("experience" in text or "expérience" in text)
        and ("education" in text or "formation" in text)
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
    # 1️⃣ BRUIT / CLAVIER ALÉATOIRE → réponse humaine
    # =====================================================
    if is_gibberish(user_text):
        return AnalyzeResponse(
            response=(
                "😅 Je n’ai pas très bien compris ce message.\n\n"
                "Mais pas de souci — dites-moi simplement ce que vous avez en tête 🙂\n\n"
                "Par exemple :\n"
                "• chercher un emploi\n"
                "• améliorer un CV\n"
                "• discuter de votre parcours\n"
            ),
            session_id=session_id
        )

    # =====================================================
    # 2️⃣ SALUTATION → accueil humain
    # =====================================================
    if is_greeting(user_text):
        return AnalyzeResponse(
            response=(
                "Bonjour 👋\n\n"
                "Ravi de vous rencontrer !\n\n"
                "Je suis **SenJobCoach** et je peux vous aider à réfléchir à votre parcours, "
                "à améliorer votre CV ou simplement à discuter de vos projets.\n\n"
                "Qu’aimeriez-vous faire aujourd’hui ? 🙂"
            ),
            session_id=session_id
        )

    # =====================================================
    # 3️⃣ CAS HUMAINS (pas de poste / pas de CV)
    # =====================================================
    if detect_no_poste(user_text):
        return AnalyzeResponse(
            response=(
                "Merci pour votre honnêteté 🙏\n\n"
                "Ne pas avoir encore de poste précis est très courant.\n\n"
                "On peut commencer par discuter de :\n"
                "• votre domaine\n"
                "• vos expériences\n"
                "• ce que vous aimeriez faire à moyen terme\n\n"
                "Parlez-moi simplement de vous."
            ),
            session_id=session_id
        )

    if detect_no_cv(user_text):
        return AnalyzeResponse(
            response=(
                "Aucun souci 🙂\n\n"
                "Un CV n’a pas besoin d’être parfait pour commencer.\n\n"
                "Vous pouvez :\n"
                "• décrire vos expériences\n"
                "• partager un brouillon\n"
                "• ou simplement expliquer ce que vous voulez améliorer\n\n"
                "Je m’adapte."
            ),
            session_id=session_id
        )

    # =====================================================
    # 4️⃣ DÉTECTION FORTE SEULEMENT
    # =====================================================
    has_poste = detect_poste_strict(user_text)
    has_cv = detect_cv_strict(user_text)

    # =====================================================
    # 5️⃣ SI RIEN DE CLAIR → DISCUSSION LIBRE (IMPORTANT)
    # =====================================================
    if not has_poste and not has_cv:
        # 👉 ici on laisse ChatGPT répondre naturellement
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload.query}
        ]

        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7  # plus humain
        )

        return AnalyzeResponse(
            response=completion.choices[0].message.content,
            session_id=session_id
        )

    # =====================================================
    # 6️⃣ POSTE OK MAIS PAS DE CV
    # =====================================================
    if has_poste and not has_cv:
        return AnalyzeResponse(
            response=(
                "Parfait 👍\n\n"
                "Pour aller plus loin et vous donner une analyse utile, "
                "j’aurai besoin de **votre CV complet**.\n\n"
                "Dès que vous êtes prêt, copiez-collez-le ici."
            ),
            session_id=session_id
        )

    # =====================================================
    # 7️⃣ ANALYSE COMPLÈTE (SEULEMENT ICI)
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
