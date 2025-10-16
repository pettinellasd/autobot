import os
import re
import pandas as pd
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pathlib import Path

# Percorsi
BASE_DIR = Path(__file__).resolve().parent
AUTO_CSV_PATH = BASE_DIR / "auto_dati.csv"

# Env
load_dotenv()

# LLM Together AI (serverless)
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0.2,
    base_url="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY"),
)

# Carica CSV
df_auto = pd.read_csv(AUTO_CSV_PATH)

# Normalizzazione colonne minime richieste
for col in ["Marca", "Modello"]:
    if col not in df_auto.columns:
        raise ValueError(f"Manca la colonna obbligatoria '{col}' in auto_dati.csv")

# Pre-calcolo: lista modelli ordinati per lunghezza (decrescente) per evitare match parziali
MODELLI_ORDINATI = sorted(
    (str(m) for m in df_auto["Modello"].dropna().unique()),
    key=lambda s: len(s),
    reverse=True,
)

# Pre-calcolo: marche
MARCHE = sorted(str(m) for m in df_auto["Marca"].dropna().unique())

# --- Utility ---------------------------------------------------------------

def estrai_marca_modello(question: str):
    """Ritorna (marca, modello) se trovati, altrimenti (marca, None) o (None, None)."""
    q = question.lower()

    # 1) prova match modello (per primo, ordinati per lunghezza)
    for modello in MODELLI_ORDINATI:
        if _token_in(q, str(modello).lower()):
            # prendi la prima marca associata al modello
            sub = df_auto[df_auto["Modello"] == modello]
            if not sub.empty:
                marca = str(sub["Marca"].iloc[0])
                return marca, modello

    # 2) fallback: match marca
    for marca in MARCHE:
        if _token_in(q, marca.lower()):
            return marca, None

    return None, None


def estrai_due_modelli(question: str):
    """Ritorna ((marca1, modello1), (marca2, modello2)) se trova almeno 2 modelli nel testo."""
    q = question.lower()
    trovati = []
    for modello in MODELLI_ORDINATI:
        if _token_in(q, str(modello).lower()):
            trovati.append(modello)
        if len(trovati) >= 2:
            break

    if len(trovati) >= 2:
        m1 = trovati[0]
        m2 = trovati[1]
        marca1 = str(df_auto[df_auto["Modello"] == m1]["Marca"].iloc[0])
        marca2 = str(df_auto[df_auto["Modello"] == m2]["Marca"].iloc[0])
        return (marca1, m1), (marca2, m2)

    return None, None


def estrai_modelli(question: str, n=10):
    """Ritorna una lista di (marca, modello) trovati nel testo (fino a n)."""
    q = question.lower()
    trovati = []
    for modello in MODELLI_ORDINATI:
        if _token_in(q, str(modello).lower()) and modello not in trovati:
            marca = str(df_auto[df_auto["Modello"] == modello]["Marca"].iloc[0])
            trovati.append((marca, modello))
        if len(trovati) >= n:
            break
    return trovati

# --- Nodi ------------------------------------------------------------------

def car_info_node(state: dict):
    question = state["input"].lower()
    marca, modello = estrai_marca_modello(question)
    if not marca or not modello:
        return {**state, "output": "Non ho trovato la marca o il modello richiesto nel database."}

    df_modello = df_auto[(df_auto["Marca"] == marca) & (df_auto["Modello"] == modello)]

    # Mappa colonne/keyword (niente duplicati di chiavi)
    colonne = {
        "Versione": ["versioni", "quante versioni", "versione"],
        "Motorizzazione": ["motorizzazione", "motorizzazioni"],
        "Bagagliaio": ["bagagliaio", "capacità bagagliaio"],
        "Prezzo": ["prezzo", "prezzi", "costo"],
        "Autonomia km": ["autonomia", "autonomie"],
        "Capacità batteria kWh": ["batteria", "capacità batteria", "kwh"],
        "Lunghezza": ["lunghezza"],
        "Larghezza": ["larghezza"],
        "Altezza": ["altezza"],
        "Posti": ["posti", "quanti posti"],
        "Garanzia": ["garanzia"],
        "Cilindrata cm³": ["cilindrata"],
        "Cilindri": ["cilindri"],
        "Potenza CV/KW": ["potenza", "cv", "kw"],
        "Potenza termico CV/KW": ["potenza termico"],
        "Potenza omologata CV/KW": ["potenza omologata"],
        "Coppia Nm": ["coppia"],
        "Velocità max km/h": ["velocità", "velocità max"],
        "Consumo medio l/100 km": ["consumo", "consumo medio", "litri"],
        "Consumo medio kWh/100 km": ["consumo elettrico", "consumo kwh"],
        "Emissioni": ["emissioni"],
        "Peso kg": ["peso"],
    }

    risposte = []
    for col_csv, keywords in colonne.items():
        if col_csv not in df_auto.columns:
            continue
        for kw in keywords:
            if kw in question:
                valori = df_modello[col_csv].dropna().astype(str).unique().tolist()
                risposte.append(f"{col_csv}: {', '.join(valori) if valori else 'dato non disponibile'}")
                break

    # Contesto per il modello
    if not risposte:
        rows = []
        for _, row in df_modello.iterrows():
            info = ", ".join([f"{c}: {row[c]}" for c in df_auto.columns if pd.notna(row.get(c, None))])
            rows.append(info)
        context = f"{marca} {modello}:\n" + "\n".join(rows)
    else:
        context = f"{marca} {modello}:\n" + "\n".join(risposte)

    prompt = (
        "Sei CARBOT, assistente tecnico per auto cinesi in Italia.\n"
        "Regole di stile (OBBLIGATORIE):\n"
        "- Rispondi conciso e preciso (massimo 4 frasi).\n"
        "- NON fare proposte non richieste (niente: “posso anche…”, “vuoi che…”).\n"
        "- Niente tono pubblicitario. Solo dati presenti.\n"
        "- Se un dato manca, scrivi “nd”.\n"
        "- Usa elenchi brevi o frasi compatte; evita tabelle Markdown con barre verticali.\n"
        f"\nDati auto:\n{context}\n"
        f"\nDomanda utente: {state['input']}\n"
        "Risposta:"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "output": response.content}


def generic_info(state: dict):
    q = state["input"]
    # se non abbiamo match, non inventiamo: spieghiamo cosa posso fare, in 2-3 frasi max
    marche = ", ".join(sorted(df_auto["Marca"].dropna().astype(str).unique())[:10])
    msg = (
        "Posso rispondere usando solo i dati presenti nel database (nessuna fonte esterna). "
        "Chiedimi di un modello specifico (es. 'BYD Seal prezzi') oppure elenchi come 'tutte le 7 posti' o 'segmento E'. "
        f"Marche presenti: {marche}."
    )
    return {**state, "output": msg}


def confronto_node(state: dict):
    question = state["input"].lower()
    modelli = estrai_modelli(question, n=2)  # Limita il confronto a 2 modelli
    if len(modelli) < 2:
        return {**state, "output": "Non ho trovato almeno due modelli da confrontare nella domanda."}

    def estrai_valore(df, col):
        v = df[col].dropna().astype(str).unique()
        return v[0] if len(v) else "nd"

    output = []
    for marca, modello in modelli:
        df_m = df_auto[(df_auto["Marca"] == marca) & (df_auto["Modello"] == modello)]
        dati = {
            "dimensioni": f"{estrai_valore(df_m, 'Lunghezza')} x {estrai_valore(df_m, 'Larghezza')} x {estrai_valore(df_m, 'Altezza')}",
            "bagagliaio": estrai_valore(df_m, "Bagagliaio"),
            "autonomia": estrai_valore(df_m, "Autonomia km"),
            "potenza": estrai_valore(df_m, "Potenza CV/KW"),
            "prezzo": estrai_valore(df_m, "Prezzo"),
        }
        output.append(
            f"{modello}:\n"
            f"• Dimensioni: {dati['dimensioni']}\n"
            f"• Bagagliaio: {dati['bagagliaio']}\n"
            f"• Autonomia: {dati['autonomia']}\n"  # <-- tolto " km"
            f"• Potenza: {dati['potenza']}\n"
            f"• Prezzo: {dati['prezzo']}"
        )
    risposta = "\n\n".join(output)
    return {**state, "output": risposta}


def lista_node(state: dict):
    """Risponde a richieste di elenco basate su CSV: posti / segmento."""
    q = state["input"].lower()

    # 1) filtri disponibili
    posti = None
    if "7 posti" in q or "sette posti" in q: posti = 7
    elif "2 posti" in q or "due posti" in q or "biposto" in q: posti = 2
    else:
        m = re.search(r"(\d+)\s*posti", q)
        if m:
            posti = int(m.group(1))

    segmento = None
    mseg = re.search(r"segmento\s*([a-h])\b", q)
    if mseg:
        segmento = mseg.group(1).upper()

    df = df_auto.copy()

    # 2) applica filtri
    if posti is not None and "Posti" in df.columns:
        df["_posti"] = df["Posti"].map(_posti_max)
        df = df[df["_posti"].fillna(0) >= posti]

    if segmento is not None and "Segmento" in df.columns:
        df = df[df["Segmento"].astype(str).str.upper() == segmento]

    if df.empty:
        return {**state, "output": "Nel database non risultano modelli che soddisfano i criteri richiesti."}

    # 3) format conciso: Marca Modello — (posti, eventuale prezzo)
    items = []
    for _, r in df[["Marca","Modello","Posti","Prezzo"]].dropna(subset=["Marca","Modello"]).drop_duplicates().head(20).iterrows():
        pmax = _posti_max(r.get("Posti"))
        price = str(r.get("Prezzo")) if pd.notna(r.get("Prezzo")) else None
        tail = []
        if pmax: tail.append(f"{pmax} posti")
        if price: tail.append(f"€{price}")
        items.append(f"- {r['Marca']} {r['Modello']}" + (f" — {', '.join(tail)}" if tail else ""))

    return {**state, "output": "\n".join(items)}


def detect_intent(state: dict):
    question = state["input"].lower()
    # Intercetta subito richieste di liste
    if any(k in question for k in [" posti", "biposto", "segmento "]):
        return {**state, "next": "lista_node"}
    if any(x in question for x in [" vs ", "vs ", " meglio ", "confronto"]):
        m1, m2 = estrai_due_modelli(question)
        if m1 and m2:
            return {**state, "next": "confronto_node"}
    marca, modello = estrai_marca_modello(question)
    if marca and modello:
        return {**state, "next": "car_info_node"}
    else:
        return {**state, "next": "generic_info"}

# --- Grafo -----------------------------------------------------------------

graph = StateGraph(dict)
graph.add_node("car_info_node", car_info_node)
graph.add_node("generic_info", generic_info)
graph.add_node("detect_intent", detect_intent)
graph.add_node("confronto_node", confronto_node)
graph.add_node("lista_node", lista_node)

graph.set_entry_point("detect_intent")
graph.add_conditional_edges(
    "detect_intent",
    lambda state: state["next"],
    {
        "car_info_node": "car_info_node",
        "generic_info": "generic_info",
        "confronto_node": "confronto_node",
        "lista_node": "lista_node",
    },
)
graph.add_edge("car_info_node", END)
graph.add_edge("generic_info", END)
graph.add_edge("confronto_node", END)
graph.add_edge("lista_node", END)

compiled_graph = graph.compile()

# Helper robusti ------------------------------------------------------------

def _token_in(q: str, token: str) -> bool:
    """True se token compare come 'parola intera' (evita match parziali tipo 6 ∈ S60)."""
    pattern = rf'(?<![\w\d]){re.escape(token.lower())}(?![\w\d])'
    return re.search(pattern, q) is not None

def _posti_max(v) -> int | None:
    """Estrae il numero di posti massimo da valori tipo '7', '7/6', ecc."""
    nums = re.findall(r"\d+", str(v))
    return max(int(n) for n in nums) if nums else None

if __name__ == "__main__":
    question = input("Fai una domanda sulle auto cinesi: ")
    result = compiled_graph.invoke({"input": question})
    print("Autobot:", result["output"])