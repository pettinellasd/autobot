import os
import re
import pandas as pd
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = ChatOpenAI(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.6,
    base_url="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY")
)

# Carica il CSV delle auto cinesi
AUTO_CSV_PATH = "autobot_env/auto_dati.csv"
df_auto = pd.read_csv(AUTO_CSV_PATH)

# Utility: trova marca e modello nella domanda
def estrai_marca_modello(question):
    question = question.lower()
    # Ordina i modelli per lunghezza decrescente per evitare match parziali
    modelli_ordinati = sorted(df_auto['Modello'].unique(), key=lambda x: -len(x))
    for modello in modelli_ordinati:
        if modello.lower() in question:
            # Prendi la prima marca associata
            marca = df_auto[df_auto['Modello'] == modello]['Marca'].iloc[0]
            return marca, modello
    # Fallback: cerca la marca
    for marca in df_auto['Marca'].unique():
        if marca.lower() in question:
            return marca, None
    return None, None

def estrai_due_modelli(question):
    question = question.lower()
    modelli_ordinati = sorted(df_auto['Modello'].unique(), key=lambda x: -len(x))
    trovati = []
    for modello in modelli_ordinati:
        if modello.lower() in question:
            trovati.append(modello)
    if len(trovati) >= 2:
        # Prendi le marche associate
        marca1 = df_auto[df_auto['Modello'] == trovati[0]]['Marca'].iloc[0]
        marca2 = df_auto[df_auto['Modello'] == trovati[1]]['Marca'].iloc[0]
        return (marca1, trovati[0]), (marca2, trovati[1])
    return None, None

# Nodo principale: risponde a domande su tutte le colonne del CSV
def car_info_node(state):
    question = state["input"].lower()
    marca, modello = estrai_marca_modello(question)
    if not marca or not modello:
        return {**state, "output": "Non ho trovato la marca o il modello richiesto nel database."}

    df_modello = df_auto[(df_auto['Marca'] == marca) & (df_auto['Modello'] == modello)]

    # Mappa colonne e varianti di domanda
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
        "Garanzia": ["garanzia"],
    }

    risposte = []
    for col_csv, keywords in colonne.items():
        for kw in keywords:
            if kw in question:
                valori = df_modello[col_csv].unique()
                valori = [str(v) for v in valori if pd.notna(v)]
                if len(valori) > 0:
                    risposte.append(f"{col_csv}: {', '.join(valori)}")
                else:
                    risposte.append(f"{col_csv}: dato non disponibile")
                break

    # Prepara il contesto da passare al modello
    if not risposte:
        rows = []
        for _, row in df_modello.iterrows():
            info = ", ".join([f"{c}: {row[c]}" for c in df_auto.columns if pd.notna(row[c])])
            rows.append(info)
        context = f"{marca} {modello}:\n" + "\n".join(rows)
    else:
        context = f"{marca} {modello}:\n" + "\n".join(risposte)

    # Prompt creativo per il modello
    prompt = (
        "Sei Autobot, un assistente esperto di auto cinesi per il mercato italiano. "
        "Rispondi a domande semplici e poco dettagliate, aiutando l'utente a capire le differenze tra versioni, motorizzazioni, prezzi, autonomia, ecc. "
        "Usa solo i dati forniti qui sotto, non inventare. "
        "Evita toni pubblicitari, sii concreto, neutro e utile per chi sta valutando l'acquisto di un'auto cinese in Italia. "
        "Se la domanda è generica, descrivi brevemente il modello e le sue versioni principali. "
        "Se la domanda è di confronto, evidenzia le differenze pratiche. "
        "Se mancano dati, dillo chiaramente. "
        f"\n\nDati auto:\n{context}\n"
        f"\nDomanda utente: {state['input']}\n"
        "Rispondi in massimo 4 frasi."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "output": response.content}

# Nodo per domande generiche o saluti
def generic_info(state):
    question = state["input"]
    prompt = (
        "Sei Autobot, un assistente specializzato nel mercato delle auto cinesi in Italia. "
        "Puoi chiedermi informazioni su versioni, motorizzazioni, prezzi, autonomia, bagagliaio, batteria, dimensioni, potenza, emissioni, ecc. di ogni modello presente nel database. "
        f"Domanda utente: {question}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "output": response.content}

def confronto_node(state):
    question = state["input"].lower()
    m1, m2 = estrai_due_modelli(question)
    if not m1 or not m2:
        return {**state, "output": "Non ho trovato due modelli da confrontare nella domanda."}
    df_m1 = df_auto[(df_auto['Marca'] == m1[0]) & (df_auto['Modello'] == m1[1])]
    df_m2 = df_auto[(df_auto['Marca'] == m2[0]) & (df_auto['Modello'] == m2[1])]

    # Scegli le colonne principali da confrontare
    colonne_confronto = [
        "Versione", "Prezzo", "Motorizzazione", "Bagagliaio", "Autonomia km", "Capacità batteria kWh",
        "Potenza CV/KW", "Coppia Nm", "Velocità max km/h", "Consumo medio kWh/100 km", "Emissioni"
    ]

    # Prepara le righe per ogni versione
    def prepara_righe(df):
        righe = []
        for _, row in df.iterrows():
            riga = [str(row.get(col, "")) for col in colonne_confronto]
            righe.append(riga)
        return righe

    righe_m1 = prepara_righe(df_m1)
    righe_m2 = prepara_righe(df_m2)

    # Costruisci la tabella markdown
    header = "| Modello | " + " | ".join(colonne_confronto) + " |\n"
    separator = "|---" * (len(colonne_confronto)+1) + "|\n"
    tabella = header + separator

    for riga in righe_m1:
        tabella += f"| {m1[1]} | " + " | ".join(riga) + " |\n"
    for riga in righe_m2:
        tabella += f"| {m2[1]} | " + " | ".join(riga) + " |\n"

    risposta = f"**Confronto tra {m1[1]} e {m2[1]}:**\n\n{tabella}"
    return {**state, "output": risposta}

# Intent detection aggiornato
def detect_intent(state):
    question = state["input"].lower()
    if "vs" in question or "meglio" in question or "confronto" in question:
        m1, m2 = estrai_due_modelli(question)
        if m1 and m2:
            return {**state, "next": "confronto_node"}
    marca, modello = estrai_marca_modello(question)
    if marca and modello:
        return {**state, "next": "car_info_node"}
    else:
        return {**state, "next": "generic_info"}

# Crea grafo
graph = StateGraph(dict)
graph.add_node("car_info_node", car_info_node)
graph.add_node("generic_info", generic_info)
graph.add_node("detect_intent", detect_intent)
graph.add_node("confronto_node", confronto_node)
graph.set_entry_point("detect_intent")
graph.add_conditional_edges(
    "detect_intent",
    lambda state: state["next"],
    {
        "car_info_node": "car_info_node",
        "generic_info": "generic_info",
        "confronto_node": "confronto_node"
    }
)
graph.add_edge("car_info_node", END)
graph.add_edge("generic_info", END)
graph.add_edge("confronto_node", END)
compiled_graph = graph.compile()

# Esegui
if __name__ == "__main__":
    question = input("Fai una domanda sulle auto cinesi: ")
    result = compiled_graph.invoke({"input": question})
    print("Autobot:", result["output"])