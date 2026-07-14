"""
test_rag_backend.py — Script de prueba del backend RAG
Ejecuta preguntas variadas para verificar que el sistema:
  1. Responde con información de los documentos (Docs/)
  2. Rechaza preguntas fuera de dominio
  3. Diferencia correctamente estructuras neuroanatómicas
"""

import os, sys, time

# Asegurar que estamos en el directorio correcto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from rag_pipeline import build_vector_store, consultar, PERSIST_DIR, COLLECTION_NAME

# ─── Colores para terminal ───
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def separador(titulo):
    print(f"\n{CYAN}{'═'*70}")
    print(f"  {titulo}")
    print(f"{'═'*70}{RESET}")

def main():
    separador("🧠 TEST DEL BACKEND RAG — NEUROANATOMÍA")

    # 1. Verificar que la base vectorial existe y cargarla
    print(f"\n{YELLOW}[1] Cargando base vectorial...{RESET}")
    try:
        vs = build_vector_store(force_rebuild=False)
        total_vectores = vs._collection.count()
        print(f"{GREEN}   ✔ Base cargada: {total_vectores} vectores en '{COLLECTION_NAME}'{RESET}")
    except Exception as e:
        print(f"{RED}   ✗ ERROR cargando base vectorial: {e}{RESET}")
        print(f"{RED}   Asegúrate de que la base fue construida previamente.{RESET}")
        return

    # 2. Verificar los documentos indexados
    print(f"\n{YELLOW}[2] Documentos indexados:{RESET}")
    try:
        all_data = vs._collection.get(include=["metadatas"])
        sources = set()
        for meta in all_data["metadatas"]:
            if meta and "source" in meta:
                sources.add(os.path.basename(meta["source"]))
        for src in sorted(sources):
            print(f"   📄 {src}")
        print(f"   Total fuentes únicas: {len(sources)}")
    except Exception as e:
        print(f"{RED}   ✗ Error leyendo metadatos: {e}{RESET}")

    # 3. Test de recuperación vectorial (sin LLM, solo búsqueda)
    separador("📡 TEST DE RECUPERACIÓN VECTORIAL (sin LLM)")
    queries_retrieval = [
        "cerebro",
        "cerebelo",
        "hipotálamo",
        "meninges",
        "lóbulo frontal",
    ]
    for q in queries_retrieval:
        print(f"\n   🔎 Query: '{q}'")
        try:
            docs = vs.similarity_search(q, k=2)
            for i, doc in enumerate(docs):
                fuente = os.path.basename(doc.metadata.get("source", "?"))
                pagina = doc.metadata.get("page", "?")
                snippet = doc.page_content[:120].replace("\n", " ")
                print(f"      [{i+1}] {fuente} p.{pagina}: \"{snippet}...\"")
        except Exception as e:
            print(f"      {RED}✗ Error: {e}{RESET}")

    # 4. Test completo con LLM (preguntas que SÍ deben responderse)
    separador("🤖 TEST COMPLETO RAG (preguntas dentro del dominio)")
    preguntas_validas = [
        "¿Qué es el cerebro y cuáles son sus principales componentes?",
        "¿Cuál es la función del cerebelo?",
        "¿Qué estructuras forman el tronco encefálico?",
    ]

    for pregunta in preguntas_validas:
        print(f"\n{BOLD}❓ {pregunta}{RESET}")
        t0 = time.time()
        try:
            resultado = consultar(pregunta, vs, k=3, nivel="avanzado")
            elapsed = time.time() - t0
            print(f"{GREEN}🤖 Respuesta ({elapsed:.1f}s):{RESET}")
            # Mostrar primeros 500 caracteres de la respuesta
            resp = resultado["respuesta"]
            if len(resp) > 500:
                print(f"   {resp[:500]}...")
            else:
                print(f"   {resp}")
            print(f"   {YELLOW}[{len(resultado['fragmentos'])} fragmentos | ~{resultado['tokens_contexto_aprox']} tokens contexto]{RESET}")
            # Mostrar fuentes usadas
            for frag in resultado["fragmentos"]:
                src = os.path.basename(frag.metadata.get("source", "?"))
                pg = frag.metadata.get("page", "?")
                print(f"   📎 {src}, pág. {pg}")
        except Exception as e:
            print(f"   {RED}✗ Error: {e}{RESET}")

    # 5. Test de preguntas FUERA DE DOMINIO (debería rechazar o limitar)
    separador("🚫 TEST DE PREGUNTAS FUERA DE DOMINIO")
    preguntas_fuera = [
        "¿Cuál es la capital de Francia?",
        "¿Cómo se programa en Python?",
    ]

    for pregunta in preguntas_fuera:
        print(f"\n{BOLD}❓ {pregunta}{RESET}")
        t0 = time.time()
        try:
            resultado = consultar(pregunta, vs, k=3, nivel="avanzado")
            elapsed = time.time() - t0
            resp = resultado["respuesta"]
            print(f"{YELLOW}🤖 Respuesta ({elapsed:.1f}s):{RESET}")
            if len(resp) > 400:
                print(f"   {resp[:400]}...")
            else:
                print(f"   {resp}")
        except Exception as e:
            print(f"   {RED}✗ Error: {e}{RESET}")

    separador("✅ TEST COMPLETO")
    print(f"\n{GREEN}Todas las pruebas ejecutadas.{RESET}\n")

if __name__ == "__main__":
    main()
