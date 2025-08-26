import fitz  # ancien nom de PyMuPDF
from openai import AzureOpenAI
import time
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer # permet de transformer le texte en vecteurs
from sklearn.metrics.pairwise import cosine_similarity # permet de calculer la similarité cosinus entre les vecteurs
from dotenv import load_dotenv
import io # permet de manipuler des flux de données binaires : convertit l'image en octets pour l'envoyer à GPT-4 Vision
import base64 # permet de transformer l'image en base64 que GPT-4 Vision peut lire
from PIL import Image # permet d'ouvrir, manipuler et enregistrer des images sous différents formats (.png, .jpg, etc.)
from nltk.corpus import stopwords # pour les stopwords en français
import psutil  # pour vérifier les instances multiples du script
import difflib  # pour comparer les réponses de l'agent OCR
import sys
import hashlib
from datetime import datetime
from email.message import EmailMessage # pour créer des emails (pas utilisé car pas accès à l'API Microsoft)
from collections import defaultdict # pour créer un dictionnaire avec des valeurs par défaut
import re  # pour les expressions régulières
import textwrap  # pour formater le texte dans les PDFs dans la fonction summary_pdf() du mode 5

french_stop_words = stopwords.words('french') # Charge les stopwords français afin de les utiliser dans le TfidfVectorizer de find_relevant_pages

load_dotenv()

pdf_cache = {}

CACHE_FOLDER = "image_descriptions_cacheV3"
os.makedirs(CACHE_FOLDER, exist_ok=True)

# Clé API OpenAI
client = AzureOpenAI(
    api_key=os.getenv("AZUREOPENAI_API_KEY"),
    api_version='2024-08-01-preview',
    azure_endpoint=os.getenv("AZUREOPENAI_API_URL"),
    azure_deployment='gpt4o-2024-08-06'
)

# Fichier mémoire de conversation
MEMORY_FILE = "memoire_conversationV3.json"

MAIL_TEMPLATE = {
    "subject": "Résultats de la recherche sur les mots-clés : {keywords}",
    "body": (
        "Bonjour,\n\n"
        "Veuillez trouver ci-joint le document PDF contenant les résultats de votre recherche sur les mots-clés suivants : {keywords}.\n\n"
        "Résumé :\n{summary}\n\n"
        "Cordialement,\nL'équipe Parser Agent"
    ),
    "to": "destinataire@example.com"
}


# Fonction pour générer un email avec les résultats de la recherche
def generate_email(keywords, summary, pdf_path, template=MAIL_TEMPLATE):
    formatted_keywords = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)

    msg = EmailMessage()
    msg['Subject'] = template['subject'].format(keywords=formatted_keywords)
    msg['From'] = "parser-agent@example.com"
    msg['To'] = template['to']
    msg.set_content(template['body'].format(keywords=formatted_keywords, summary=summary))

    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
        pdf_name = os.path.basename(pdf_path)
        msg.add_attachment(pdf_data, maintype='application', subtype='pdf', filename=pdf_name)

    return msg


# Fonction pour créer un PDF lisible à partir des lignes de texte
def create_readable_pdf(text_lines, output_path):
    PAGE_WIDTH = 595    # Largeur A4 en points
    PAGE_HEIGHT = 842   # Hauteur A4 en points
    MARGIN = 50         # Marge en points
    LINE_HEIGHT = 14    # Hauteur entre les lignes   
    FONT_SIZE = 11      # Taille de la police

    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = MARGIN

    for line in text_lines:
        while len(line) > 0:
            max_chars = int((PAGE_WIDTH - 2 * MARGIN) / (FONT_SIZE * 0.6))
            chunk = line[:max_chars]
            line = line[max_chars:]

            if y + LINE_HEIGHT > PAGE_HEIGHT - MARGIN:
                page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
                y = MARGIN

            page.insert_text((MARGIN, y), chunk, fontsize=FONT_SIZE)
            y += LINE_HEIGHT

    doc.save(output_path)


# Fonction pour vérifier si les nouvelles descriptions contiennent des éléments non présents dans le cache car auparavent les deux modes
# écrivaent chacun leur version des descriptions d'images dans le cache, ce qui pouvait entraîner des doublons
def should_update_cacheV0(existing_descriptions, new_descriptions, threshold=0.95):
    """Vérifie si les nouvelles descriptions contiennent des éléments non présents dans le cache."""
    for new_desc in new_descriptions:
        if not any(difflib.SequenceMatcher(None, new_desc, cached).ratio() > threshold for cached in existing_descriptions):
            return True
    return False


def should_update_cache(existing, new, threshold=0.95):
    return any(
        not any(difflib.SequenceMatcher(None, n, e).ratio() > threshold for e in existing)
        for n in new
    )


# Fonction pour sauvegarder le cache si nécessaire (ancienne version)
def save_cache_if_neededV0(cache_file, existing_descriptions, updated_descriptions):
    updated_deduplicated = deduplicate_similar_descriptions(updated_descriptions)
    if not os.path.exists(cache_file) or should_update_cache(existing_descriptions, updated_deduplicated):
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(updated_deduplicated, f, ensure_ascii=False, indent=2)
        print(f"✓ Cache mis à jour : {cache_file}")
    else:
        print(f"✓ Cache conservé sans modification : {cache_file}")


# Fonction pour sauvegarder le cache si nécessaire (version améliorée)
def save_cache_if_needed(cache_file, existing_descriptions, updated_descriptions):
    """
    Nettoie les descriptions redondantes et met à jour le cache si nécessaire.
    """
    updated_deduplicated = clean_redundant_descriptions(updated_descriptions)
    updated_deduplicated_cleaner = deduplicate_by_image_id(updated_deduplicated)
    if not os.path.exists(cache_file) or should_update_cache(existing_descriptions, updated_deduplicated_cleaner):
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(updated_deduplicated_cleaner, f, ensure_ascii=False, indent=2)
        print(f"✓ Cache mis à jour : {cache_file}")
    else:
        print(f"✓ Cache conservé sans modification : {cache_file}")


# Fonction pour supprimer les descriptions très similaires et optimiser la mémoire
def deduplicate_similar_descriptions(descriptions, threshold=0.95):
    """Supprime les descriptions très similaires en ne gardant qu'une seule par groupe."""
    unique = []
    for desc in descriptions:
        if not any(difflib.SequenceMatcher(None, desc, u).ratio() > threshold for u in unique):
            unique.append(desc)
    return unique


# Fonction pour supprimer les doublons par identifiant d'image
def deduplicate_by_image_id(descriptions):
    """
    Supprime les doublons en ne gardant qu'une seule description par identifiant [Image X].
    """
    seen = {}
    for desc in descriptions:
        match = re.match(r"(\[Image \d+\])", desc)
        if match:
            key = match.group(1)
            if key not in seen:
                seen[key] = desc
    return list(seen.values())


# Fonction pour nettoyer les descriptions redondantes (ancienne version)
def clean_redundant_descriptionsV0(descriptions, threshold=0.95):
    """
    Supprime les doublons textuels pour chaque identifiant d'image.
    Regroupe par [Image X] et conserve une seule description par image.
    """
    grouped = defaultdict(list)

    # Regrouper les descriptions par identifiant
    for desc in descriptions:
        if desc.startswith("[Image"):
            key = desc.split("]:")[0] + "]"
            grouped[key].append(desc)

    cleaned = []

    for key, group in grouped.items():
        unique = []
        for desc in group:
            content = desc.split("]:", 1)[-1].strip()
            if not any(difflib.SequenceMatcher(None, content, u.split("]:", 1)[-1].strip()).ratio() > threshold for u in unique):
                unique.append(desc)
        cleaned.extend(unique)

    return cleaned


# Fonction pour normaliser le texte (supprimer ponctuation, espaces multiples, casse)
def normalize_text(text):
    # Supprime ponctuation, espaces multiples, casse
    text = re.sub(r'\W+', ' ', text)
    return text.lower().strip()


# Fonction pour nettoyer les descriptions redondantes (version améliorée)
def clean_redundant_descriptions(descriptions, threshold=0.90):
    grouped = defaultdict(list)

    for desc in descriptions:
        if desc.startswith("[Image"):
            key = desc.split("]:")[0] + "]"
            grouped[key].append(desc)

    cleaned = []

    for key, group in grouped.items():
        unique = []
        for desc in group:
            content = normalize_text(desc.split("]:", 1)[-1])
            if not any(difflib.SequenceMatcher(None, content, normalize_text(u.split("]:", 1)[-1])).ratio() > threshold for u in unique):
                unique.append(desc)
        cleaned.extend(unique)

    return cleaned


# Fonction pour obtenir le hash d'une image
def get_image_hash(image: Image.Image) -> str:
    """Retourne un hash unique de l'image pour identifier les doublons."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return hashlib.md5(buffered.getvalue()).hexdigest()


# Fonction pour déterminer si une description d'image est inutile (image noire, vide, sans contenu)
def is_useless_description(description: str) -> bool:
    """Détermine si une description d'image est inutile (ex: image noire, vide, sans contenu)."""
    keywords = [
        # Français
        "entièrement noire", "aucun élément", "aucun détail", "image vide",
        "image corrompue", "absence de lumière", "aucune forme", "aucune couleur",
        "aucun motif", "aucun texte", "obscurité complète", "vide complet",
        # Anglais
        "completely black", "no visible elements", "no details", "empty image",
        "corrupted image", "no light", "no shapes", "no colors", "no patterns",
        "no text", "total darkness", "blank image", "black screen", "no content"
    ]
    return any(kw in description.lower() for kw in keywords)


# Fonction pour obtenir la date formatée pour le nom de fichier
def get_formatted_date():
    return datetime.now().strftime("%d_%m_%Y")


# Fonction pour formater les mots-clés pour le nom de fichier
def format_keywords_for_filename(keywords):
    return "_".join(keywords.strip().split())


# 1ère étape : Extraire le texte page par page (version 1, pas à jour mais de référence)
def extract_pdf_pages_and_imagesV0(pdf_path):
    if pdf_path in pdf_cache:
        return pdf_cache[pdf_path]

    doc = fitz.open(pdf_path)
    pages_text = []
    all_images = []
    
    pdf_name = os.path.basename(pdf_path)
    cache_file = os.path.join(CACHE_FOLDER, f"{pdf_name}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            all_images = json.load(f)

    global_img_index = 1  
    for page_number, page in enumerate(doc):
        text = page.get_text()
        image_descriptions = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            image = Image.open(io.BytesIO(base_image["image"]))
            if image.size == (0, 0):
                print(f"[Image {img_index+1}]: Image vide ou corrompue, ignorée.")
                continue

            # Redimensionner à 1024x1024 max tout en conservant le ratio
            max_size = (1024, 1024)
            image.thumbnail(max_size, Image.LANCZOS)

            # Convertir l'image en base64 pour l'envoyer à GPT-4 Vision
            buffered = io.BytesIO()
            # Convertir l'image en RGB si elle est en RGBA
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Appel à GPT-4 Vision pour décrire l'image
            try:
                response = client.chat.completions.create(
                    model="gpt4o-2024-08-06",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Décris cette image de manière complète même si elle ne contient pas de texte, si elle en contient, lis tout texte visible. Mentionne les éléments visuels, le contexte, et toute information pertinente."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                            ]
                        }
                    ],
                    temperature=0.3
                )
                description = response.choices[0].message.content
                # image_descriptions.append(f"[Image {img_index+1}]: {description}")
                image_descriptions.append(f"[Image {global_img_index}]: {description}")
                global_img_index += 1
            except Exception as e:
                # image_descriptions.append(f"[Image {img_index+1}]: Erreur d'analyse - {str(e)}")
                preview_b64 = img_b64[:100] + "..." if len(img_b64) > 100 else img_b64
                error_msg = f"[Image {img_index+1}]: Erreur d'analyse - {str(e)} | base64 preview: {preview_b64}"
                print(error_msg)
                image_descriptions.append(error_msg)

        # page_content = f"{text}\n\n" + "\n".join(image_descriptions) if image_descriptions else text
        pages_text.append(text)
        
        all_images.extend(image_descriptions)
        # for desc in image_descriptions:
        #     if desc not in all_images:
        #         all_images.append(desc)
        
        # Nettoyage des descriptions similaires
        all_images = deduplicate_similar_descriptions(all_images)

    pdf_cache[pdf_path] = (pages_text, all_images)

    # with open(cache_file, "w", encoding="utf-8") as f:
    #     json.dump(all_images, f, ensure_ascii=False, indent=2)
    save_cache_if_needed(cache_file, pdf_cache.get(pdf_path, ([], []))[1], all_images)

    return pages_text, all_images


# 1ère étape : Extraire le texte page par page (version 2, à jour)
def extract_pdf_pages_and_images(pdf_path):
    if pdf_path in pdf_cache:
        return pdf_cache[pdf_path]

    doc = fitz.open(pdf_path)
    pages_text = []
    all_images = []
    seen_hashes = {}

    pdf_name = os.path.basename(pdf_path)
    cache_file = os.path.join(CACHE_FOLDER, f"{pdf_name}.json")

    # if os.path.exists(cache_file):
    #     with open(cache_file, "r", encoding="utf-8") as f:
    #         all_images = json.load(f)
    existing_descriptions = []
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            existing_descriptions = json.load(f)
            all_images = existing_descriptions.copy()

    global_img_index = 1

    for page_number, page in enumerate(doc):
        text = page.get_text()
        image_descriptions = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            if image.size == (0, 0):
                print(f"[Image {img_index+1}]: Image vide ou corrompue, ignorée.")
                continue

            max_size = (1024, 1024)
            image.thumbnail(max_size, Image.LANCZOS)

            if image.mode == 'RGBA': # red, green, blue, alpha(opacité)
                image = image.convert('RGB')

            img_hash = get_image_hash(image)

            if img_hash in seen_hashes:
                description = seen_hashes[img_hash]
            else:
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                try:
                    response = client.chat.completions.create(
                        model="gpt4o-2024-08-06",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Décris cette image de manière complète même si elle ne contient pas de texte..."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                                ]
                            }
                        ],
                        temperature=0.3
                    )
                    description = response.choices[0].message.content
                    seen_hashes[img_hash] = description
                except Exception as e:
                    preview_b64 = img_b64[:100] + "..." if len(img_b64) > 100 else img_b64
                    description = f"[Image {img_index+1}]: Erreur d'analyse - {str(e)}\n base64 preview: {preview_b64}"
                    print(description)

            if not is_useless_description(description): 
                image_descriptions.append(f"[Image {global_img_index}]: {description}")
            global_img_index += 1

        pages_text.append(text)
        # Ajout du texte et des descriptions d'images
        all_images.extend(image_descriptions)

    all_images = deduplicate_similar_descriptions(all_images) # Nettoyage du cache des descriptions similaires
    pdf_cache[pdf_path] = (pages_text, all_images)

    # with open(cache_file, "w", encoding="utf-8") as f:
    #     json.dump(all_images, f, ensure_ascii=False, indent=2)

    # if should_update_cache(existing_descriptions, all_images):
    #     with open(cache_file, "w", encoding="utf-8") as f:
    #         json.dump(all_images, f, ensure_ascii=False, indent=2)
    #     print(f"✓ Cache mis à jour : {cache_file}")
    # else:
    #     print(f"✓ Cache conservé sans modification : {cache_file}")
    save_cache_if_needed(cache_file, existing_descriptions, all_images) # Sauvegarde du cache si nécessaire

    return pages_text, all_images


# 2ème étape : Trouver les pages pertinentes via méthode TD-IDF + similarité cosinus 
def find_relevant_pages(pages, question, top_k=6): 
    vectorizer = TfidfVectorizer(  # Initialisation du vectoriseur TF-IDF
        stop_words=french_stop_words,       # Ignore les mots courants
        max_features=5000,                  # Limite le vocabulaire
        ngram_range=(1, 2)                  # Prend en compte les bigrammes
    ).fit(pages + [question])
    page_vectors = vectorizer.transform(pages)
    question_vector = vectorizer.transform([question]) 
    similarities = cosine_similarity(question_vector, page_vectors).flatten() # Calcul de la similarité cosinus
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [pages[i] for i in top_indices]


# 3ème étape : Charger et sauvegarder la mémoire de conversation
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except json.JSONDecodeError:
            print("Erreur de décodage JSON dans le fichier mémoire.")
            return []
    return []


# 4ème étape : Sauvegarder la mémoire de conversation
def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


# 5ème étape : Afficher l'aide
def display_help():
    print("\nCommandes disponibles :")
    print("  ➤ next     : Passer au document suivant")
    print("  ➤ previous : Revenir au document précédent")
    print("  ➤ list     : Afficher la liste des documents PDF")
    print("  ➤ summary  : Générer un résumé automatique du document")
    print("  ➤ reset    : Effacer la mémoire liée à ce document")
    print("  ➤ exit     : Quitter le programme")


# 6ème étape : Traiter les PDF dans un dossier et interagir avec l'utilisateur (différents appels spéciaux)
def process_pdf_folder(folder_path):
    check_multiple_instances()  # Vérifie les instances multiples (potentielles) du script
    current_index = 0
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    while True:
        pdf_file = pdf_files[current_index]
        full_path = os.path.join(folder_path, pdf_file)
        print(f"\nTraitement du fichier : {pdf_file}")
        doc_id = pdf_file
        conversation_memory = load_memory()
        pages, image_descriptions = extract_pdf_pages_and_images(full_path)
        print("PDF chargé et indexé.")
        while True:
            # pdf_file = pdf_files[current_index]
            # full_path = os.path.join(folder_path, pdf_file)
            # pages, image_descriptions = extract_pdf_pages_and_images(full_path)

            question = input(f"➢ Question sur {pdf_file} (ou 'next', 'previous', 'list', 'summary', 'reset', 'exit', 'help') : ").strip().lower()

            if question == "next":
                if current_index < len(pdf_files) - 1:
                    current_index += 1
                    break
                else:
                    print("Vous êtes déjà au dernier document.")
                continue

            elif question == "list":
                print("\nListe des fichiers PDF disponibles :")
                for i, f in enumerate(pdf_files):
                    marker = "➢" if i == current_index else "  "
                    print(f"{marker} {f}")

            elif question == "previous":
                if current_index > 0:
                    current_index -= 1
                    break
                else :
                    print("Vous êtes déjà au premier document.")
                continue

            elif question == "summary":
                full_text = "\n\n".join(pages)
                context = full_text + "\n\n" + "\n\n".join(image_descriptions)
                summary_prompt = "Peux-tu me faire un résumé clair et structuré de ce document PDF ?"
                answer, duree = ask_openai(context, summary_prompt, conversation_memory, doc_id)
                print(f"\nRésumé :\n{answer}")
                print(f"Temps de réponse : {duree:.2f} secondes\n")

            elif question == "reset":
                conversation_memory = [entry for entry in conversation_memory if entry[2] != doc_id]
                save_memory(conversation_memory)
                print(f"Mémoire effacée pour le document : {doc_id}")
                continue

            elif question == "exit":
                print("Fin de la session.")
                sys.exit(0)

            elif question == "help":
                display_help()

            elif question == "":
                print("Veuillez poser une question ou utiliser une commande.")
                continue

            else :
                relevant_pages = find_relevant_pages(pages, question, top_k=6)
                relevant_text = "\n\n".join(relevant_pages)
                unique_images = [desc for desc in image_descriptions if desc not in relevant_text]
                context = relevant_text + "\n\n" + "\n\n".join(unique_images)
                answer, duree = ask_openai(context, question, conversation_memory, doc_id)
                print(f"\nRéponse : {answer}")
                print(f"Temps de réponse : {duree:.2f} secondes\n\n")


# 7ème étape : Interroger OpenAI avec mémoire limitée à 5 échanges pour éviter les prompts trop longs
def ask_openai(context, question, memory, doc_id):
    specific_doc_memory = [entry for entry in memory if entry[2] == doc_id][-5:] # on prend seulement les 5 derniers échanges pour ce document
    memory_context = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in specific_doc_memory])

    prompt = f"""You are a helpful assistant. Based on the following document excerpts and previous conversation, answer the question.
    You are allowed to search on the internet if you are asked a question that is not in the document,
    if so, tell the user that you searched on the internet and provide the answer.

    Previous conversation:
    {memory_context}

    Document excerpts:
    {context}

    Current question:
    {question}
    """

    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt4o-2024-08-06",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    end_time = time.time()
    duree = end_time - start_time
    answer = response.choices[0].message.content

    memory.append((question, answer, doc_id))
    save_memory(memory)

    return answer, duree


# 8ème étape : Vérifier les instances multiples du script (débugg car double voir triple affichage de certaines lignes de la console)
def check_multiple_instances():
    current_pid = os.getpid()
    script_name = os.path.basename(__file__)
    print(f"✓ Instance actuelle PID : {current_pid}")
    running_instances = [p.info for p in psutil.process_iter(['pid', 'name', 'cmdline'])
        if p.info['cmdline'] and script_name in ' '.join(p.info['cmdline']) and p.info['pid'] != current_pid]
    if running_instances:
        print(f"/!\\ Attention : {len(running_instances)} autre(s) instance(s) du script sont en cours d'exécution.")
    else:
        print("✓ Aucune autre instance du script détectée.")


# Fonction pour traiter un dossier contenant des PDFs avec une question spécifique
def process_pdf_folder_with_keywords(folder_path):
    check_multiple_instances()  
    # def get_formatted_date():
    #     return datetime.now().strftime("%d_%m_%Y")

    # def format_keywords_for_filename(keywords):
    #     return "_".join(keywords.strip().split())

    if not os.path.exists(folder_path):
        print(f"/!\\ Dossier introuvable : {folder_path}")
        return

    results_folder = "parser_agent_results"
    os.makedirs(results_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("/!\\ Aucun fichier PDF trouvé dans le dossier.")
        return

    while True:
        question = input("\nEntrez une question ou des mots-clés à rechercher (ou 'exit' pour quitter) : ").strip()
        if question.lower() == "exit":
            print("Fin du mode parser.")
            break

        print(f"\nRecherche dans tous les fichiers PDF pour : {question}\n")

        cumulative_results = []
        export_filename = None

        for pdf_file in pdf_files:
            full_path = os.path.join(folder_path, pdf_file)
            print(f"\nTraitement du fichier : {pdf_file}")
            pages, image_descriptions = extract_pdf_pages_and_images(full_path)
            relevant_pages = find_relevant_pages(pages, question, top_k=6)
            relevant_text = "\n\n".join(relevant_pages)
            unique_images = [desc for desc in image_descriptions if desc not in relevant_text]

            print("\nPages pertinentes :")
            for i, page in enumerate(relevant_pages, 1):
                print(f"\n--- Page {i} ---\n{page[:500]}...")

            print("\nDescriptions d'images pertinentes :")
            for desc in unique_images:
                if any(word.lower() in desc.lower() for word in question.split()):
                    print(f"\n{desc[:500]}...")

            export_choice = input("\nSouhaitez-vous ajouter ces résultats à un fichier PDF ? [oui/non] (ou 'exit' pour quitter) : ").strip().lower()
            if export_choice == "oui":
                cumulative_results.append((pdf_file, relevant_pages, unique_images))
                if export_filename is None:
                    formatted_keywords = format_keywords_for_filename(question)
                    date_str = get_formatted_date()
                    export_filename = os.path.join(results_folder, f"{formatted_keywords}_{date_str}.pdf")

                # doc = fitz.open(export_filename) if os.path.exists(export_filename) else fitz.open()
                doc = fitz.open() # Création du PDF
                for pdf_name, pages_to_add, images_to_add in [cumulative_results[-1]]:
                    title = f"Résultats pertinents pour : {pdf_name}"
                    page = doc.new_page()
                    page.insert_text((50, 50), title, fontsize=16, fontname="helv", fill=(0, 0, 0))
                    y = 100
                    for i, content in enumerate(pages_to_add, 1):   # Ajouts des pages pertinentes
                        page = doc.new_page()
                        page.insert_text((50, 50), f"--- Page {i} ---", fontsize=14, fontname="helv", fill=(0, 0, 0))
                        y = 80
                        for line in content.split('\n'):
                            page.insert_text((50, y), line, fontsize=11, fontname="helv", fill=(0, 0, 0))
                            y += 15
                    if images_to_add:   # Ajouts des descriptions d'images pertinentes
                        page = doc.new_page()
                        page.insert_text((50, 50), "Descriptions d'images pertinentes :", fontsize=14, fontname="helv", fill=(0, 0, 0))
                        y = 80
                        for desc in images_to_add:
                            for line in desc.split('\n'):
                                page.insert_text((50, y), line, fontsize=11, fontname="helv", fill=(0, 0, 0))
                                y += 15
                                if y > 800:
                                    page = doc.new_page()
                                    y = 50
                doc.save(export_filename)
                doc.close()
                print(f"✓ Résultats ajoutés au fichier PDF : {export_filename}")

            elif export_choice == "exit":
                print("Fin du mode parser.")
                sys.exit(3)

# Fonction pour traduire un texte en utilisant un LLM
def translate_with_llm(text, target_lang):
    prompt = f"Voici un extrait de document à traduire en '{target_lang}'. Traduis-le intégralement page par page sans commentaire :\n\n{text}"
    try:
        response = client.chat.completions.create(
            model="gpt4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erreur de traduction avec LLM : {e}")
        return text

# Fonction pour traduire un PDF en utilisant un LLM
def translate_pdf_with_llm(folder_path):
    check_multiple_instances()
    if not os.path.exists(folder_path):
        print(f"/!\\ Dossier introuvable : {folder_path}")
        return
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("/!\\ Aucun fichier PDF trouvé dans le dossier.")
        return

    print("\nListe des fichiers PDF disponibles :")
    for i, f in enumerate(pdf_files):
        print(f"  {i+1}. {f}")

    try:
        index = int(input("\nEntrez le numéro du fichier à traduire : ").strip()) - 1
        if index < 0 or index >= len(pdf_files):
            print("/!\\ Numéro invalide.")
            return
    except ValueError:
        print("/!\\ Entrée invalide.")
        return

    pdf_file = pdf_files[index]
    base_name = os.path.splitext(pdf_file)[0]

    target_lang = input("Entrez la langue cible :\n'fr' : Français\n, 'en' : Anglais\n, 'de' : Allemand\n, 'it' : Italien\n, 'tr' : Turque\n, 'pl' : Polonais\n, 'ru' : Russe\n➤  ").strip().lower()
    if target_lang not in ['fr', 'en', 'de', 'it', 'tr', 'pl', 'ru']:
        print("/!\\ Langue non supportée.")
        return

    base_name = os.path.splitext(pdf_file)[0]
    parts = base_name.split('_')
    if len(parts) > 1 : 
        source_lang = parts[-1]
    else:
        source_lang = "unknown"
    
    if source_lang == target_lang:
        print(f"/!\\ Le fichier '{pdf_file}' est déjà en '{target_lang}'. Aucune traduction nécessaire.")
        return
    
    for f in os.listdir(folder_path):
        if f.lower().endswith(".pdf") and f.startswith(f"{base_name}_") and f.endswith(f"_{target_lang}.pdf"):          
            print(f"/!\\ Une traduction en '{target_lang}' existe déjà pour le fichier '{pdf_file}' : {f}")
            return

    translated_name = f"{base_name}_{target_lang}.pdf"
    translated_path = os.path.join(folder_path, translated_name)

    if os.path.exists(translated_path):
        print(f"/!\\ Le fichier traduit existe déjà : {translated_name}")
        return

    original_path = os.path.join(folder_path, pdf_file)
    doc = fitz.open(original_path)
    translated_doc = fitz.open()

    for page in doc:    # Traduction page par page
        text = page.get_text()
        translated_text = translate_with_llm(text, target_lang)

        new_page = translated_doc.new_page(width=595, height=842)
        y = 50
        for line in translated_text.split('\n'):
            while len(line) > 0:
                max_chars = int((595 - 100) / (11 * 0.6))
                chunk = line[:max_chars]
                line = line[max_chars:]
                if y + 14 > 842 - 50:
                    new_page = translated_doc.new_page(width=595, height=842)
                    y = 50
                new_page.insert_text((50, y), chunk, fontsize=11)
                y += 14

    translated_doc.save(translated_path)
    print(f"✓ PDF traduit enregistré : {translated_name}")


# Fonction pour comparer les résumés de plusieurs PDFs
def compare_summaries(summaries):
    """
    Compare les résumés de plusieurs PDFs pour identifier les similarités et différences.
    summaries : liste de tuples (nom_fichier, résumé)
    """
    print("\n~~~~~~~~~~~~~ Comparaison des résumés ~~~~~~~~~~~~~")

    # Affichage des résumés
    for name, summary in summaries:
        print(f"\nRésumé de {name} :\n{'-'*40}\n{summary[:1000]}...\n")

    # Similarité globale entre les résumés
    texts = [s for _, s in summaries]
    vectorizer = TfidfVectorizer(stop_words=french_stop_words)
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    print("\nSimilarité entre les documents (cosine similarity) :")
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            sim_score = similarity_matrix[i][j]
            print(f" - {summaries[i][0]} vs {summaries[j][0]} : {sim_score:.2f}") # Nom des fichiers + score (ex : doc1.pdf vs doc2.pdf : 0.87)

    # Analyse des différences textuelles
    print("\nDifférences textuelles entre les résumés :")
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            diff = difflib.unified_diff(
                summaries[i][1].splitlines(),
                summaries[j][1].splitlines(),
                fromfile=summaries[i][0],
                tofile=summaries[j][0],
                lineterm=''
            )
            print(f"\n--- Différences entre {summaries[i][0]} et {summaries[j][0]} ---")
            for line in list(diff)[:50]:  # Limite à 50 lignes pour lisibilité
                print(line)


# Fonction pour créer un PDF des résultats du mode comparaison (mode 4) (ancienne version, devenue summary_pdf et mode 5)
def create_comparison_pdfV00(folder_path, selected_files, summaries):
    output_folder = "comparison_agent_results"
    os.makedirs(output_folder, exist_ok=True)

    # Formation pour le nom du fichier
    numbers = [re.search(r'(\d+)', name).group(1) for name in selected_files if re.search(r'(\d+)', name)]
    date = get_formatted_date()
    filename = f"{date}_{'_'.join(numbers)}.pdf"
    output_path = os.path.join(output_folder, filename)

    doc = fitz.open()

    # Page de garde
    cover = doc.new_page()
    cover.insert_text((50, 50), "Résumés des documents PDFs", fontsize=16)
    cover.insert_text((50, 80), f"Dossier source : {folder_path}", fontsize=12)
    cover.insert_text((50, 110), "Fichiers PDF utilisés :", fontsize=12)
    y = 140
    for name in selected_files:
        cover.insert_text((50, y), f"- {name}", fontsize=11)
        y += 20

    # Résumés
    for name, summary in summaries:
        page = doc.new_page()
        page.insert_text((50, 50), f"Résumé de {name}", fontsize=14)
        y = 80
        for line in summary.split('\n'):
            page.insert_text((50, y), line, fontsize=11)
            y += 15
            if y > 800:
                page = doc.new_page()
                y = 50

    doc.save(output_path)
    doc.close()
    print(f"✓ Fichier PDF de comparaison enregistré : {output_path}")


# Fonction pour créer un PDF des résultats du mode résumé (mode 5) (ancienne version, devenue summary_pdf et mode 5)
def create_summary_pdf(folder_path, selected_files, summaries):
    """
    Crée un PDF affichant tous les résumés disponibles.
    Chaque ligne est découpée à 80 caractères pour éviter les débordements.
    """
    output_folder = "comparison_agent_results"
    os.makedirs(output_folder, exist_ok=True)

    # Création du nom du fichier
    numbers = [re.search(r'(\d+)', name).group(1) for name in selected_files if re.search(r'(\d+)', name)]
    date = datetime.now().strftime("%d_%m_%Y")
    filename = f"{date}_{'_'.join(numbers)}.pdf"
    output_path = os.path.join(output_folder, filename)

    doc = fitz.open()
    
    # Paramètres de mise en page
    page_width = 595  # A4
    page_height = 842
    margin = 50
    font_size = 11
    line_height = 15
    max_chars_per_line = int((page_width - 2 * margin) / (font_size * 0.6))

    # Page de garde
    cover = doc.new_page()
    cover.insert_text((margin, 50), "Résumés des documents PDFs", fontsize=16)
    cover.insert_text((margin, 80), f"Dossier source : {folder_path}", fontsize=12)
    cover.insert_text((margin, 110), "Fichiers PDF utilisés :", fontsize=12)
    y = 140
    for name in selected_files:
        cover.insert_text((margin, y), f"- {name}", fontsize=11)
        y += 20

    # Pages de résumés
    for name, summary in summaries:
        page = doc.new_page()
        page.insert_text((margin, 50), f"Résumé de {name}", fontsize=14)
        y = 80
        for line in summary.split('\n'):
            wrapped_lines = textwrap.wrap(line, width=max_chars_per_line, break_long_words=False)
            for wrapped_line in wrapped_lines:
                if y > page_height - margin:
                    page = doc.new_page()
                    y = margin
                page.insert_text((margin, y), wrapped_line, fontsize=font_size)
                y += line_height

    doc.save(output_path)
    doc.close()
    print(f"✓ Fichier PDF de résumé enregistré : {output_path}")
    return output_path


def summary_pdf(folder_path):
    """
    Mode 5 : Génère un résumé automatique pour un ou plusieurs fichiers PDF et crée un PDF de synthèse.
    """
    if not os.path.exists(folder_path):
        print(f"/!\\ Dossier introuvable : {folder_path}")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("/!\\ Aucun fichier PDF trouvé dans le dossier.")
        return

    print("\nListe des fichiers PDF disponibles :")
    for i, f in enumerate(pdf_files):
        print(f"  {i+1}. {f}")

    selected_indices = input("\nEntrez le numéro des fichiers à résumer (ex: '1, 2, 3') :\n➤  ").strip()
    try:
        indices = [int(i.strip()) - 1 for i in selected_indices.split(",")]
        selected_files = [pdf_files[i] for i in indices if 0 <= i < len(pdf_files)]
    except ValueError:
        print("/!\\ Entrée invalide.")
        return

    summaries = []
    for pdf_file in selected_files:
        full_path = os.path.join(folder_path, pdf_file)
        pages, image_descriptions = extract_pdf_pages_and_images(full_path)
        full_text = "\n\n".join(pages)
        context = full_text + "\n\n" + "\n\n".join(image_descriptions)
        summary_prompt = "Peux-tu me faire un résumé clair et structuré de ce document PDF ?"
        summary, _ = ask_openai(context, summary_prompt, [], pdf_file)
        summaries.append((pdf_file, summary))

    # Confirmation utilisateur avant création du PDF
    choice = input("\nSouhaitez-vous sauvegarder ces résumés dans un fichier PDF ? [oui/non] (ou 'exit') : ").strip().lower()
    if choice == "oui":
        create_summary_pdf(folder_path, selected_files, summaries)

    elif choice == "non":
        print("Résumé non sauvegardé.")
        
    elif choice == "exit":
        print("Fin du mode résumé.")
        sys.exit(5)
        
    while True:
        choice2 = input("\nSouhaitez-vous relancer une nouvelle comparaison entre des fichiers PDF ? [oui/non] : ").strip().lower()
        if choice2 == "oui":
            compare_pdf(folder_path)
            return
        
        elif choice2 == "non":
            print("Fin du mode comparaison.")
            sys.exit(5)

        else:
            print("Réponse invalide. Veuillez répondre par 'oui' ou par 'non'.")



# Fonction pour créer un PDF lisible des résultats du mode comparaison (mode 4) (version améliorée)
def create_comparison_pdf(folder_path, selected_files, summaries):
    output_folder = "comparison_agent_results"
    os.makedirs(output_folder, exist_ok=True)

    # Formattage du nom du fichier
    numbers = [re.search(r'(\\d+)', name).group(1) for name in selected_files if re.search(r'(\\d+)', name)]
    date = get_formatted_date()
    filename = f"{date}_{'_'.join(numbers)}_readable.pdf"
    output_path = os.path.join(output_folder, filename)

    # Création du contenu textuel
    text_lines = []
    text_lines.append("Résumé de la comparaison de documents")
    text_lines.append(f"Dossier comparé : {folder_path}")
    text_lines.append("Fichiers PDF utilisés :")
    for name in selected_files:
        text_lines.append(f"- {name}")
    text_lines.append("")

    for name, summary in summaries:
        text_lines.append(f"Résumé de {name}")
        text_lines.extend(summary.splitlines())
        text_lines.append("")

    create_readable_pdf(text_lines, output_path)
    print(f"✓ Fichier PDF lisible enregistré : {output_path}")


# Fonction pour comparer plusieurs PDFs entre eux pour faire ressortir les similarités et différences
def compare_pdf(folder_path):
    check_multiple_instances()
    if not os.path.exists(folder_path):
        print(f"/!\\ Dossier introuvable : {folder_path}")
        return
        
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("/!\\ Aucun fichier PDF trouvé dans le dossier.")
        return

    print("\nListe des fichiers PDF disponibles :")
    for i, f in enumerate(pdf_files):
        print(f"  {i+1}. {f}")

    selected_indices = input("\nEntrez le numéro des fichiers à comparer (de la sorte : '1, 2, etc...') :\n➤  ").strip()
    try:
        indices = [int(i.strip()) - 1 for i in selected_indices.split(",")]
        selected_files = [pdf_files[i] for i in indices if 0 <= i < len(pdf_files)]
    except ValueError:
        print("/!\\ Entrée invalide.")
        return

    summaries = []
    for pdf_file in selected_files:
        full_path = os.path.join(folder_path, pdf_file)
        pages, image_descriptions = extract_pdf_pages_and_images(full_path)
        full_text = "\n\n".join(pages)
        context = full_text + "\n\n" + "\n\n".join(image_descriptions)
        summary_prompt = "Peux-tu me faire un résumé clair et structuré de ce document PDF ?"
        summary, _ = ask_openai(context, summary_prompt, [], pdf_file)
        summaries.append((pdf_file, summary))

    compare_summaries(summaries)

    print("\n✓ Comparaison terminée.")
    choice = input("Souhaitez-vous sauvegarder ces compraisons dans un fichier PDF ? [oui/non] (ou 'exit'): ").strip().lower()

    if choice == "oui":
        # create_comparison_pdfV0(folder_path, selected_files, summaries) # fonction ancienne qui n'affichait que les résumés, elle a été renommée create_summmary_pdf et est devenu le mode 5
        # create_summary_pdf(folder_path, selected_files, summaries)
        create_comparison_pdf(folder_path, selected_files, summaries) # nouvelle fonction qui ne mets dans le pdf crée que les similarités et différences entre les résumés
    
    elif choice == "non":
        print("Comparaison non sauvegardée X.")
        return
    
    elif choice == "exit":
        print("Fin du mode comparaison.")
        sys.exit(4)
 
    while True:
        choice2 = input("\nSouhaitez-vous relancer une nouvelle comparaison entre des fichiers PDF ? [oui/non] : ").strip().lower()
        if choice2 == "oui":
            compare_pdf(folder_path)
            return
        
        elif choice2 == "non":
            print("Fin du mode comparaison.")
            sys.exit(5)

        else:
            print("Réponse invalide. Veuillez répondre par 'oui' ou par 'non'.")

    
# Fonction pour afficher le menu principal et choisir le mode d'éxécution : classique ou batch
def main_menu():
    print("\n~~~~~~~~~~~~~~~~~~~~ Menu Principal ~~~~~~~~~~~~~~~~~~~~")
    print("1 - Mode classique (navigation interactive dans les PDFs, questions sur un PDF à la fois)")
    print("2 - Mode parser (recherche par mots-clés dans tous les PDFs)")
    print("3 - Mode traducteur (traduction de PDFs produits par le mode parser en une autre langue choisie)")
    print("4 - Mode comparaison (comparaison de plusieurs PDFs entre eux afin de voir les similarités et différences)")
    print("5 - Mode résumé (résumé automatique d'un PDF ou de plusieurs PDFs)")
    print("exit - Quitter le programme")
    print("P.S. : Nécessite une version de Python 3.11 ou inférieure pour fonctionner correctement.")
    choice = input("➤  Entrez 1, 2, 3, 4, 5 ou 'exit': ").strip()

    if choice == "1":
        folder_path = input("➤  Chemin vers le dossier de PDFs : ").strip()
        process_pdf_folder(folder_path)
        
    elif choice == "2":
        folder_path = input("➤  Chemin vers le dossier de PDFs : ").strip()
        process_pdf_folder_with_keywords(folder_path)

    elif choice == "3":
        folder_path = input("➤  Chemin vers le dossier de PDFs : ").strip()
        translate_pdf_with_llm(folder_path)

    elif choice == "4":
        folder_path = input("➤  Chemin vers le dossier de PDFs : ").strip()
        compare_pdf(folder_path)

    elif choice == "5":
        folder_path = input("➤  Chemin vers le dossier de PDFs : ").strip()
        summary_pdf(folder_path)

    elif choice == "exit":
        print("Fin de la session.")
        sys.exit(1)

    else:
        print("/!\\ Choix invalide.")
        sys.exit(2)


# 9ème étape : Test de l'agent OCR sur des PDF dans un dossier via leur chemin, un PDF à la fois ou tous les PDFs en même temps
if __name__ == "__main__":
    # print("Script lancé") # test pour bug d'affichage de la console
    # check_multiple_instances() n'est plus appelé ici

    # PDF_PATH = "/Users/L1163464/Franz_Kafka.pdf"
    # PDF_PATH = "/Users/L1163464/GM-GR-HSE-404_en_1.pdf"
    # PDF_PATH = "/Users/L1163464/Agent_OCR_testPDF.pdf"
    # PDF_PATH = "/Users/L1163464/Agent_OCR_testPDFv2.pdf"
    # FOLDER_PATH = "/Users/L1163464/OCR_testV3"  # Remplace par le chemin réel de ton dossier

    # if not os.path.exists(FOLDER_PATH):
    #     print(f"Dossier introuvable : {FOLDER_PATH}")
    # else:
    #     process_pdf_folder(FOLDER_PATH)

    # Chemin test pour tester le mode parser : C:/Users/L1163464/OCR_testV3
    # Chemin test pour le mode traducteur : C:/Users/L1163464/py_test/parser_agent_results ou C:/Users/L1163464/OCR_testV3
    # Chemin test pour le mode traducteur : C:/Users/L1163464/OCR_testV3
    main_menu()