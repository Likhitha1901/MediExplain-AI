from pathlib import Path
import hashlib
import re
import requests
import xml.etree.ElementTree as ET
import html
import zipfile
from pypdf import PdfReader
from docx import Document
from rag.vector_store import MedicalVectorStore
# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "medical_documents"
)
# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ============================================================
# ONLINE MEDICAL SOURCE
# ============================================================

MEDLINEPLUS_XML_URL = (
    "https://medlineplus.gov/xml/health-topics.xml"
)

MEDLINEPLUS_SOURCE = "MedlinePlus"
MEDLINEPLUS_CATEGORY = "medical_information"

# ============================================================
# DOWNLOAD MEDLINEPLUS DATA
# ============================================================

def download_medlineplus_xml():

    print("\nConnecting to MedlinePlus...")

    xml_page_url = "https://medlineplus.gov/xml.html"

    response = requests.get(
        xml_page_url,
        timeout=30
    )

    response.raise_for_status()

    page = response.text

    # Find the current compressed XML file
    match = re.search(
        r'href="([^"]+\.zip)"',
        page
    )

    if not match:
        raise RuntimeError(
            "Could not find the current MedlinePlus XML file."
        )

    xml_url = match.group(1)

    if xml_url.startswith("/"):
        xml_url = (
            "https://medlineplus.gov"
            + xml_url
        )

    print(
        f"Downloading MedlinePlus data from:\n{xml_url}"
    )

    xml_response = requests.get(
        xml_url,
        timeout=120
    )

    xml_response.raise_for_status()

    output_folder = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "medlineplus"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    zip_path = (
        output_folder
        / "medlineplus_health_topics.zip"
    )

    zip_path.write_bytes(
        xml_response.content
    )

    print(
        f"Saved MedlinePlus data to:\n{zip_path}"
    )

    return zip_path
# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(file_path: Path) -> str:

    print(f"Reading PDF: {file_path.name}")

    reader = PdfReader(str(file_path))

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)

# ============================================================
# EXTRACT MEDLINEPLUS XML
# ============================================================

def extract_medlineplus_xml(zip_path: Path):

    print("\nExtracting MedlinePlus XML...")

    output_folder = zip_path.parent

    extract_folder = (
        output_folder
        / "extracted"
    )

    extract_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_file:

        zip_file.extractall(
            extract_folder
        )

    xml_files = list(
        extract_folder.rglob("*.xml")
    )

    if not xml_files:

        raise RuntimeError(
            "No XML file was found inside the MedlinePlus ZIP."
        )

    xml_file = xml_files[0]

    print(
        f"MedlinePlus XML extracted to:\n{xml_file}"
    )

    return xml_file
# ============================================================
# DOCX EXTRACTION
# ============================================================
# ============================================================
# PARSE MEDLINEPLUS XML
# ============================================================

def parse_medlineplus_xml(xml_path: Path):

    print("\nReading MedlinePlus medical topics...")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    documents = []

    for index, topic in enumerate(root.findall(".//health-topic")):

        title = topic.get("title", "").strip()
        url = topic.get("url", "").strip()

        # Get summary
        summary_element = topic.find("full-summary")

        if summary_element is not None:
            summary = "".join(
                summary_element.itertext()
            ).strip()
        else:
            summary = ""

        # Get alternate name if available
        also_called_element = topic.find("also-called")

        alternate_names = ""

        if also_called_element is not None:
            alternate_names = " ".join(
                "".join(element.itertext()).strip()
                for element in also_called_element.findall("name")
            )

        # Skip topics without useful content
        if not title or not summary:
            continue

        content_parts = [
            f"Medical Topic: {title}",
            "",
            summary
        ]

        if alternate_names:
            content_parts.extend([
                "",
                f"Also called: {alternate_names}"
            ])

        content = "\n".join(content_parts)

        documents.append({

            "id": f"medlineplus_{index}",

            "content": content,

            "source": MEDLINEPLUS_SOURCE,

            "title": title,

            "category": MEDLINEPLUS_CATEGORY,

            "url": url,

            "file_name": "medlineplus_health_topics.xml",

            "chunk_index": index
        })

    print(
        f"MedlinePlus topics found: {len(documents)}"
    )

    return documents
def extract_docx_text(file_path: Path) -> str:

    print(f"Reading DOCX: {file_path.name}")

    document = Document(str(file_path))

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt_text(file_path: Path) -> str:

    print(f"Reading TXT: {file_path.name}")

    return file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


# ============================================================
# GENERAL DOCUMENT READER
# ============================================================

def extract_text(file_path: Path) -> str:

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        return extract_pdf_text(file_path)

    elif extension == ".docx":

        return extract_docx_text(file_path)

    elif extension == ".txt":

        return extract_txt_text(file_path)

    else:

        return ""


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text: str) -> str:

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CHUNK TEXT
# ============================================================

def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_category(file_name: str) -> str:

    name = file_name.lower()

    disease_words = [
        "disease",
        "diabetes",
        "asthma",
        "hypertension",
        "cancer",
        "arthritis",
        "infection",
        "syndrome"
    ]

    medication_words = [
        "medicine",
        "medication",
        "drug",
        "tablet",
        "dolo",
        "paracetamol",
        "metformin",
        "ibuprofen"
    ]

    treatment_words = [
        "treatment",
        "therapy",
        "management",
        "procedure"
    ]

    guideline_words = [
        "guideline",
        "guidelines",
        "clinical",
        "protocol"
    ]

    concept_words = [
        "medical",
        "concept",
        "hba1c",
        "ecg",
        "mri",
        "blood"
    ]

    if any(word in name for word in medication_words):
        return "medication"

    if any(word in name for word in treatment_words):
        return "treatment"

    if any(word in name for word in guideline_words):
        return "guideline"

    if any(word in name for word in concept_words):
        return "medical_concept"

    if any(word in name for word in disease_words):
        return "disease"

    return "medical_information"


# ============================================================
# CREATE UNIQUE DOCUMENT ID
# ============================================================

def create_document_id(
    file_path: Path,
    chunk_index: int
):

    unique_string = (
        f"{file_path.resolve()}_{chunk_index}"
    )

    return hashlib.md5(
        unique_string.encode("utf-8")
    ).hexdigest()


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(
    file_path: Path
):

    print("\n" + "=" * 60)

    print(
        f"Processing: {file_path.name}"
    )

    print("=" * 60)

    raw_text = extract_text(file_path)

    if not raw_text.strip():

        print(
            "WARNING: No text could be extracted."
        )

        return []

    text = clean_text(raw_text)

    chunks = create_chunks(text)

    category = detect_category(
        file_path.name
    )

    documents = []

    for index, chunk in enumerate(chunks):

        document_id = create_document_id(
            file_path,
            index
        )

        documents.append({

            "id": document_id,

            "content": chunk,

            "source": file_path.stem,

            "title": file_path.stem,

            "category": category,

            "url": "",

            "file_name": file_path.name,

            "chunk_index": index

        })

    print(
        f"Extracted characters: {len(text)}"
    )

    print(
        f"Created chunks: {len(chunks)}"
    )

    print(
        f"Category: {category}"
    )

    return documents


# ============================================================
# FIND MEDICAL DOCUMENTS
# ============================================================

def find_documents():

    extensions = [
        "*.pdf",
        "*.docx",
        "*.txt"
    ]

    files = []

    for extension in extensions:

        files.extend(
            DOCUMENT_FOLDER.glob(extension)
        )

    return sorted(files)


# ============================================================
# MAIN INGESTION
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("       MEDIEXPLAIN AI")
    print("   AUTOMATIC MEDICAL INGESTION")
    print("=" * 60)

    DOCUMENT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )
        # --------------------------------------------------------
    # MEDLINEPLUS
    # --------------------------------------------------------

    medlineplus_zip = download_medlineplus_xml()

    medlineplus_xml = extract_medlineplus_xml(
        medlineplus_zip
    )

    medlineplus_documents = parse_medlineplus_xml(
        medlineplus_xml
    )
    files = find_documents()

    if not files:

        print("\nNo local PDF/DOCX/TXT files found.")

        print(
            f"\nContinuing with MedlinePlus data...."
        )

        print(
            DOCUMENT_FOLDER
        )

        return

    print(
        f"\nFound {len(files)} medical document(s)."
    )

    all_documents = list(medlineplus_documents)

    for file_path in files:

        documents = process_document(
            file_path
        )

        all_documents.extend(
            documents
        )

    if not all_documents:

        print(
            "\nNo usable medical information found."
        )

        return

    # --------------------------------------------------------
    # CHROMADB
    # --------------------------------------------------------

    print("\nConnecting to ChromaDB...")

    vector_store = MedicalVectorStore()

    print(
        f"Adding {len(all_documents)} chunks "
        "to ChromaDB..."
    )

    vector_store.add_documents(
        all_documents
    )

    print("\n" + "=" * 60)

    print("INGESTION COMPLETED SUCCESSFULLY")

    print("=" * 60)

    print(
        f"Documents processed: {len(files)}"
    )

    print(
        f"Chunks added: {len(all_documents)}"
    )

    print(
        "ChromaDB is now ready for RAG."
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()