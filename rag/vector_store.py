
from urllib import response
import re
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse, parse_qs, unquote
from bs4 import BeautifulSoup


class MedicalVectorStore:

    def __init__(self, collection_name="medical_knowledge"):

        # Create a persistent Chroma database
        self.client = chromadb.PersistentClient(
            path="data/chroma_db"
        )

        # Create or load our collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        # Embedding model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def add_documents(self, documents):

        if not documents:
            return

        texts = []
        ids = []
        metadatas = []

        for index, document in enumerate(documents):

            texts.append(document["content"])

            ids.append(
                document.get("id", f"doc_{index}")
            )

            metadatas.append({
                "source": document.get("source", ""),
                "title": document.get("title", ""),
                "category": document.get("category", ""),
                "url": document.get("url", ""),
                "file_name": document.get("file_name", ""),
                "chunk_index": str(document.get("chunk_index", ""))
            })

        # Convert documents into embeddings
        embeddings = self.embedding_model.encode(
            texts
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query, top_k=5):

        # Convert user question into embedding
        query_embedding = self.embedding_model.encode(
            query
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results

    def search_medlineplus(self, query, top_k=5):
        """Search MedlinePlus live for relevant health topics."""

        url = "https://wsearch.nlm.nih.gov/ws/query"

        params = {
            "db": "healthTopics",
            "term": query,
            "retmax": top_k,
            "rettype": "brief",
            "tool": "mediexplainai",
        }

        try:
            response = requests.get(
                url,
                params=params,
                timeout=10,
            )

            response.raise_for_status()

            root = ET.fromstring(response.content)
            results = []

            for doc in root.findall(".//document"):

                title = ""
                snippet = ""
                full_summary = ""

                for content_element in doc.findall("content"):

                    name = content_element.attrib.get("name", "")
                    text = "".join(content_element.itertext()).strip()

                    if name == "title":
                        title = text

                    elif name == "snippet":
                        snippet = text

                    elif name == "FullSummary":
                        full_summary = text

                page_url = doc.attrib.get("url", "")
                content_text = snippet or full_summary

                if title and page_url and content_text:

                    results.append({
                        "id": page_url,
                        "content": content_text,
                        "title": title,
                        "source": "MedlinePlus",
                        "category": "medical_information",
                        "url": page_url,
                    })

            print(
                f"Live MedlinePlus retrieval: {len(results)} documents"
            )

            return results

        except (requests.RequestException, ET.ParseError) as exc:

            print(f"Live MedlinePlus search failed: {exc}")
            return []

    def search_nhs(self, query, top_k=5):
        """Search NHS live and retrieve actual medical article content."""

        print("\n========== NHS FUNCTION EXECUTED ==========")

        search_url = "https://www.nhs.uk/search/"

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MediExplainAI/1.0)"
        }

        try:

            # Search NHS
            response = requests.get(
                search_url,
                params={"q": query},
                headers=headers,
                timeout=15
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            print("NHS Status:", response.status_code)
            print("NHS Final URL:", response.url)

            all_links = soup.find_all("a", href=True)

            print("NHS total links found:", len(all_links))

            results = []
            seen_urls = set()

            # Process NHS search-result links
            for link in all_links:

                title = link.get_text(" ", strip=True)
                href = link.get("href", "").strip()

                if not title or not href:
                    continue

                # Resolve NHS tracking links
                parsed = urlparse(href)
                params = parse_qs(parsed.query)

                if "/search/click" in parsed.path:

                    target = params.get("url", [""])[0]

                    if not target:
                        continue

                    href = unquote(target)

                # Convert relative URLs into complete URLs
                page_url = urljoin(
                    "https://www.nhs.uk",
                    href
                )

                parsed_article = urlparse(page_url)

                # Allow only genuine NHS URLs
                if parsed_article.hostname not in (
                    "www.nhs.uk",
                    "nhs.uk"
                ):
                    continue

                # Keep medical article pages only
                if not any(
                    section in parsed_article.path.lower()
                    for section in [
                        "/conditions/",
                        "/medicines/",
                        "/symptoms/"
                    ]
                ):
                    continue

                # Avoid duplicate articles
                if page_url in seen_urls:
                    continue

                seen_urls.add(page_url)

                print("\nNHS ARTICLE FOUND:", title)
                print("NHS ARTICLE URL:", page_url)

                # Retrieve the actual NHS article
                try:

                    page_response = requests.get(
                        page_url,
                        headers=headers,
                        timeout=15
                    )

                    page_response.raise_for_status()

                    article_soup = BeautifulSoup(
                        page_response.text,
                        "html.parser"
                    )

                    main_content = article_soup.find("main")

                    if not main_content:

                        print(
                            "NHS main content not found:",
                            page_url
                        )

                        continue

                    # Remove irrelevant elements
                    for element in main_content.find_all(
                        [
                            "script",
                            "style",
                            "nav",
                            "footer",
                            "button"
                        ]
                    ):
                        element.decompose()

                    # Extract actual article content
                    content = main_content.get_text(
                        separator=" ",
                        strip=True
                    )

                    print(
                        "NHS extracted content length:",
                        len(content)
                    )

                    if len(content) < 150:

                        print("NHS article content too short")
                        continue

                    # Prefer the actual article heading
                    heading = article_soup.find("h1")

                    article_title = (
                        heading.get_text(" ", strip=True)
                        if heading
                        else title
                    )

                    results.append({
                        "id": page_url,
                        "content": content[:8000],
                        "title": article_title,
                        "source": "NHS",
                        "category": "medical_information",
                        "url": page_url
                    })

                    print(
                        "NHS article successfully extracted:",
                        article_title
                    )

                    if len(results) >= top_k:
                        break

                except requests.RequestException as exc:

                    print(
                        "NHS article retrieval failed:",
                        page_url,
                        exc
                    )

                    continue

            print(
                f"\nNHS live retrieval: {len(results)} documents"
            )

            return results

        except requests.RequestException as exc:

            print(f"NHS search failed: {exc}")
            return []
    
    
    def search_niddk(self, query, top_k=5):
        """Retrieve NIDDK articles and rank them by query relevance."""

        print("\n========== NIDDK FUNCTION EXECUTED ==========")

        search_url = "https://www.niddk.nih.gov/search"

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MediExplainAI/1.0)"
        }

        stop_words = {
            "what", "are", "is", "the", "a", "an", "and",
            "or", "of", "in", "on", "to", "for", "how",
            "does", "do", "can", "with", "about", "me",
            "explain", "tell", "please", "information"
        }

        def normalize_words(text):
            words = []

            for word in text.lower().split():
                word = "".join(
                    char for char in word if char.isalnum()
                )

                if len(word) < 3 or word in stop_words:
                    continue

                if word.endswith("ies") and len(word) > 4:
                    word = word[:-3] + "y"
                elif word.endswith("s") and not word.endswith("ss"):
                    word = word[:-1]
                elif word.endswith("ed") and len(word) > 4:
                    word = word[:-2]
                elif word.endswith("ing") and len(word) > 5:
                    word = word[:-3]

                words.append(word)

            return set(words)

        query_words = normalize_words(query)

        try:
            response = requests.get(
                search_url,
                params={"query": query},
                headers=headers,
                timeout=20
            )

            print("NIDDK status:", response.status_code)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.find_all("a", href=True)

            print("NIDDK links found:", len(links))

            candidates = []
            seen_urls = set()

            for link in links:
                title = link.get_text(" ", strip=True)
                href = link.get("href", "").strip()

                if not title or not href:
                    continue

                page_url = urljoin(
                    "https://www.niddk.nih.gov",
                    href
                )

                parsed = urlparse(page_url)

                if parsed.hostname not in (
                    "www.niddk.nih.gov",
                    "niddk.nih.gov"
                ):
                    continue

                if "/health-information/" not in parsed.path:
                    continue

                if page_url in seen_urls:
                    continue

                seen_urls.add(page_url)

                title_words = normalize_words(title)
                title_match = len(query_words & title_words)

                candidates.append({
                    "title": title,
                    "url": page_url,
                    "title_match": title_match
                })

            # Try the most promising article titles first.
            candidates.sort(
                key=lambda item: item["title_match"],
                reverse=True
            )

            results = []

            # Avoid downloading every link from the search page.
            for candidate in candidates[:20]:

                page_url = candidate["url"]

                try:
                    article_response = requests.get(
                        page_url,
                        headers=headers,
                        timeout=20
                    )

                    article_response.raise_for_status()

                    article_soup = BeautifulSoup(
                        article_response.text,
                        "html.parser"
                    )

                    main = article_soup.find("main")

                    if not main:
                        continue

                    for element in main.find_all([
                        "script", "style", "nav",
                        "footer", "button"
                    ]):
                        element.decompose()

                    content = main.get_text(
                        separator=" ",
                        strip=True
                    )

                    if len(content) < 200:
                        continue

                    heading = article_soup.find("h1")

                    article_title = (
                        heading.get_text(" ", strip=True)
                        if heading else candidate["title"]
                    )

                    title_words = normalize_words(article_title)
                    content_words = normalize_words(content[:6000])

                    title_matches = len(query_words & title_words)
                    content_matches = len(query_words & content_words)
                    # Require at least one meaningful query-word match in the title.
                    if title_matches == 0:
                        print("Skipped: no relevant title match")
                        continue

                    # Exclude professional blogs from patient-facing results.
                    if "/professionals/" in page_url:
                        print("Skipped: professional content")
                        continue
                    # Give article titles more importance than body text.
                    relevance = (
                        title_matches * 3
                        + content_matches
                    )

                    print(
                        f"NIDDK relevance: {relevance} | "
                        f"{article_title}"
                    )

                    # Reject articles with insufficient query relevance.
                    if relevance < 2:
                        print("Skipped irrelevant article")
                        continue

                    results.append({
                        "id": page_url,
                        "content": content[:8000],
                        "title": article_title,
                        "source": "NIDDK",
                        "category": "medical_information",
                        "url": page_url,
                        "relevance": relevance
                    })

                except requests.RequestException as exc:
                    print("NIDDK article retrieval failed:", exc)

            # Return the most relevant NIDDK articles.
            results.sort(
                key=lambda item: item["relevance"],
                reverse=True
            )

            results = results[:top_k]

            for result in results:
                result.pop("relevance", None)

            print(
                "NIDDK relevant documents:",
                len(results)
            )

            return results

        except requests.RequestException as exc:
            print("NIDDK search failed:", exc)
            return []


    
    def search_mayo_clinic(self, query, top_k=5):
        """Search Mayo Clinic articles using its website search."""

        print("\n========== MAYO CLINIC FUNCTION EXECUTED ==========")

        search_url = "https://www.mayoclinic.org/search/search-results"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        }

        try:
            response = requests.get(
                search_url,
                params={"q": query},
                headers=headers,
                timeout=20
            )

            print("Mayo status:", response.status_code)
            print("Mayo URL:", response.url)

            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            links = soup.find_all("a", href=True)

            print("Mayo total links:", len(links))

            results = []
            seen_urls = set()

            for link in links:

                title = link.get_text(" ", strip=True)
                href = link.get("href", "").strip()

                if not title or not href:
                    continue

                page_url = urljoin(
                    "https://www.mayoclinic.org",
                    href
                )

                parsed = urlparse(page_url)

                if parsed.hostname not in (
                    "www.mayoclinic.org",
                    "mayoclinic.org"
                ):
                    continue

                if not any(section in parsed.path.lower() for section in [
                    "/diseases-conditions/",
                    "/drugs-supplements/",
                    "/tests-procedures/"
                ]):
                    continue

                if page_url in seen_urls:
                    continue

                seen_urls.add(page_url)

                print("Mayo article found:", title)

                try:
                    article_response = requests.get(
                        page_url,
                        headers=headers,
                        timeout=20
                    )

                    article_response.raise_for_status()

                    article_soup = BeautifulSoup(
                        article_response.text,
                        "html.parser"
                    )

                    main = article_soup.find("main")

                    if not main:
                        continue

                    for element in main.find_all([
                        "script", "style", "nav",
                        "footer", "button"
                    ]):
                        element.decompose()

                    content = main.get_text(
                        separator=" ",
                        strip=True
                    )

                    if len(content) < 200:
                        continue

                    heading = article_soup.find("h1")

                    article_title = (
                        heading.get_text(" ", strip=True)
                        if heading else title
                    )

                    results.append({
                        "id": page_url,
                        "content": content[:8000],
                        "title": article_title,
                        "source": "Mayo Clinic",
                        "category": "medical_information",
                        "url": page_url
                    })

                    print("Mayo article extracted:", article_title)

                    if len(results) >= top_k:
                        break

                except requests.RequestException as exc:
                    print("Mayo article error:", exc)

            print("Mayo Clinic live retrieval:", len(results), "documents")

            return results

        except requests.RequestException as exc:
            print("Mayo Clinic search failed:", exc)
            return []
    
    def search_cleveland_clinic(self, query, top_k=5):
    
        print("\n========== CLEVELAND CLINIC FUNCTION EXECUTED ==========")

        BASE_URL = "https://my.clevelandclinic.org"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        try:
            # Step 1: Get homepage configuration
            response = requests.get(
                BASE_URL + "/health",
                headers=headers,
                timeout=20
            )
            response.raise_for_status()

            html = response.text

            # Step 2: Get Algolia search configuration
            script_url = (
                BASE_URL +
                "/assets/scripts/dist/health-library-search-scripts.js"
            )

            script_response = requests.get(
                script_url,
                headers=headers,
                timeout=20
            )
            script_response.raise_for_status()

            combined_text = html + "\n" + script_response.text

            app_match = re.search(
                r'ALGOLIA_APPLICATION_ID\s*[:=]\s*["\']([^"\']+)',
                combined_text
            )

            key_match = re.search(
                r'ALGOLIA_SEARCH_ONLY_API_KEY\s*[:=]\s*["\']([^"\']+)',
                combined_text
            )

            index_match = re.search(
                r'HEALTH_INDEX_NAME_DEFAULT\s*=\s*["\']([^"\']+)',
                combined_text
            )

            if not all([app_match, key_match, index_match]):
                print("Cleveland Clinic Algolia configuration not found.")
                return []

            app_id = app_match.group(1)
            api_key = key_match.group(1)
            index_name = index_match.group(1)

            # Step 3: Search Algolia
            search_url = (
                f"https://{app_id}-dsn.algolia.net/1/indexes/"
                f"{index_name}/query"
            )

            search_response = requests.post(
                search_url,
                headers={
                    "X-Algolia-Application-Id": app_id,
                    "X-Algolia-API-Key": api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "query": query,
                    "hitsPerPage": top_k * 10
                },
                timeout=20
            )

            print(
                "Cleveland Clinic status:",
                search_response.status_code
            )

            search_response.raise_for_status()

            data = search_response.json()
            hits = data.get("hits", [])

            print("Cleveland Clinic total hits:", data.get("nbHits"))

            # Step 4: Extract relevant medical articles
            results = []
            seen_urls = set()

            for hit in hits:

                page_url = hit.get("url", "").strip()

                if not page_url.startswith(BASE_URL + "/health/"):
                    continue

                if page_url in seen_urls:
                    continue

                seen_urls.add(page_url)

                title = (
                    hit.get("meta title")
                    or hit.get("content heading")
                    or "Cleveland Clinic Medical Article"
                )

                rendered_content = hit.get("rendered content", "")

                if not rendered_content:
                    rendered_content = hit.get("meta description", "")

                soup = BeautifulSoup(
                    rendered_content,
                    "html.parser"
                )

                for element in soup.find_all([
                    "script", "style", "nav", "footer"
                ]):
                    element.decompose()

                content = soup.get_text(
                    separator=" ",
                    strip=True
                )

                if len(content) < 200:
                    continue
                # Strict relevance filtering using title and meta description

                stop_words = {
                    "what", "is", "are", "the", "of", "and", "how",
                    "it", "does", "do", "can", "for", "with", "in",
                    "on", "a", "an", "to", "be", "about",
                    "symptom", "symptoms", "diagnosis", "diagnosed",
                    "diagnose", "treatment", "treatments", "cause",
                    "causes", "signs", "condition"
                }

                query_keywords = {
                    word.lower()
                    for word in re.findall(r"\b[a-zA-Z]+\b", query)
                    if word.lower() not in stop_words
                }

                title_description = (
                    title + " " + hit.get("meta description", "")
                ).lower()

                matched_keywords = [
                    word for word in query_keywords
                    if re.search(
                        r"\b" + re.escape(word) + r"\b",
                        title_description
                    )
                ]

                print("Checking:", title)
                print("Matched title keywords:", matched_keywords)

                # Match the user's medical intent
                intent_groups = {
                    "symptoms": ["symptom", "sign"],
                    "diagnosis": ["diagnos", "test", "screen"]
                }

                query_intents = [
                    group for group, terms in intent_groups.items()
                    if any(term in query.lower() for term in terms)
                ]

                article_intent_text = title_description.lower()

                matched_intents = [
                    group for group, terms in intent_groups.items()
                    if any(term in article_intent_text for term in terms)
                ]

                print("Query intents:", query_intents)
                print("Matched intents:", matched_intents)
                
                required_matches = min(1, len(query_keywords))
                if required_matches and len(matched_keywords) < required_matches:
                    print("Skipped irrelevant article:", title)
                    continue
                # Match the user's medical intent
                intent_groups = {
                    "symptoms": ["symptom", "sign"],
                    "diagnosis": ["diagnos", "test", "screen"]
                }

                query_intents = [
                    group for group, terms in intent_groups.items()
                    if any(term in query.lower() for term in terms)
                ]

                article_intent_text = title_description.lower()

                matched_intents = [
                    group for group, terms in intent_groups.items()
                    if any(term in article_intent_text for term in terms)
                ]

                print("Query intents:", query_intents)
                print("Matched intents:", matched_intents)

                required_matches = min(1, len(query_keywords))
                if required_matches and len(matched_keywords) < required_matches:
                    print("Skipped irrelevant article:", title)
                    continue

                if query_intents and not any(
                    intent in matched_intents for intent in query_intents
                ):
                    print("Skipped article: medical intent mismatch:", title)
                    continue
                if query_intents and not any(
                    intent in matched_intents for intent in query_intents
                ):
                    print("Skipped article: medical intent mismatch:", title)
                    continue

                
                results.append({
                    "id": page_url,
                    "title": title,
                    "content": content[:8000],
                    "source": "Cleveland Clinic",
                    "category": hit.get("type", "medical_information"),
                    "url": page_url
                })

                print("Cleveland Clinic article:", title)

                if len(results) >= top_k:
                    break

            print(
                "Cleveland Clinic live retrieval:",
                len(results),
                "documents"
            )

            return results

        except requests.RequestException as exc:
            print("Cleveland Clinic retrieval failed:", exc)
            return []

        except Exception as exc:
            print("Cleveland Clinic processing error:", exc)
            return []
    
    def search_who(self, query: str, top_k: int = 5):
        import requests
        import re
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        documents = []

        try:
            api_url = "https://www.who.int/api/default/healthtopics"

            stop_words = {
                "what", "is", "are", "the", "of", "and", "how",
                "it", "does", "do", "can", "for", "with", "in",
                "on", "a", "an", "to", "be", "about", "symptoms",
                "diagnosis", "diagnosed", "diagnose", "treatment",
                "causes", "cause", "signs"
            }

            query_keywords = {
                word.lower()
                for word in re.findall(r"\b[a-zA-Z]+\b", query)
                if word.lower() not in stop_words
            }

            if not query_keywords:
                return documents

            # Search WHO using actual query keywords
            filter_conditions = [
                f"contains(tolower(Title),'{word}')"
                for word in query_keywords
            ]

            odata_filter = " or ".join(filter_conditions)

            response = requests.get(
                api_url,
                params={
                    "$filter": odata_filter,
                    "$select": "Title,Summary,ItemDefaultUrl,MetaDescription",
                    "$top": 50
                },
                timeout=30
            )

            response.raise_for_status()

            topics = response.json().get("value", [])

            print("WHO matching topics:", len(topics))

            scored_topics = []

            # Rank matching topics by title keyword overlap
            for topic in topics:
                title = topic.get("Title", "")

                title_matches = sum(
                    1 for word in query_keywords
                    if re.search(
                        r"\b" + re.escape(word) + r"\b",
                        title.lower()
                    )
                )

                if title_matches > 0:
                    scored_topics.append((title_matches, topic))

            # Sort only after checking all topics
            scored_topics.sort(
                key=lambda x: x[0],
                reverse=True
            )

            print("\nWHO selected topics:")

            for score, topic in scored_topics[:10]:
                print(
                    "Score:", score,
                    "| Title:", topic.get("Title"),
                    "| URL:", topic.get("ItemDefaultUrl")
                )

            # Retrieve article pages
            for score, topic in scored_topics[:top_k]:

                title = topic.get("Title", "")

                page_url = urljoin(
                    "https://www.who.int",
                    topic.get("ItemDefaultUrl", "")
                )

                print("WHO processing:", title, "|", page_url)

                try:
                    page_response = requests.get(
                        page_url,
                        timeout=20
                    )
                    page_response.raise_for_status()

                except requests.RequestException as e:
                    print("WHO article skipped:", page_url, "|", str(e))
                    continue

                soup = BeautifulSoup(
                    page_response.text,
                    "html.parser"
                )

                body = soup.find("body")

                if not body:
                    print("WHO article has no body:", title)
                    continue

                for element in body.select(
                    "nav, header, footer, script, style"
                ):
                    element.decompose()

                content = body.get_text(" ", strip=True)

                print(
                    "WHO extracted content length:",
                    len(content)
                )

                # Remove text before the article title
                title_position = content.find(title)

                if title_position != -1:
                    content = content[
                        title_position + len(title):
                    ]

                # Remove trailing navigation content
                for marker in ["Fact sheets", "Related topics"]:
                    if marker in content:
                        content = content.split(marker)[0]

                content = re.sub(r"\s+", " ", content).strip()

                if not content:
                    print("WHO article has empty content:", title)
                    continue

                documents.append({
                    "id": page_url,
                    "title": title,
                    "content": content[:8000],
                    "source": "WHO",
                    "category": "medical_information",
                    "url": page_url
                })

                print("WHO article retrieved:", title)

        except Exception as e:
            print("WHO retrieval failed:", str(e))

        print("Total WHO documents retrieved:", len(documents))

        return documents
