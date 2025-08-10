from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import TextLoader
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from typing import List, Optional
import os

load_dotenv()


class GraphRAG:
    def __init__(self, neo4j_uri: str = None, neo4j_username: str = None, neo4j_password: str = None, google_api_key: str = None):
        print("[INFO] Initializing GraphRAG...")
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.google_api_key = google_api_key
        self.global_documents = []

        self.llm = self.initialize_llm()
        self.graph = self.initialize_graph()
        self.embeddings = self.initialize_emeddings()
        self.vector_index = self.initialize_vector_index()
        self.entity_chain = self.initialize_entity_chain()
        
        print("[INFO] GraphRAG initialization complete.\n")

    def initialize_llm(self):
        print("[INFO] Initializing Google Generative AI LLM...")
        return ChatGoogleGenerativeAI(
            temperature=0.1,
            model="gemini-2.0-flash-lite",
            google_api_key=self.google_api_key
        )

    def initialize_graph(self):
        print("[INFO] Initializing Neo4j graph connection...")
        graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password
        )

        graph.refresh_schema()
        print("[INFO] Refreshed Neo4j schema.")

        try:
            graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
            print("[INFO] Ensured fulltext index 'entity' exists.")
        except Exception as e:
            print(f"[ERROR] Failed to create fulltext index: {e}")

        return graph
    
    def initialize_emeddings(self):
        print("[INFO] Initializing embeddings...")
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )
            print("[INFO] Embeddings initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize embeddings: {e}")
            raise e
        return embeddings

    def initialize_vector_index(self):
        print("[INFO] Initializing vector index with Google Generative Embeddings...")
        try:
            vector_index = Neo4jVector.from_existing_graph(
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                embedding=self.embeddings,
                search_type="hybrid",
                node_label="Document",
                text_node_properties=["text"],
                embedding_node_property="embedding",
            )
            print("[INFO] Vector index initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize vector index: {e}")
            raise e
        return vector_index

    def initialize_entity_chain(self):
        print("[INFO] Creating entity extraction chain...")

        class Entities(BaseModel):
            names: List[str] = Field(
                default=[],
                description="All entities mentioned in the query, including persons (e.g., '46-year-old male', 'John Doe'), organizations (e.g., 'XYZ Insurance', 'Apollo Hospital'), locations (e.g., 'Pune', 'Maharashtra'), medical procedures (e.g., 'knee surgery', 'MRI scan'), conditions (e.g., 'diabetes', 'age > 40' inferred from '46-year-old'), policy types (e.g., 'health insurance', 'comprehensive plan'), monetary amounts (e.g., '$5000 payout', '1000 INR deductible'), dates or durations (e.g., '3-month policy', 'January 2025'), defined terms or concepts (e.g., 'pre-existing condition', 'deductible'), and additional relevant details (e.g., 'emergency procedure', 'coverage inquiry'). Preserve full context for each entity as it appears in the query to support policy coverage decisions, term definitions, and general information retrieval. Handle ambiguous terms (e.g., 'deductible' as amount or concept) using query context, and include inferred entities (e.g., 'age > 40') for policy criteria."
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting entities from natural language queries related to insurance policies, legal documents, and general information. Your task is to identify and categorize entities from the query to support both specific policy-related queries (e.g., coverage decisions) and general-purpose queries (e.g., definitions, summaries).

            **Instructions**:
            - Extract entities in the following categories: 
            - Persons (e.g., names, descriptors like '46-year-old male')
            - Organizations (e.g., insurers, hospitals like 'XYZ Insurance')
            - Locations (e.g., cities, facilities like 'Pune')
            - Medical procedures (e.g., 'knee surgery', 'cardiac bypass')
            - Conditions (e.g., medical diagnoses, policy criteria like 'diabetes', 'age > 40')
            - Policy types (e.g., 'health insurance', 'comprehensive plan')
            - Monetary amounts (e.g., '$5000 payout', '1000 INR deductible')
            - Dates or durations (e.g., '3-month policy', 'January 2025')
            - Defined terms or concepts (e.g., 'pre-existing condition', 'deductible')
            - Additional relevant details (e.g., 'emergency procedure', 'elective surgery')
            - Preserve the full context of each entity as it appears in the query (e.g., '46-year-old male' instead of just '46').
            - Handle ambiguous terms by using context to determine their category (e.g., 'deductible' as a monetary amount or a defined term).
            - Infer implicit entities when not explicitly stated (e.g., 'age > 40' as a condition from '46-year-old').
            - Retain exact values and context for numerical or temporal entities (e.g., '3-month policy' as a duration, '$5000' as a payout).
            - Avoid duplicating entities; merge similar mentions (e.g., 'knee surgery' and 'surgery on knee' as one procedure).
            - Capture nuances like urgency or intent in the additional details category (e.g., 'emergency' for procedure urgency).
            - Ensure extracted entities are precise and relevant to support downstream graph queries for coverage decisions, term definitions, or general information retrieval.
            - For multilingual queries, extract entities in the query's primary language and note the language if relevant.

            **Examples**:
            - Query: "Is knee surgery covered for a 46-year-old male in Pune with a 3-month health policy?"
            - Persons: ['46-year-old male']
            - Locations: ['Pune']
            - Medical Procedures: ['knee surgery']
            - Policy Types: ['health policy']
            - Dates/Durations: ['3-month']
            - Conditions: ['age > 40'] (inferred)
            - Query: "What is a pre-existing condition?"
            - Terms: ['pre-existing condition']
            - Query: "What are the exclusions for XYZ Insurance policy?"
            - Organizations: ['XYZ Insurance']
            - Terms: ['exclusions']
            - Query: "What is a pre-existing condition, and is it covered for a 46-year-old?"
            - Persons: ['46-year-old']
            - Conditions: ['age > 40'] (inferred)
            - Terms: ['pre-existing condition']
            - Additional Details: ['coverage']
            """),
            ("human", "Extract entities from the following query:\n{question}")
        ])
        return prompt | self.llm.with_structured_output(Entities)

    def load_document(self, file_path):
        documents = []

        def load_single_file(file):
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(file)
            elif file.lower().endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file)
            elif file.lower().endswith(".txt"):
                loader = TextLoader(file)
            else:
                print(f"[WARN] Skipping unsupported file type: {file}")
                return []
            try:
                return loader.load()
            except Exception as e:
                print(f"[ERROR] Failed to load {file}: {e}")
                return []

        if os.path.isdir(file_path):
            print(f"[INFO] Loading files from directory: {file_path}")
            for root, _, files in os.walk(file_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    documents.extend(load_single_file(full_path))
        elif os.path.isfile(file_path):
            print(f"[INFO] Loading single file: {file_path}")
            documents.extend(load_single_file(file_path))
        else:
            print(f"[ERROR] Provided path is not valid: {file_path}")
            return []

        print(f"[INFO] Loaded {len(documents)} documents.")
        self.global_documents.extend(documents)
        return documents

    def split_documents(self, documents):
        print("[INFO] Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunks = text_splitter.split_documents(documents)
        print(f"[INFO] Split into {len(chunks)} text chunks.")
        return chunks

    def change_to_graph_documents(self, documents):
        additional_instructions = """
        Convert documents into a knowledge graph for insurance, legal, and general document processing. Follow these instructions to extract entities and relationships:
        1. **Entity Extraction**:
        - Identify key concepts and objects in the document, such as documents, sections, people, organizations, locations, policies, medical procedures, conditions, monetary amounts, time periods, benefits, exclusions, defined terms, and general terms or phrases.
        - Assign unique identifiers to each entity to ensure traceability and avoid duplication.
        - Capture relevant metadata for each entity, including source document identifiers, page numbers (for paginated documents), and contextual details (e.g., currency for amounts, type of entity like medical or policy-related).
        - For documents, extract attributes like title, type (e.g., PDF, email), and source information.
        - For sections or clauses, preserve their full text and reference their position in the document (e.g., section number or page).
        - For defined terms, extract both the term and its definition, linking them to their source section or document.
        - Handle ambiguous terms by using context to determine their role (e.g., 'deductible' as a financial amount or a policy concept).

        2. **Relationship Extraction**:
        - Identify logical connections between entities based on their roles in the document, such as coverage, exclusions, eligibility requirements, locations, costs, time constraints, processing entities, approvals, definitions, or general associations.
        - Create relationships that reflect policy rules, such as which procedures or conditions are covered or excluded, and what criteria (e.g., age, duration) apply.
        - Link entities to their source documents or sections to ensure traceability.
        - Establish connections between related concepts (e.g., a medical procedure and its associated condition) to support contextual queries.
        - Avoid redundant relationships by merging similar connections (e.g., multiple mentions of the same coverage rule).

        3. **Guidelines**:
        - Prioritize precision in extracting policy-specific details, such as eligibility criteria, coverage limits, exclusions, and waiting periods, to support automated decision-making.
        - Ensure definitions of terms are extracted and linked to their source text for queries asking for explanations or meanings.
        - Capture general document content (e.g., paragraphs or sections) to support queries about non-policy-specific information, such as summaries or background details.
        - Include metadata like document IDs and page numbers in all entities to enable source attribution in responses.
        - Handle numerical and temporal information carefully, preserving exact values (e.g., '$5000', '> 2 months') and their context (e.g., payout vs. premium).
        - Avoid creating duplicate entities for the same concept; use context to merge identical entities (e.g., multiple mentions of the same procedure).
        - Ensure the graph structure supports traceability by linking all entities and relationships to their source document or section.
        - For ambiguous phrases, infer their type based on surrounding text (e.g., is 'premium' a financial amount or a policy term?).
        - Structure the graph to support both specific queries (e.g., policy coverage decisions) and general queries (e.g., term definitions or document summaries).

        4. **Query Support**:
        - Design the graph to answer specific queries about policy coverage, such as whether a procedure is covered for a given individual based on age, location, or policy duration.
        - Support definition queries by extracting and linking terms and their explanations to their source text.
        - Enable general queries about document content, such as summaries, exclusions, or background information, by capturing general sections and their connections.
        - Ensure all responses can reference specific document sections, clauses, or pages for explainability and auditability.
        """
        print("[INFO] Extracting graph documents using LLMGraphTransformer...")
        llm_transformer = LLMGraphTransformer(llm=self.llm,
                                              node_properties=True, 
                                              relationship_properties=True,
                                              additional_instructions=additional_instructions)
        graph_docs = llm_transformer.convert_to_graph_documents(documents)
        print(f"[INFO] Extracted {len(graph_docs)} graph documents.")
        if graph_docs:
            print(f"[DEBUG] Sample graph doc: {graph_docs[0]}")
        return graph_docs

    def update_indices_with_documents(self, documents):
        print("[INFO] Updating graph and vector indices with new documents...")
        
        chunks = self.split_documents(documents)
        graph_documents = self.change_to_graph_documents(documents)
        
        if graph_documents:
            self.graph.add_graph_documents(
                graph_documents=graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
            print("[INFO] Added graph documents to Neo4j.")
        
        if chunks:
            self.vector_index.add_documents(chunks)
            print("[INFO] Added document chunks to vector index.")
        
        self.graph.refresh_schema()
        print("[INFO] Refreshed Neo4j schema.")
        
        result = self.graph.query("MATCH (n) RETURN count(n) AS total_nodes")
        print(f"[INFO] Total nodes in graph: {result[0]['total_nodes']}")

    def add_documents(self, file_path):
        print(f"[INFO] Adding documents from: {file_path}")
        documents = self.load_document(file_path)
        if documents:
            self.update_indices_with_documents(documents)
        else:
            print("[WARNING] No documents found or loaded.")

    def add_documents_from_text(self, text_content, source_name="manual_input"):
        print(f"[INFO] Adding document from text content: {source_name}")
        from langchain_core.documents import Document
        
        document = Document(
            page_content=text_content,
            metadata={"source": source_name}
        )
        self.global_documents.append(document)
        self.update_indices_with_documents([document])

    def add_documents_from_files(self, file_paths):
        print(f"[INFO] Adding documents from {len(file_paths)} files...")
        all_documents = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            else:
                print(f"[WARNING] File does not exist: {file_path}")
        
        if all_documents:
            self.update_indices_with_documents(all_documents)
        else:
            print("[WARNING] No valid documents found in provided files.")
        print("[INFO] Clearing all data from graph and vector index...")
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            print("[INFO] Cleared all nodes from Neo4j graph.")
        except Exception as e:
            print(f"[ERROR] Failed to clear graph: {e}")

    def get_graph_stats(self):
        print("[INFO] Getting graph statistics...")
        try:
            node_count = self.graph.query("MATCH (n) RETURN count(n) AS total_nodes")[0]['total_nodes']
            rel_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) AS total_relationships")[0]['total_relationships']
            entity_count = self.graph.query("MATCH (n:__Entity__) RETURN count(n) AS entity_count")[0]['entity_count']
            doc_count = self.graph.query("MATCH (n:Document) RETURN count(n) AS document_count")[0]['document_count']
            
            stats = {
                'total_nodes': node_count,
                'total_relationships': rel_count,
                'entity_count': entity_count,
                'document_count': doc_count
            }
            
            print(f"[INFO] Graph Stats: {stats}")
            return stats
        except Exception as e:
            print(f"[ERROR] Failed to get graph stats: {e}")
            return {}

    def generate_full_text_query(self, input: str) -> str:
        words = [el for el in remove_lucene_chars(input).split() if el]
        full_text_query = " AND ".join(f"{word}~2" for word in words)
        return full_text_query.strip()
    
    def create_generalized_query(self, question: str) -> str:
        print(f"[INFO] Generating query for question: {question}")
        try:
            prompt = """
            Based on the Graph Schema and the question, create a generalized query that can be further used to extract entities and then that entities can be used to search the graph database to answer the user query effectievly.
            Question: {question}
            Graph Schema: {schema}
            Make sure to not make the query too huge."""

            answer = self.llm.invoke(
                prompt.format(
                    question=question,
                    schema=self.graph.schema
                )
            )
            print(f"[INFO] Generated query for question: {question}: {answer}")
            return answer.strip()
        
        except Exception as e:
            print(f"[ERROR] Failed to generate query: {e}")
            return ""

    def structured_retriever(self, question: str) -> str:
        print(f"[INFO] Running structured retriever for question: {question}")
        result = ""
        generalized_question = question
        entities = self.entity_chain.invoke({"question": generalized_question})
        print(f"[DEBUG] Extracted entities: {entities.names}")

        for entity in entities.names:
            print(f"[DEBUG] Searching graph for entity: {entity}")
            try:
                response = self.graph.query(
                    """
                    CALL db.index.fulltext.queryNodes('entity', $query, {limit:4})
                    YIELD node, score
                    CALL (node) {
                    MATCH (node)-[r]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    MATCH (node)<-[r]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                    }
                    RETURN output
                    LIMIT 70;
                    """,
                    {"query": self.generate_full_text_query(entity)},
                )
                result += "\n".join([el['output'] for el in response])
            except Exception as e:
                print(f"[ERROR] Structured query failed for entity '{entity}': {e}")
        return result

    def retriever(self, question: str):
        print(f"\n[INFO] Performing hybrid retrieval for question: {question}")
        structured_data = self.structured_retriever(question)
        print("[INFO] Structured data retrieval complete.")

        try:
            unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question,k=7)]
            print(f"[INFO] Retrieved {len(unstructured_data)} vector results.")
        except Exception as e:
            print(f"[ERROR] Unstructured retrieval failed: {e}")
            unstructured_data = []

        final_data = f"""Structured Data: 
                        {structured_data}
                        
                        Unstructured Data:
                        {unstructured_data}
                        """
        return final_data

    def create_chain(self):
        print("[INFO] Creating RAG chain for answering questions...")

        # Define a Pydantic model for structured output (optional for policy queries)
        class Response(BaseModel):
            answer: str = Field(description="The natural language answer to the query.")
            decision: Optional[str] = Field(default=None, description="The decision for policy-related queries (e.g., 'Approved', 'Not covered').")
            details: Optional[List[str]] = Field(default=None, description="Relevant details like payout amounts or conditions.")
            sources: Optional[List[str]] = Field(default=None, description="Traceable references like clause numbers or document IDs. or any other citation that supports the answer.")

        # Enhanced prompt with multilingual support
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in processing insurance policies, legal documents, and general information, tasked with answering queries based solely on the provided context. Your goal is to provide accurate, concise, and natural language responses that support both specific policy-related queries (e.g., coverage decisions with clause-based justifications) and general-purpose queries (e.g., term definitions, document summaries).

            **Instructions**:
            - Answer the question using only the information in the provided context, avoiding any external assumptions or knowledge.
            - For policy-related queries (e.g., coverage for a procedure), provide a clear decision (e.g., 'Approved' or 'Not covered'), relevant details (e.g., payout amount, conditions), and cite specific clauses or sections from the context as justification.
            - For definition queries (e.g., 'What is a pre-existing condition?'), provide the exact definition from the context and reference its source (e.g., section or clause).
            - For general queries (e.g., 'What are the policy exclusions?'), summarize the relevant information concisely and cite the source (e.g., document section or page).
            - Include traceable references (e.g., clause number, document ID, page number) in all responses to ensure auditability.
            - Use natural language, keep answers concise, and structure responses clearly (e.g., decision, details, justification, sources).
            - If the context lacks sufficient information to answer, state clearly that the answer is not available and explain why (e.g., 'The context does not specify coverage for this procedure').
            - For multilingual context or queries, process the primary language and note if language-specific nuances affect the answer.

            **Examples**:
            - Context: "Clause 3.2: Knee surgery is covered for patients over 40 with policies older than 2 months, payout $5000."
            - Question: "Is knee surgery covered for a 46-year-old with a 3-month policy?"
            - Answer: "Knee surgery is approved with a payout of $5000, as per Clause 3.2, which covers knee surgery for patients over 40 with policies older than 2 months."
            - Context: "Section 2.1: Pre-existing condition: Any condition diagnosed before policy start."
            - Question: "What is a pre-existing condition?"
            - Answer: "A pre-existing condition is any condition diagnosed before the policy start, as defined in Section 2.1."
            - Context: "Clause 4.1: Exclusions include cosmetic surgery and experimental treatments."
            - Question: "What are the policy exclusions?"
            - Answer: "The policy excludes cosmetic surgery and experimental treatments, as stated in Clause 4.1."
            - Context: ""
            - Question: "Is knee surgery covered?"
            - Answer: "The context does not provide sufficient information to determine if knee surgery is covered."
            """),
            ("human", "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])

        # Validate context and handle errors
        def validate_context(input_dict):
            question = input_dict.get("question", "")
            context = input_dict.get("context", "")
            if not context:
                return {
                    "answer": "The context is empty or insufficient to answer the question.",
                    "decision": None,
                    "details": None,
                    "sources": None
                }
            return {"context": context, "question": question}

        # Normalize input to handle both string and dict
        def normalize_input(input_data):
            if isinstance(input_data, str):
                return {"question": input_data}
            elif isinstance(input_data, dict) and "question" in input_data:
                return input_data
            else:
                raise ValueError("Input must be a string or a dictionary with a 'question' key")

        chain = (
            RunnableLambda(normalize_input)
            | RunnableParallel(
                {
                    "context": RunnableLambda(lambda x: self.retriever(x["question"])),
                    "question": RunnablePassthrough()
                }
            )
            | RunnableLambda(validate_context)
            | prompt
            | self.llm.with_structured_output(Response)
        )
        print("[INFO] Chain created successfully.\n")
        return chain
    
    def flush(self):
        print("[INFO] Flushing all data from graph and vector index...")
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            print("[INFO] Cleared all nodes and relationships from Neo4j graph.")
        except Exception as e:
            print(f"[ERROR] Failed to clear graph: {e}")
        try:
            self.vector_index=self.initialize_vector_index()
            print("[INFO] Cleared all documents from vector index.")
        except Exception as e:
            print(f"[ERROR] Failed to clear vector index: {e}")

