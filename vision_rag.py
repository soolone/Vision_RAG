import os
import io
import base64
import PIL
from PIL import Image
import numpy as np
import streamlit as st
import cohere
import google.generativeai as genai
import fitz # PyMuPDF
from typing import List, Dict, Tuple
from collections import defaultdict

# Import query expansion agent
try:
    from agent.query_agent import QueryExpansionAgent
except ImportError:
    QueryExpansionAgent = None

# --- Constants ---
max_pixels = 1568*1568  # Max resolution for images

# --- Helper functions ---

# Resize too large images
def resize_image(pil_image: PIL.Image.Image) -> None:
    """Resizes the image in-place if it exceeds max_pixels."""
    org_width, org_height = pil_image.size

    # Resize image if too large
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

# Convert images to a base64 string before sending it to the API
def base64_from_image(img_path: str) -> str:
    """Converts an image file to a base64 encoded string."""
    pil_image = PIL.Image.open(img_path)
    img_format = pil_image.format if pil_image.format else "PNG"

    resize_image(pil_image)

    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64,"+base64.b64encode(img_buffer.read()).decode("utf-8")

    return img_data

# Convert PIL image to base64 string
def pil_to_base64(pil_image: PIL.Image.Image) -> str:
    """Converts a PIL image to a base64 encoded string."""
    if pil_image.format is None:
        img_format = "PNG"
    else:
        img_format = pil_image.format
    
    resize_image(pil_image)

    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64,"+base64.b64encode(img_buffer.read()).decode("utf-8")

    return img_data

# Compute embedding for an image
@st.cache_data(ttl=3600, show_spinner=False)
def compute_image_embedding(base64_img: str, _cohere_client) -> np.ndarray | None:
    """Computes an embedding for an image using Cohere's Embed-4 model."""
    try:
        api_response = _cohere_client.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            images=[base64_img],
        )
        
        if api_response.embeddings and api_response.embeddings.float:
            return np.asarray(api_response.embeddings.float[0])
        else:
            st.warning("无法获取嵌入向量。API 响应可能为空。")
            return None
    except Exception as e:
        st.error(f"计算嵌入向量时出错：{e}")
        return None

# Process a PDF file: extract pages as images and embed them
# Note: Caching PDF processing might be complex due to potential large file sizes and streams
# We will process it directly for now, but show progress.
def process_pdf_file(pdf_file, cohere_client, base_output_folder="pdf_pages") -> tuple[list[str], list[np.ndarray] | None]:
    """Extracts pages from a PDF as images, embeds them, and saves them.

    Args:
        pdf_file: UploadedFile object from Streamlit.
        cohere_client: Initialized Cohere client.
        base_output_folder: Directory to save page images.

    Returns:
        A tuple containing: 
          - list of paths to the saved page images.
          - list of numpy array embeddings for each page, or None if embedding fails.
    """
    page_image_paths = []
    page_embeddings = []
    pdf_filename = pdf_file.name
    output_folder = os.path.join(base_output_folder, os.path.splitext(pdf_filename)[0])
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Open PDF from stream
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        st.write(f"正在处理 PDF：{pdf_filename} ({len(doc)} 页)")
        pdf_progress = st.progress(0.0)

        for i, page in enumerate(doc.pages()):
            page_num = i + 1
            page_img_path = os.path.join(output_folder, f"page_{page_num}.png")
            page_image_paths.append(page_img_path)

            # Render page to pixmap (image)
            pix = page.get_pixmap(dpi=150) # Adjust DPI as needed for quality/performance
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Save the page image temporarily
            pil_image.save(page_img_path, "PNG")
            
            # Convert PIL image to base64
            base64_img = pil_to_base64(pil_image)
            
            # Compute embedding for the page image
            emb = compute_image_embedding(base64_img, _cohere_client=cohere_client)
            if emb is not None:
                page_embeddings.append(emb)
            else:
                st.warning(f"无法嵌入 {pdf_filename} 的第 {page_num} 页。跳过。")
                # Add a placeholder to keep lists aligned, will be filtered later
                page_embeddings.append(None)

            # Update progress
            pdf_progress.progress((i + 1) / len(doc))

        doc.close()
        pdf_progress.empty() # Remove progress bar after completion
        
        # Filter out pages where embedding failed
        valid_paths = [path for i, path in enumerate(page_image_paths) if page_embeddings[i] is not None]
        valid_embeddings = [emb for emb in page_embeddings if emb is not None]
        
        if not valid_embeddings:
             st.error(f"无法为 {pdf_filename} 生成任何嵌入向量。")
             return [], None

        return valid_paths, valid_embeddings

    except Exception as e:
        st.error(f"处理 PDF {pdf_filename} 时出错：{e}")
        return [], None



# Reciprocal Rank Fusion (RRF) algorithm
def reciprocal_rank_fusion(rankings: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    """Apply Reciprocal Rank Fusion to combine multiple rankings.
    
    Args:
        rankings: List of rankings, each ranking is a list of (index, score) tuples
        k: RRF parameter (smoothing factor)
        
    Returns:
        Fused ranking as list of (index, fused_score) tuples
    """
    rrf_scores = defaultdict(float)
    
    for ranking in rankings:
        for rank, (doc_idx, _) in enumerate(ranking):
            # RRF formula: 1 / (k + rank)
            rrf_scores[doc_idx] += 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed
    
    # Sort by RRF score (descending)
    fused_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_ranking

# Multi-query search with RRF fusion
def multi_query_search(queries: List[str], co_client: cohere.Client, embeddings: np.ndarray, image_paths: list[str], top_k: int = 3) -> List[Tuple[str, float]] | None:
    """Perform multi-query search with RRF fusion.
    
    Args:
        queries: List of query strings
        co_client: Cohere client
        embeddings: Document embeddings
        image_paths: List of image paths
        top_k: Number of top results to return
        
    Returns:
        List of (image_path, score) tuples for top-k results
    """
    if not co_client or embeddings is None or embeddings.size == 0 or not image_paths:
        st.warning("多查询搜索前提条件不满足（客户端、嵌入向量或路径缺失/为空）。")
        return None
    
    if embeddings.shape[0] != len(image_paths):
        st.error(f"嵌入向量数量 ({embeddings.shape[0]}) 与图片路径数量 ({len(image_paths)}) 不匹配。无法执行搜索。")
        return None
    
    try:
        # Get embeddings for all queries
        api_response = co_client.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=queries,
        )
        
        if not api_response.embeddings or not api_response.embeddings.float:
            st.error("获取查询嵌入向量失败。")
            return None
        
        query_embeddings = np.array(api_response.embeddings.float)
        
        # Ensure query embeddings have the correct shape
        if query_embeddings.shape[1] != embeddings.shape[1]:
            st.error(f"查询嵌入向量维度 ({query_embeddings.shape[1]}) 与文档嵌入向量维度 ({embeddings.shape[1]}) 不匹配。")
            return None
        
        # Compute similarities for each query and create rankings
        rankings = []
        for i, query_emb in enumerate(query_embeddings):
            # Compute cosine similarities
            cos_sim_scores = np.dot(query_emb, embeddings.T)
            
            # Create ranking (sorted by similarity, descending)
            ranking = [(idx, score) for idx, score in enumerate(cos_sim_scores)]
            ranking.sort(key=lambda x: x[1], reverse=True)
            rankings.append(ranking)
            
            print(f"Query {i+1}: '{queries[i]}' - Top result: {image_paths[ranking[0][0]]} (score: {ranking[0][1]:.4f})")
        
        # Apply RRF fusion
        fused_ranking = reciprocal_rank_fusion(rankings)
        
        # Get the top-k results
        if fused_ranking:
            top_k_results = []
            for i in range(min(top_k, len(fused_ranking))):
                idx = fused_ranking[i][0]
                score = fused_ranking[i][1]
                img_path = image_paths[idx]
                top_k_results.append((img_path, score))
                print(f"RRF Fusion result {i+1}: {img_path} (RRF score: {score:.4f})")
            return top_k_results
        else:
            return None
            
    except Exception as e:
        st.error(f"多查询搜索过程中出错：{e}")
        return None

# Original single query search function
def search(question: str, co_client: cohere.Client, embeddings: np.ndarray, image_paths: list[str], top_k: int = 3, max_img_size: int = 800) -> List[Tuple[str, float]] | None:
    """Finds the most relevant image paths for a given question.
    
    Args:
        question: Query string
        co_client: Cohere client
        embeddings: Document embeddings
        image_paths: List of image paths
        top_k: Number of top results to return
        max_img_size: Maximum image size (unused)
        
    Returns:
        List of (image_path, score) tuples for top-k results
    """
    if not co_client or embeddings is None or embeddings.size == 0 or not image_paths:
        st.warning("搜索前提条件不满足（客户端、嵌入向量或路径缺失/为空）。")
        return None
    if embeddings.shape[0] != len(image_paths):
         st.error(f"嵌入向量数量 ({embeddings.shape[0]}) 与图片路径数量 ({len(image_paths)}) 不匹配。无法执行搜索。")
         return None

    try:
        # Compute the embedding for the query
        api_response = co_client.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[question],
        )

        if not api_response.embeddings or not api_response.embeddings.float:
            st.error("获取查询嵌入向量失败。")
            return None

        query_emb = np.asarray(api_response.embeddings.float[0])

        # Ensure query embedding has the correct shape for dot product
        if query_emb.shape[0] != embeddings.shape[1]:
            st.error(f"查询嵌入向量维度 ({query_emb.shape[0]}) 与文档嵌入向量维度 ({embeddings.shape[1]}) 不匹配。")
            return None

        # Compute cosine similarities
        cos_sim_scores = np.dot(query_emb, embeddings.T)

        # Get the top-k most relevant images
        top_indices = np.argsort(cos_sim_scores)[::-1][:top_k]  # Sort descending and take top-k
        top_k_results = []
        
        print(f"Question: {question}") # Keep for debugging
        for i, idx in enumerate(top_indices):
            img_path = image_paths[idx]
            score = cos_sim_scores[idx]
            top_k_results.append((img_path, score))
            print(f"Result {i+1}: {img_path} (score: {score:.4f})") # Keep for debugging

        return top_k_results
    except Exception as e:
        st.error(f"搜索过程中出错：{e}")
        return None

# Answer function for multiple images
def answer_multiple(question: str, img_results: List[Tuple[str, float]], gemini_client) -> str:
    """Answers the question based on multiple provided images using Gemini."""
    if not gemini_client or not img_results:
        missing = []
        if not gemini_client: missing.append("Gemini client")
        if not img_results: missing.append("Image results")
        return f"回答前提条件不满足（{', '.join(missing)} 缺失或无效）。"
    
    try:
        # Prepare images and prompt
        valid_images = []
        image_info = []
        
        for i, (img_path, score) in enumerate(img_results):
            if os.path.exists(img_path):
                img = PIL.Image.open(img_path)
                valid_images.append(img)
                
                # Create image description for prompt
                source_info = os.path.basename(img_path)
                if img_path.startswith("pdf_pages/"):
                    parts = img_path.split(os.sep)
                    if len(parts) >= 3:
                        pdf_name = parts[1]
                        page_name = parts[-1]
                        source_info = f"{pdf_name}.pdf，{page_name.replace('.png','')}"
                
                image_info.append(f"图片{i+1}（来源：{source_info}，相关度：{score:.3f}）")
        
        if not valid_images:
            return "没有找到有效的图片文件。"
        
        # Create comprehensive prompt
        prompt_text = f"""请基于以下{len(valid_images)}张相关图片回答问题。请综合分析所有图片的内容，提供全面详细的答案。

问题：{question}

图片信息：
{chr(10).join(image_info)}

请注意：
1. 综合分析所有图片的内容
2. 如果图片内容相关，请整合信息提供完整答案
3. 如果图片内容不同，请分别说明每张图片的相关信息
4. 请用中文回答，对于专业术语请使用准确的中文表达
5. 不要使用markdown格式
6. 提供充分的上下文背景"""
        
        # Prepare content for API call
        prompt_content = [prompt_text] + valid_images
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=prompt_content
        )
        
        llm_answer = response.text
        print(f"LLM Answer for {len(valid_images)} images:", llm_answer) # Keep for debugging
        return llm_answer
        
    except Exception as e:
        st.error(f"生成答案时出错：{e}")
        return f"生成答案失败：{e}"

# Answer function (single image, kept for compatibility)
def answer(question: str, img_path: str, gemini_client) -> str:
    """Answers the question based on the provided image using Gemini."""
    return answer_multiple(question, [(img_path, 1.0)], gemini_client)




def initialize_session_state():
    """初始化Streamlit会话状态变量"""
    if 'image_paths' not in st.session_state:
        st.session_state.image_paths = []
    if 'doc_embeddings' not in st.session_state:
        st.session_state.doc_embeddings = None
    if 'use_query_expansion' not in st.session_state:
        st.session_state.use_query_expansion = True
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'num_results' not in st.session_state:
        st.session_state.num_results = 3

def setup_sidebar():
    """设置侧边栏UI并返回API密钥和设置"""
    with st.sidebar:
        st.header("🔑 API 密钥")
        cohere_api_key = st.text_input("Cohere API 密钥", type="password", key="cohere_key")
        google_api_key = st.text_input("Google API 密钥 (Gemini)", type="password", key="google_key")
        "[获取 Cohere API 密钥](https://dashboard.cohere.com/api-keys)"
        "[获取 Google API 密钥](https://aistudio.google.com/app/apikey)"

        st.markdown("---")
        if not cohere_api_key:
            st.warning("请输入您的 Cohere API 密钥以继续。")
        if not google_api_key:
            st.warning("请输入您的 Google API 密钥以继续。")
        
        st.markdown("---")
        st.header("🔧 高级设置")
        use_query_expansion = st.checkbox(
            "启用查询扩展", 
            value=st.session_state.use_query_expansion,
            help="使用AI生成多个查询变体并融合结果，提高检索准确性"
        )
        st.session_state.use_query_expansion = use_query_expansion
        
        if use_query_expansion and QueryExpansionAgent is None:
            st.warning("⚠️ 查询扩展代理不可用，将使用原始查询模式")
        
        # Add setting for number of results to return
        num_results = st.slider(
            "返回结果数量",
            min_value=1,
            max_value=10,
            value=st.session_state.num_results,
            help="设置检索返回的相关图片数量"
        )
        st.session_state.num_results = num_results
        
        st.markdown("---")
    
    return cohere_api_key, google_api_key

def initialize_api_clients(cohere_api_key, google_api_key):
    """初始化API客户端"""
    co = None
    genai_client = None
    query_agent = None
    
    if cohere_api_key and google_api_key:
        try:
            co = cohere.ClientV2(api_key=cohere_api_key)
            st.sidebar.success("Cohere 客户端初始化成功！")
        except Exception as e:
            st.sidebar.error(f"Cohere 初始化失败：{e}")

        try:
            genai_client = genai.Client(api_key=google_api_key)
            st.sidebar.success("Gemini 客户端初始化成功！")
        except Exception as e:
            st.sidebar.error(f"Gemini 初始化失败：{e}")
        
        # Initialize query expansion agent
        if st.session_state.use_query_expansion and QueryExpansionAgent:
            try:
                query_agent = QueryExpansionAgent()
                st.sidebar.success("查询扩写代理初始化成功！")
            except Exception as e:
                st.sidebar.error(f"查询扩写代理初始化失败：{e}")
                query_agent = None
    else:
        st.info("请在侧边栏输入您的 API 密钥以开始使用。")
    
    return co, genai_client, query_agent

def show_model_info():
    """显示模型和技术信息"""
    with st.expander("ℹ️ 关于使用的模型和技术"):
        st.markdown("""
        ### Cohere Embed-4
        
        Cohere 的 Embed-4 是一个为企业搜索和检索设计的最先进的多模态嵌入模型。
        它具有以下功能：
        
        - **多模态搜索**：无缝地同时搜索文本和图像
        - **高精度**：在检索任务中具有最先进的性能
        - **高效嵌入**：处理复杂图像，如图表、图形和信息图
        
        该模型无需复杂的 OCR 预处理即可处理图像，并保持视觉元素与文本之间的连接。
        
        ### Google Gemini 2.5 Flash
        
        Gemini 2.5 Flash 是 Google 的高效多模态模型，可以处理文本和图像输入以生成高质量的响应。
        它专为快速推理而设计，同时保持高精度，非常适合像这个 RAG 系统这样的实时应用。
        
        ### 查询扩展与多查询融合 🔍
        
        本系统集成了先进的查询优化技术：
        
        - **智能查询扩展**：使用 Qwen3-235B-A22B 模型自动生成多个英文查询变体
        - **语义多样化**：通过同义词、相关概念和不同表达方式扩展查询
        - **RRF 融合算法**：采用 Reciprocal Rank Fusion 算法融合多个查询结果
        - **提升检索精度**：显著提高检索相关性和准确性，减少词汇不匹配问题
        
        启用查询扩展后，系统会自动生成 3-5 个查询变体，并使用 RRF 算法融合结果，为您提供更精准的答案。
        """)

def handle_file_upload(co):
    """处理文件上传逻辑"""
    st.subheader("📤 上传您的图片")
    st.info("上传您的图片或 PDF 文件。RAG 过程将在所有已加载的内容中进行搜索。")

    # File uploader
    uploaded_files = st.file_uploader("上传图片 (PNG, JPG, JPEG) 或 PDF 文件", 
                                    type=["png", "jpg", "jpeg", "pdf"], 
                                    accept_multiple_files=True, key="image_uploader",
                                    label_visibility="collapsed")

    # Show uploaded files info and process button
    if uploaded_files:
        st.write(f"已选择 {len(uploaded_files)} 个文件")
        
        # Show file list
        for uploaded_file in uploaded_files:
            file_status = "✅ 已处理" if uploaded_file.name in st.session_state.processed_files else "⏳ 待处理"
            st.write(f"- {uploaded_file.name} ({uploaded_file.type}) {file_status}")
        
        # Process button - only show if there are unprocessed files and API keys are available
        unprocessed_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if unprocessed_files and co:
            if st.button(f"处理 {len(unprocessed_files)} 个新文件", key="process_files_button"):
                st.write(f"正在处理 {len(unprocessed_files)} 个上传的文件...")
                progress_bar = st.progress(0)
                
                # Create a temporary directory for uploaded images
                upload_folder = "uploaded_img"
                os.makedirs(upload_folder, exist_ok=True)
                
                newly_uploaded_paths = []
                newly_uploaded_embeddings = []

                for i, uploaded_file in enumerate(unprocessed_files):
                    # Check if already processed this session (simple name check)
                    img_path = os.path.join(upload_folder, uploaded_file.name)
                    if img_path not in st.session_state.image_paths:
                        try:
                            # Check file type
                            file_type = uploaded_file.type
                            if file_type == "application/pdf":
                                # Process PDF - returns list of paths and list of embeddings
                                pdf_page_paths, pdf_page_embeddings = process_pdf_file(uploaded_file, cohere_client=co)
                                if pdf_page_paths and pdf_page_embeddings:
                                     # Add only paths/embeddings not already in session state
                                     current_paths_set = set(st.session_state.image_paths)
                                     unique_new_paths = [p for p in pdf_page_paths if p not in current_paths_set]
                                     if unique_new_paths:
                                         indices_to_add = [i for i, p in enumerate(pdf_page_paths) if p in unique_new_paths]
                                         newly_uploaded_paths.extend(unique_new_paths)
                                         newly_uploaded_embeddings.extend([pdf_page_embeddings[idx] for idx in indices_to_add])
                            elif file_type in ["image/png", "image/jpeg"]:
                                # Process regular image
                                # Save the uploaded file
                                with open(img_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                # Get embedding
                                base64_img = base64_from_image(img_path)
                                emb = compute_image_embedding(base64_img, _cohere_client=co)
                                
                                if emb is not None:
                                    newly_uploaded_paths.append(img_path)
                                    newly_uploaded_embeddings.append(emb)
                            else:
                                 st.warning(f"跳过不支持的文件类型：{uploaded_file.name} ({file_type})")
                            
                            # Mark file as processed
                            st.session_state.processed_files.add(uploaded_file.name)

                        except Exception as e:
                            st.error(f"处理 {uploaded_file.name} 时出错：{e}")
                    # Update progress regardless of processing status for user feedback
                    progress_bar.progress((i + 1) / len(unprocessed_files))

                # Add newly processed files to session state
                if newly_uploaded_paths:
                    st.session_state.image_paths.extend(newly_uploaded_paths)
                    if newly_uploaded_embeddings:
                        new_embeddings_array = np.vstack(newly_uploaded_embeddings)
                        if st.session_state.doc_embeddings is None or st.session_state.doc_embeddings.size == 0:
                            st.session_state.doc_embeddings = new_embeddings_array
                        else:
                            st.session_state.doc_embeddings = np.vstack((st.session_state.doc_embeddings, new_embeddings_array))
                        st.success(f"成功处理并添加了 {len(newly_uploaded_paths)} 张新图片。")
                    else:
                         st.warning("为新上传的图片生成嵌入向量失败。")
                
                progress_bar.empty()  # Remove progress bar after completion
                st.rerun()  # Refresh to update file status display
        
        elif not co:
            st.warning("请输入 API 密钥以启用文件处理功能。")
        elif not unprocessed_files:
              st.info("所有文件都已处理完成。")
    else:
        st.info("请选择要上传的文件。")

    # Handle file removal logic
    if not uploaded_files and st.session_state.processed_files:
            # Clear all uploaded file related data
            upload_folder = "uploaded_img"
            indices_to_remove = []
            
            for i, img_path in enumerate(st.session_state.image_paths[:]):
                # Remove uploaded images and PDF pages
                if img_path.startswith(upload_folder) or img_path.startswith("pdf_pages"):
                    indices_to_remove.append(i)
            
            # Remove images and embeddings in reverse order
            for idx in sorted(indices_to_remove, reverse=True):
                if idx < len(st.session_state.image_paths):
                    st.session_state.image_paths.pop(idx)
                    if st.session_state.doc_embeddings is not None and idx < st.session_state.doc_embeddings.shape[0]:
                        st.session_state.doc_embeddings = np.delete(st.session_state.doc_embeddings, idx, axis=0)
            
            # Clear processed files set
            st.session_state.processed_files.clear()
            
            # Clean up empty embeddings array
            if st.session_state.doc_embeddings is not None and st.session_state.doc_embeddings.shape[0] == 0:
                st.session_state.doc_embeddings = None

    # Sync uploaded files with processed data
    if uploaded_files:
        # Get current uploaded file names
        current_uploaded_names = {f.name for f in uploaded_files}
        
        # Remove processed files that are no longer uploaded
        files_to_remove = st.session_state.processed_files - current_uploaded_names
        if files_to_remove:
            # Remove from processed files set
            st.session_state.processed_files -= files_to_remove
            
            # Remove corresponding images and embeddings from session state
            upload_folder = "uploaded_img"
            indices_to_remove = []
            
            for i, img_path in enumerate(st.session_state.image_paths[:]):
                should_remove = False
                
                # Check if this image corresponds to a removed file
                if img_path.startswith(upload_folder):
                    filename = os.path.basename(img_path)
                    if filename in files_to_remove:
                        should_remove = True
                # Also check PDF pages
                elif img_path.startswith("pdf_pages"):
                    # Extract PDF name from path - more precise matching
                    # Path format: pdf_pages/PDF_NAME_WITHOUT_EXTENSION/page_X.png
                    path_parts = img_path.replace("\\", "/").split("/")
                    if len(path_parts) >= 2:
                        pdf_folder_name = path_parts[1]
                        # Check if any removed file matches this PDF folder
                        for removed_file in files_to_remove:
                            if removed_file.endswith('.pdf'):
                                # Get PDF name without extension
                                pdf_name_without_ext = os.path.splitext(removed_file)[0]
                                # Exact match with folder name
                                if pdf_folder_name == pdf_name_without_ext:
                                    should_remove = True
                                    break
                
                if should_remove:
                    indices_to_remove.append(i)
            
            # Remove images and embeddings in reverse order to maintain indices
            for idx in sorted(indices_to_remove, reverse=True):
                if idx < len(st.session_state.image_paths):
                    st.session_state.image_paths.pop(idx)
                    if st.session_state.doc_embeddings is not None and idx < st.session_state.doc_embeddings.shape[0]:
                        st.session_state.doc_embeddings = np.delete(st.session_state.doc_embeddings, idx, axis=0)
            
            # Clean up empty embeddings array
            if st.session_state.doc_embeddings is not None and st.session_state.doc_embeddings.shape[0] == 0:
                st.session_state.doc_embeddings = None
            
            # Show feedback if files were removed
            if indices_to_remove:
                st.info(f"已清理 {len(indices_to_remove)} 张图片（对应已删除的文件）")

def handle_query_interface(co, genai_client, query_agent, cohere_api_key, google_api_key):
    """处理查询界面逻辑"""
    st.markdown("---")
    st.subheader("❓ 提出问题")

    if not st.session_state.image_paths:
        st.warning("请先上传您自己的图片或PDF。")
    else:
        st.info(f"准备回答关于 {len(st.session_state.image_paths)} 张图片的问题。")

        # Display thumbnails of all loaded images (optional)
        with st.expander("查看已加载的图片", expanded=False):
            if st.session_state.image_paths:
                num_images_to_show = len(st.session_state.image_paths)
                cols = st.columns(5) # Show 5 thumbnails per row
                for i in range(num_images_to_show):
                    with cols[i % 5]:
                        # Add try-except for missing files during display
                        try:
                             # Display PDF pages differently? For now, just show the image
                             st.image(st.session_state.image_paths[i], width=100, caption=os.path.basename(st.session_state.image_paths[i]))
                        except FileNotFoundError:
                            st.error(f"缺失：{os.path.basename(st.session_state.image_paths[i])}")
            else:
                st.write("尚未加载任何图片。")

    # Use form to enable Enter key submission
    with st.form(key="rag_form", clear_on_submit=False):
        question = st.text_input("询问关于已加载图片的问题：", 
                                  key="main_question_input",
                                  placeholder="例如：给我讲解下xxx模型的结构？",
                                  disabled=not st.session_state.image_paths)
        
        run_button = st.form_submit_button("运行视觉 RAG", 
                                          disabled=not (cohere_api_key and google_api_key and st.session_state.image_paths and st.session_state.doc_embeddings is not None and st.session_state.doc_embeddings.size > 0))
        
        # Enable form submission when question is entered (even if button is not clicked)
        if question and not run_button:
            run_button = True

    # Output Area
    st.markdown("### 结果")
    retrieved_images_placeholder = st.empty()
    answer_placeholder = st.empty()

    # Run search and answer logic
    if run_button:
        if co and genai_client and st.session_state.doc_embeddings is not None and len(st.session_state.doc_embeddings) > 0:
            # Ensure embeddings and paths match before search
            if len(st.session_state.image_paths) != st.session_state.doc_embeddings.shape[0]:
                st.error("错误：图片数量与嵌入向量数量不匹配。无法继续。")
            else:
                # Determine search strategy
                if st.session_state.use_query_expansion and query_agent:
                    with st.spinner("正在扩展查询..."):
                        try:
                            expanded_queries = query_agent.expand_query_sync(question)
                            if len(expanded_queries) > 1:
                                st.info(f"🔍 查询已扩展为 {len(expanded_queries)} 个变体")
                                with st.expander("查看扩展的查询", expanded=False):
                                    for i, eq in enumerate(expanded_queries, 1):
                                        st.write(f"{i}. {eq}")
                            else:
                                st.info("使用原始查询进行搜索")
                        except Exception as e:
                            st.warning(f"查询扩展失败，使用原始查询：{e}")
                            expanded_queries = [question]
                    
                    with st.spinner("正在执行多查询融合搜索..."):
                        search_results = multi_query_search(expanded_queries, co, st.session_state.doc_embeddings, st.session_state.image_paths, st.session_state.num_results)
                else:
                    with st.spinner("正在查找相关图片..."):
                        search_results = search(question, co, st.session_state.doc_embeddings, st.session_state.image_paths, st.session_state.num_results)

                if search_results:
                    # Display multiple retrieved images
                    with retrieved_images_placeholder.container():
                        st.markdown(f"**检索到 {len(search_results)} 个相关结果：**")
                        
                        # Create columns for displaying images
                        cols = st.columns(min(len(search_results), 3))  # Max 3 columns
                        
                        for i, (img_path, score) in enumerate(search_results):
                            col_idx = i % 3
                            with cols[col_idx]:
                                # Create caption with source info
                                source_info = os.path.basename(img_path)
                                if img_path.startswith("pdf_pages/"):
                                    parts = img_path.split(os.sep)
                                    if len(parts) >= 3:
                                        pdf_name = parts[1]
                                        page_name = parts[-1]
                                        source_info = f"{pdf_name}.pdf，{page_name.replace('.png','')}"
                                
                                caption = f"结果 {i+1}\n来源：{source_info}\n相关度：{score:.3f}"
                                st.image(img_path, caption=caption, use_container_width=True)
                        
                        # If more than 3 results, show remaining in additional rows
                        if len(search_results) > 3:
                            for row_start in range(3, len(search_results), 3):
                                row_end = min(row_start + 3, len(search_results))
                                row_cols = st.columns(row_end - row_start)
                                
                                for i in range(row_start, row_end):
                                    img_path, score = search_results[i]
                                    col_idx = i - row_start
                                    with row_cols[col_idx]:
                                        source_info = os.path.basename(img_path)
                                        if img_path.startswith("pdf_pages/"):
                                            parts = img_path.split(os.sep)
                                            if len(parts) >= 3:
                                                pdf_name = parts[1]
                                                page_name = parts[-1]
                                                source_info = f"{pdf_name}.pdf，{page_name.replace('.png','')}"
                                        
                                        caption = f"结果 {i+1}\n来源：{source_info}\n相关度：{score:.3f}"
                                        st.image(img_path, caption=caption, use_container_width=True)

                    with st.spinner("正在基于多个图片生成综合答案..."):
                        final_answer = answer_multiple(question, search_results, genai_client)
                        answer_placeholder.markdown(f"**综合答案：**\n{final_answer}")
                else:
                    retrieved_images_placeholder.warning("无法找到与您的问题相关的图片。")
                    answer_placeholder.text("") # Clear answer placeholder
        else:
            # This case should ideally be prevented by the disabled state of the button
            st.error("无法运行 RAG。请检查 API 客户端并确保图片已加载并生成嵌入向量。")

def main():
    """主函数 - 应用程序入口点"""
    # --- Streamlit App Configuration ---
    st.set_page_config(layout="wide", page_title="视觉RAG")
    st.title("视觉RAG 🖼️")
    
    # 检查查询扩展代理是否可用
    if QueryExpansionAgent is None:
        st.warning("查询扩写代理导入失败，将使用原始查询模式。")
    
    # 初始化会话状态
    initialize_session_state()
    
    # 设置侧边栏并获取API密钥
    cohere_api_key, google_api_key = setup_sidebar()
    
    # 初始化API客户端
    co, genai_client, query_agent = initialize_api_clients(cohere_api_key, google_api_key)
    
    # 显示模型信息
    show_model_info()
    
    # 处理文件上传
    handle_file_upload(co)
    
    # 处理查询界面
    handle_query_interface(co, genai_client, query_agent, cohere_api_key, google_api_key)
    
    # Footer
    st.markdown("---")
    st.caption("基于 Cohere Embed-4 的视觉检索增强生成 | 使用 Streamlit、Qwen-3、Cohere Embed-4 和 Google Gemini 2.5 Flash 构建")

if __name__ == "__main__":
    main()