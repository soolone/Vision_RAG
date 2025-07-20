import chromadb
import sys
import os
import uuid
import hashlib
from typing import List, Dict, Optional, Any, Union

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.qwen_embedding import TongyiEmbedding

class ChromaDBManager:
    """
    ChromaDB向量数据库管理工具类
    支持向量的增删改查功能，使用UUID作为唯一标识符
    """
    
    def __init__(self, embedding_tool, db_path: str = "./chromadb_data", collection_name: str = "documents"):
        """
        初始化ChromaDB管理器
        
        Args:
            embedding_tool: 嵌入工具实例，如TongyiEmbedding
            db_path: 数据库存储路径
            collection_name: 集合名称
        """
        self.embedding_tool = embedding_tool
        self.db_path = db_path
        self.collection_name = collection_name
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def _generate_embedding(self, text: str, embedding_type: str = 'text', image_path: str = None) -> Optional[List[float]]:
        """
        生成向量表示
        
        Args:
            text: 输入文本
            embedding_type: embedding类型，'text'或'visual'
            image_path: 图片路径（当embedding_type为'visual'时使用）
            
        Returns:
            向量列表或None（如果失败）
        """
        try:
            if embedding_type == 'visual' and image_path:
                # 使用图像embedding
                result = self.embedding_tool.get_image_embedding(image_path)
            else:
                # 使用文本embedding
                result = self.embedding_tool.get_text_embedding(text)
                
            if result.status_code == 200:
                return result.output['embeddings'][0]['embedding']
            else:
                print(f"向量化失败: {result.message}")
                return None
        except Exception as e:
             print(f"向量化出错: {str(e)}")
             return None
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> Optional[str]:
        """
        添加文档到向量数据库
        
        Args:
            text: 文档文本内容
            metadata: 文档元数据
            doc_id: 文档ID，如果不提供则自动生成UUID
            
        Returns:
            文档ID或None（如果失败）
        """
        try:
            # 根据embedding类型生成向量
            embedding_type = metadata.get('embedding_type', 'text') if metadata else 'text'
            image_path = None
            
            # 如果是视觉embedding，需要获取图片路径
            if embedding_type == 'visual':
                # 尝试从不同的metadata字段获取图片路径
                if metadata:
                    image_path = metadata.get('image_path') or metadata.get('table_image_path') or metadata.get('equation_image_path')
                    
                    # 如果是相对路径，需要构建完整路径
                    if image_path and not os.path.isabs(image_path):
                        # 假设相对路径是相对于数据输出目录的
                        base_dir = '/home/rt/proj_lab/Vision_RAG/data/output/default/pcs/pcs/auto'
                        image_path = os.path.join(base_dir, image_path)
            
            embedding = self._generate_embedding(text, embedding_type, image_path)
            if embedding is None:
                return None
            
            # 生成文档ID
            if doc_id is None:
                doc_id = str(uuid.uuid4())
            
            # 准备元数据
            if metadata is None:
                metadata = {}
            metadata.update({
                "text_length": len(text),
                "created_at": str(uuid.uuid1().time)
            })
            
            # 添加到数据库
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            print(f"文档添加成功，ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            print(f"添加文档失败: {str(e)}")
            return None
    
    def add_documents_batch(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, doc_ids: Optional[List[str]] = None) -> List[str]:
        """
        批量添加文档
        
        Args:
            texts: 文档文本列表
            metadatas: 元数据列表
            doc_ids: 文档ID列表，如果不提供则自动生成
            
        Returns:
            成功添加的文档ID列表
        """
        successful_ids = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else None
            doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else None
            
            result_id = self.add_document(text, metadata, doc_id)
            if result_id:
                successful_ids.append(result_id)
        
        print(f"批量添加完成: {len(successful_ids)}/{len(texts)} 成功")
        return successful_ids
    
    def search_documents(self, query: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, embedding_type: str = 'text', image_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            where: 元数据过滤条件
            similarity_threshold: 相似度阈值（0-1之间），距离值小于(1-threshold)的结果会被过滤掉
            embedding_type: embedding类型（'text' 或 'visual'）
            image_path: 图像路径（当embedding_type为'visual'时使用）
            
        Returns:
            搜索结果字典或None（如果失败）
        """
        try:
            # 生成查询向量
            query_embedding = self._generate_embedding(query, embedding_type=embedding_type, image_path=image_path)
            if query_embedding is None:
                return None
            
            # 执行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            # 如果设置了相似度阈值，进行过滤
            if similarity_threshold is not None:
                # 将相似度阈值转换为距离阈值（距离 = 1 - 相似度）
                distance_threshold = 1.0 - similarity_threshold
                
                # 过滤结果
                filtered_documents = []
                filtered_distances = []
                filtered_metadatas = []
                filtered_ids = []
                
                for i, distance in enumerate(results['distances'][0]):
                    if distance <= distance_threshold:
                        filtered_documents.append(results['documents'][0][i])
                        filtered_distances.append(distance)
                        filtered_metadatas.append(results['metadatas'][0][i])
                        filtered_ids.append(results['ids'][0][i])
                
                return {
                    "documents": filtered_documents,
                    "distances": filtered_distances,
                    "metadatas": filtered_metadatas,
                    "ids": filtered_ids
                }
            else:
                return {
                    "documents": results['documents'][0],
                    "distances": results['distances'][0],
                    "metadatas": results['metadatas'][0],
                    "ids": results['ids'][0]
                }
            
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return None
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档信息字典或None（如果不存在）
        """
        try:
            results = self.collection.get(ids=[doc_id])
            
            if results['ids']:
                return {
                    "id": results['ids'][0],
                    "document": results['documents'][0],
                    "metadata": results['metadatas'][0]
                }
            else:
                print(f"文档不存在: {doc_id}")
                return None
                
        except Exception as e:
            print(f"获取文档失败: {str(e)}")
            return None
    
    def update_document(self, doc_id: str, text: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新文档
        
        Args:
            doc_id: 文档ID
            text: 新的文档文本（如果提供）
            metadata: 新的元数据（如果提供）
            
        Returns:
            是否更新成功
        """
        try:
            # 检查文档是否存在
            existing_doc = self.get_document(doc_id)
            if existing_doc is None:
                return False
            
            # 准备更新数据
            update_data = {"ids": [doc_id]}
            
            if text is not None:
                # 如果更新文本，需要重新生成向量
                embedding = self._generate_embedding(text)
                if embedding is None:
                    return False
                update_data["embeddings"] = [embedding]
                update_data["documents"] = [text]
            
            if metadata is not None:
                # 合并现有元数据和新元数据
                existing_metadata = existing_doc["metadata"] or {}
                existing_metadata.update(metadata)
                existing_metadata["updated_at"] = str(uuid.uuid1().time)
                update_data["metadatas"] = [existing_metadata]
            
            # 执行更新
            self.collection.update(**update_data)
            print(f"文档更新成功: {doc_id}")
            return True
            
        except Exception as e:
            print(f"更新文档失败: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            是否删除成功
        """
        try:
            self.collection.delete(ids=[doc_id])
            print(f"文档删除成功: {doc_id}")
            return True
            
        except Exception as e:
            print(f"删除文档失败: {str(e)}")
            return False
    
    def delete_documents_batch(self, doc_ids: List[str]) -> int:
        """
        批量删除文档
        
        Args:
            doc_ids: 文档ID列表
            
        Returns:
            成功删除的文档数量
        """
        success_count = 0
        for doc_id in doc_ids:
            if self.delete_document(doc_id):
                success_count += 1
        
        print(f"批量删除完成: {success_count}/{len(doc_ids)} 成功")
        return success_count
    
    def delete_documents_by_source(self, source_file: str) -> int:
        """
        根据源文件名删除所有相关的文档切片
        
        Args:
            source_file: 源文件名（可以是parent_document或source_file）
            
        Returns:
            成功删除的文档数量
        """
        try:
            # 获取所有文档
            all_docs = self.collection.get()
            
            # 找到匹配的文档ID
            doc_ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                # 检查parent_document或source_file字段
                doc_source = metadata.get('parent_document') or metadata.get('source_file', '')
                if doc_source == source_file:
                    doc_ids_to_delete.append(all_docs['ids'][i])
            
            if not doc_ids_to_delete:
                print(f"未找到源文件为 '{source_file}' 的文档")
                return 0
            
            # 批量删除
            success_count = self.delete_documents_batch(doc_ids_to_delete)
            print(f"已删除源文件 '{source_file}' 的 {success_count} 个切片")
            return success_count
            
        except Exception as e:
            print(f"按源文件删除文档失败: {str(e)}")
            return 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息
        
        Returns:
            集合信息字典
        """
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "db_path": self.db_path
            }
        except Exception as e:
            print(f"获取集合信息失败: {str(e)}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        清空集合中的所有文档
        
        Returns:
            是否清空成功
        """
        try:
            # 删除现有集合
            self.client.delete_collection(name=self.collection_name)
            # 重新创建集合
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print(f"集合已清空: {self.collection_name}")
            return True
            
        except Exception as e:
            print(f"清空集合失败: {str(e)}")
            return False
    
    def generate_content_id(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        基于内容和关键元数据生成唯一ID
        
        Args:
            content: 文档内容
            metadata: 元数据字典
            
        Returns:
            生成的唯一ID
        """
        # 创建用于生成ID的字符串
        id_components = [content]
        
        if metadata:
            # 对于图片，使用原始文件名和尺寸
            if metadata.get('source_type') == 'image':
                if 'original_filename' in metadata:
                    id_components.append(metadata['original_filename'])
                if 'image_size_str' in metadata:
                    id_components.append(metadata['image_size_str'])
            
            # 对于PDF页面，使用PDF文件名和页码
            elif metadata.get('source_type') == 'pdf':
                if 'pdf_filename' in metadata:
                    id_components.append(metadata['pdf_filename'])
                if 'page_number' in metadata:
                    id_components.append(str(metadata['page_number']))
        
        # 生成MD5哈希作为ID
        content_str = '|'.join(id_components)
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()
    
    def check_duplicates(self, contents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        检查内容是否已存在于数据库中
        
        Args:
            contents: 内容列表
            metadatas: 元数据列表
            
        Returns:
            包含重复信息的字典：
            {
                'duplicates': [{'content': str, 'metadata': dict, 'existing_id': str, 'new_id': str}],
                'new_items': [{'content': str, 'metadata': dict, 'new_id': str}]
            }
        """
        duplicates = []
        new_items = []
        
        try:
            for i, content in enumerate(contents):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else None
                new_id = self.generate_content_id(content, metadata)
                
                # 检查是否已存在
                existing_doc = self.get_document(new_id)
                
                if existing_doc:
                    duplicates.append({
                        'content': content,
                        'metadata': metadata,
                        'existing_id': new_id,
                        'new_id': new_id,
                        'existing_doc': existing_doc
                    })
                else:
                    new_items.append({
                        'content': content,
                        'metadata': metadata,
                        'new_id': new_id
                    })
            
            return {
                'duplicates': duplicates,
                'new_items': new_items
            }
            
        except Exception as e:
            print(f"检查重复项失败: {str(e)}")
            return {'duplicates': [], 'new_items': []}
    
    def add_document_with_id(self, text: str, metadata: Optional[Dict[str, Any]] = None, force_overwrite: bool = False, original_content: Optional[str] = None) -> Optional[str]:
        """
        使用内容生成的ID添加文档，支持覆盖已存在的文档
        
        Args:
            text: 文档文本内容
            metadata: 文档元数据
            force_overwrite: 是否强制覆盖已存在的文档
            original_content: 原始内容（Base64编码的图片或原始文本）
            
        Returns:
            文档ID或None（如果失败）
        """
        try:
            # 生成基于内容的ID
            doc_id = self.generate_content_id(text, metadata)
            
            # 检查是否已存在
            existing_doc = self.get_document(doc_id)
            
            if existing_doc and not force_overwrite:
                print(f"文档已存在，ID: {doc_id}，使用force_overwrite=True来覆盖")
                return None
            
            # 生成向量
            embedding = self._generate_embedding(text)
            if embedding is None:
                return None
            
            # 准备元数据
            if metadata is None:
                metadata = {}
            metadata.update({
                "text_length": len(text),
                "content_id": doc_id
            })
            
            # 如果有原始内容，添加到元数据中
            if original_content:
                metadata["original_content"] = original_content
                metadata["has_original_content"] = True
            else:
                metadata["has_original_content"] = False
            
            if existing_doc:
                # 覆盖现有文档
                metadata["updated_at"] = str(uuid.uuid1().time)
                metadata["original_created_at"] = existing_doc['metadata'].get('created_at', str(uuid.uuid1().time))
                
                # 删除旧文档
                self.collection.delete(ids=[doc_id])
            else:
                # 新文档
                metadata["created_at"] = str(uuid.uuid1().time)
            
            # 添加到数据库
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            action = "覆盖" if existing_doc else "添加"
            print(f"文档{action}成功，ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            print(f"添加/覆盖文档失败: {str(e)}")
            return None
    
    def batch_add_with_overwrite(self, items: List[Dict[str, Any]], force_overwrite: bool = False) -> Dict[str, Any]:
        """
        批量添加文档，支持覆盖
        
        Args:
            items: 包含content、metadata和可选original_content的字典列表
            force_overwrite: 是否强制覆盖已存在的文档
            
        Returns:
            处理结果字典
        """
        results = {
            'successful_ids': [],
            'failed_items': [],
            'overwritten_ids': [],
            'new_ids': []
        }
        
        for item in items:
            content = item.get('content', '')
            metadata = item.get('metadata', {})
            original_content = item.get('original_content', None)
            
            # 检查是否已存在
            doc_id = self.generate_content_id(content, metadata)
            existing_doc = self.get_document(doc_id)
            
            result_id = self.add_document_with_id(content, metadata, force_overwrite, original_content)
            
            if result_id:
                results['successful_ids'].append(result_id)
                if existing_doc:
                    results['overwritten_ids'].append(result_id)
                else:
                    results['new_ids'].append(result_id)
            else:
                results['failed_items'].append(item)
        
        print(f"批量处理完成: 成功{len(results['successful_ids'])}, 失败{len(results['failed_items'])}, 覆盖{len(results['overwritten_ids'])}, 新增{len(results['new_ids'])}")
        return results
    
    def get_original_content(self, doc_id: str) -> Optional[str]:
        """
        获取文档的原始内容
        
        Args:
            doc_id: 文档ID
            
        Returns:
            原始内容（Base64编码）或None
        """
        try:
            doc = self.get_document(doc_id)
            if doc and doc.get('metadata', {}).get('has_original_content', False):
                return doc['metadata'].get('original_content')
            return None
        except Exception as e:
            print(f"获取原始内容失败: {str(e)}")
            return None
    
    @staticmethod
    def image_to_base64(image_path: str) -> Optional[str]:
        """
        将图片文件转换为Base64编码字符串
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            Base64编码的图片字符串或None
        """
        try:
            import base64
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"图片转Base64失败: {str(e)}")
            return None
    
    @staticmethod
    def image_to_base64_from_pil(pil_image) -> Optional[str]:
        """
        将PIL图像对象转换为Base64编码字符串
        
        Args:
            pil_image: PIL图像对象
            
        Returns:
            Base64编码的图片字符串或None
        """
        try:
            import base64
            import io
            
            # 确定图片格式
            img_format = pil_image.format if pil_image.format else "PNG"
            
            # 将PIL图像转换为字节流
            with io.BytesIO() as img_buffer:
                pil_image.save(img_buffer, format=img_format)
                img_buffer.seek(0)
                encoded_string = base64.b64encode(img_buffer.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"PIL图片转Base64失败: {str(e)}")
            return None
    
    @staticmethod
    def base64_to_image(base64_string: str, output_path: str) -> bool:
        """
        将Base64编码字符串转换为图片文件
        
        Args:
            base64_string: Base64编码的图片字符串
            output_path: 输出图片文件路径
            
        Returns:
            是否成功
        """
        try:
            import base64
            import os
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 解码并保存
            image_data = base64.b64decode(base64_string)
            with open(output_path, "wb") as image_file:
                image_file.write(image_data)
            return True
        except Exception as e:
            print(f"Base64转图片失败: {str(e)}")
            return False


def test_chromadb_manager():
    """
    测试ChromaDB管理器的功能
    """
    print("=== ChromaDB管理器功能测试 ===")
    
    # 初始化embedding工具和ChromaDB管理器
    embedding_tool = TongyiEmbedding()
    db_manager = ChromaDBManager(embedding_tool, "./test_chromadb_data", "test_collection")
    
    # 清空测试集合
    db_manager.clear_collection()
    
    # 测试数据
    test_documents = [
        "深度学习是机器学习的一个分支，使用多层神经网络模拟人脑工作方式。",
        "机器学习通过算法让计算机从数据中自动学习模式和规律。",
        "自然语言处理专注于让计算机理解和生成人类语言。",
        "计算机视觉让机器能够理解和解释视觉信息。",
        "数据科学结合统计学、计算机科学和领域专业知识。"
    ]
    
    print("\n1. 测试批量添加文档...")
    doc_ids = db_manager.add_documents_batch(test_documents)
    print(f"添加的文档ID: {doc_ids}")
    
    print("\n2. 测试集合信息查询...")
    info = db_manager.get_collection_info()
    print(f"集合信息: {info}")
    
    print("\n3. 测试文档搜索...")
    search_result = db_manager.search_documents("神经网络和深度学习", n_results=3)
    if search_result:
        print("搜索结果:")
        for i, (doc, distance, doc_id) in enumerate(zip(
            search_result['documents'], 
            search_result['distances'], 
            search_result['ids']
        )):
            print(f"  {i+1}. ID: {doc_id}, 距离: {distance:.4f}")
            print(f"     内容: {doc[:60]}...")
    
    print("\n4. 测试单个文档获取...")
    if doc_ids:
        doc_info = db_manager.get_document(doc_ids[0])
        if doc_info:
            print(f"文档信息: ID={doc_info['id']}, 内容={doc_info['document'][:50]}...")
    
    print("\n5. 测试文档更新...")
    if doc_ids:
        success = db_manager.update_document(
            doc_ids[0], 
            metadata={"category": "AI", "updated": True}
        )
        print(f"更新结果: {success}")
    
    print("\n6. 测试文档删除...")
    if len(doc_ids) > 1:
        success = db_manager.delete_document(doc_ids[-1])
        print(f"删除结果: {success}")
    
    print("\n7. 最终集合信息...")
    final_info = db_manager.get_collection_info()
    print(f"最终集合信息: {final_info}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    # 运行测试
    test_chromadb_manager()
    
    print("\n=== 使用示例 ===")
    print("""
    # 基本使用方法:
    from agent.qwen_embedding import TongyiEmbedding
    from agent.chromadb_data import ChromaDBManager
    
    # 1. 初始化
    embedding_tool = TongyiEmbedding()
    db_manager = ChromaDBManager(embedding_tool)
    
    # 2. 添加文档
    doc_id = db_manager.add_document("这是一个测试文档")
    
    # 3. 搜索文档
    results = db_manager.search_documents("测试", n_results=5)
    
    # 4. 获取文档
    doc_info = db_manager.get_document(doc_id)
    
    # 5. 更新文档
    db_manager.update_document(doc_id, metadata={"category": "test"})
    
    # 6. 删除文档
    db_manager.delete_document(doc_id)
    """)