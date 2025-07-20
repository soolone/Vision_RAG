import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

from utils.apis import Qwen25VL72BInstruct, Qwen3_235B_A22B
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# --- Graph State Definition ---
class AnswerState(TypedDict):
    user_question: str
    retrieved_images: List[Dict[str, Any]]  # 检索到的相似图像信息
    image_urls: List[str]  # 图像的data URL列表
    system_prompt: str
    user_prompt_text: str
    llm_response: str
    error_message: str
    usage_metadata: Dict[str, Any]

# --- Helper Functions ---
def image_path_to_data_url(image_path: str) -> str:
    """将图像路径转换为data URL格式"""
    import base64
    from PIL import Image
    import io
    
    try:
        with Image.open(image_path) as img:
            # 转换为RGB格式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整图像大小以避免过大
            max_size = (1024, 1024)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 转换为base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/jpeg;base64,{img_data}"
    except Exception as e:
        print(f"转换图像 {image_path} 时出错: {e}")
        return ""

# --- Graph Nodes ---
async def setup_answer_inputs_node(state: AnswerState) -> AnswerState:
    """设置问答输入节点"""
    print("---SETTING UP ANSWER INPUTS---")
    
    user_question = state["user_question"]
    retrieved_images = state["retrieved_images"]
    
    if not user_question or not user_question.strip():
        error_msg = "错误：用户问题不能为空"
        print(error_msg)
        return {**state, "error_message": error_msg, "image_urls": []}
    
    # 允许没有检索到图像的情况，用于纯文本问答
    has_images = bool(retrieved_images)
    if not has_images:
        print("没有检索到相关图像，将进行纯文本问答")
        retrieved_images = []
    
    # 转换图像为data URL
    image_urls = []
    valid_images = []
    
    for img_info in retrieved_images:
        metadata = img_info.get('metadata', {})
        
        # 优先从ChromaDB获取原始内容
        if metadata.get('has_original_content', False):
            original_content = metadata.get('original_content')
            if original_content:
                try:
                    # 直接使用Base64内容构建data URL
                    data_url = f"data:image/jpeg;base64,{original_content}"
                    image_urls.append(data_url)
                    valid_images.append(img_info)
                except Exception as e:
                    print(f"处理原始内容失败: {e}")
                    continue
        # 如果没有原始内容，尝试本地路径（向后兼容）
        else:
            image_path = metadata.get('image_path', '')
            if image_path and os.path.exists(image_path):
                data_url = image_path_to_data_url(image_path)
                if data_url:
                    image_urls.append(data_url)
                    valid_images.append(img_info)
    
    # 如果没有图像但原本有检索图像，打印警告但继续处理
    if not image_urls and has_images:
        print("警告：无法加载任何检索到的图像，将转为纯文本问答")
        has_images = False  # 转为纯文本模式
    
    # 检查是否为自动扩写的查询
    is_auto_expanded = "自动扩写:" in user_question and "原始问题:" in user_question
    original_question = user_question
    expanded_question = user_question
    
    if is_auto_expanded:
        # 解析自动扩写的查询
        parts = user_question.split("\n\n原始问题: ")
        if len(parts) == 2:
            expanded_part = parts[0].replace("自动扩写: ", "")
            original_part = parts[1]
            original_question = original_part
            expanded_question = expanded_part
    
    # 构建系统提示词
    if image_urls:  # 有图像的情况
        if is_auto_expanded:
            system_prompt = f"""
你是一个专业的视觉问答助手，能够基于用户提供的问题和检索到的相关图像来提供准确、详细的回答。

任务说明：
1. 用户的原始问题："{original_question}"
2. 系统自动扩写的查询："{expanded_question}"
3. 系统基于扩写查询检索到了 {len(valid_images)} 张相关图像
4. 你需要仔细分析这些图像，并基于图像内容回答用户的问题

**重要提示：虽然检索使用了扩写后的查询，但你的回答准确性应该更加关注原始问题。请优先确保原始问题得到准确回答，然后可以适当扩展相关内容。**
"""
        else:
            system_prompt = f"""
你是一个专业的视觉问答助手，能够基于用户提供的问题和检索到的相关图像来提供准确、详细的回答。

任务说明：
1. 用户提出了一个问题："{user_question}"
2. 系统检索到了 {len(valid_images)} 张相关图像
3. 你需要仔细分析这些图像，并基于图像内容回答用户的问题

回答要求：
- 仔细观察和分析提供的图像
- 基于图像内容提供准确、详细的回答
- 如果图像中包含文字，请准确识别和引用
- 如果图像包含图表、数据或技术内容，请详细解释
- 保持回答的逻辑性和条理性
- 如果多张图像内容相关，请综合分析
- 如果图像内容与问题不完全匹配，请说明并提供能够回答的部分
- **重要：当你在回答中使用了某个参考资料的信息时，必须在相关语句后立即标注【参考资料 X】，其中X是对应的参考资料编号**

图像信息：
"""
    else:  # 纯文本问答的情况
        if is_auto_expanded:
            system_prompt = f"""
你是一个专业的问答助手，能够基于用户提供的问题和检索到的相关文本内容来提供准确、详细的回答。

任务说明：
1. 用户的原始问题："{original_question}"
2. 系统自动扩写的查询："{expanded_question}"
3. 系统基于扩写查询检索到了 {len(retrieved_images)} 条相关文本内容
4. 你需要基于这些文本内容回答用户的问题

**重要提示：虽然检索使用了扩写后的查询，但你的回答准确性应该更加关注原始问题。请优先确保原始问题得到准确回答，然后可以适当扩展相关内容。**
"""
        else:
            system_prompt = f"""
你是一个专业的问答助手，能够基于用户提供的问题和检索到的相关文本内容来提供准确、详细的回答。

任务说明：
1. 用户提出了一个问题："{user_question}"
2. 系统检索到了 {len(retrieved_images)} 条相关文本内容
3. 你需要基于这些文本内容回答用户的问题

回答要求：
- 基于检索到的文本内容提供准确、详细的回答
- 保持回答的逻辑性和条理性
- 如果检索内容与问题不完全匹配，请说明并提供能够回答的部分
- 引用相关的文本内容来支持你的回答
- **重要：当你在回答中使用了某个参考资料的信息时，必须在相关语句后立即标注【参考资料 X】，其中X是对应的参考资料编号**

检索到的相关内容：
"""
    
    # 添加内容元数据信息
    if image_urls:  # 有图像的情况
        for i, img_info in enumerate(valid_images, 1):
            metadata = img_info.get('metadata', {})
            document = img_info.get('document', '')
            similarity = 1 - img_info.get('distance', 1)
            
            system_prompt += f"""
参考资料 {i}:
- 来源: {metadata.get('source_file', '未知')}
- 描述: {document}
- 相似度: {similarity:.3f}
- 类型: {metadata.get('source_type', '未知')}
"""
            
            if metadata.get('page_number'):
                system_prompt += f"- 页码: {metadata['page_number']}"
            if metadata.get('image_size_str'):
                system_prompt += f"- 尺寸: {metadata['image_size_str']}"
            system_prompt += "\n"
    else:  # 纯文本的情况
        for i, content_info in enumerate(retrieved_images, 1):
            metadata = content_info.get('metadata', {})
            document = content_info.get('document', '')
            similarity = 1 - content_info.get('distance', 1)
            
            system_prompt += f"""
参考资料 {i}:
- 来源: {metadata.get('source_file', '未知')}
- 内容: {document}
- 相似度: {similarity:.3f}
- 类型: {metadata.get('chunk_type', '未知')}
"""
            
            if metadata.get('page_number'):
                system_prompt += f"- 页码: {metadata['page_number']}"
            system_prompt += "\n"
    
    if image_urls:
        system_prompt += """

请基于以上信息和提供的图像，详细回答用户的问题。记住：当你使用某个参考资料的信息时，必须在相关语句后标注【参考资料 X】。
"""
        if is_auto_expanded:
            user_prompt_text = f"请基于提供的图像回答以下问题：{original_question}"
        else:
            user_prompt_text = f"请基于提供的图像回答以下问题：{user_question}"
    else:
        system_prompt += """

请基于以上检索到的相关内容，详细回答用户的问题。记住：当你使用某个参考资料的信息时，必须在相关语句后标注【参考资料 X】。
"""
        if is_auto_expanded:
            user_prompt_text = f"请基于检索到的相关内容回答以下问题：{original_question}"
        else:
            user_prompt_text = f"请基于检索到的相关内容回答以下问题：{user_question}"
    
    return {
        **state,
        "image_urls": image_urls,
        "system_prompt": system_prompt,
        "user_prompt_text": user_prompt_text,
        "retrieved_images": valid_images,
        "error_message": state.get("error_message", "")
    }

async def invoke_answer_llm_node(state: AnswerState, llm, text_llm=None) -> AnswerState:
    """调用LLM生成回答"""
    print("---INVOKING LLM FOR ANSWER---")
    
    if state.get("error_message"):
        print("跳过LLM调用，因为之前的步骤中存在错误。")
        return {**state, "llm_response": "", "usage_metadata": {}}
    
    # 判断是否有图像
    has_images = bool(state.get("image_urls"))
    current_llm = llm if has_images else (text_llm or llm)
    
    system_message = SystemMessage(content=state["system_prompt"])
    
    if has_images:
        # 构建包含多张图像的human message
        human_message_content = []
        
        # 添加所有图像
        for image_url in state["image_urls"]:
            human_message_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        # 添加文本问题
        human_message_content.append({
            "type": "text",
            "text": state["user_prompt_text"]
        })
        
        human_message = HumanMessage(content=human_message_content)
    else:
        # 纯文本消息
        human_message = HumanMessage(content=state["user_prompt_text"])
    
    try:
        response = await current_llm.ainvoke([system_message, human_message])
        
        # 提取使用元数据
        usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_metadata = {
                "input_tokens": response.usage_metadata.get("input_tokens", 0),
                "output_tokens": response.usage_metadata.get("output_tokens", 0),
                "total_tokens": response.usage_metadata.get("total_tokens", 0)
            }
        elif hasattr(response, 'response_metadata') and response.response_metadata:
            token_usage = response.response_metadata.get('token_usage', {})
            if token_usage:
                usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0)
                }
        
        return {
            **state,
            "llm_response": response.content,
            "usage_metadata": usage_metadata
        }
        
    except Exception as e:
        error_msg = f"调用LLM时出错: {e}"
        print(error_msg)
        return {
            **state,
            "llm_response": "",
            "error_message": ((state.get("error_message") or "") + " " + error_msg).strip(),
            "usage_metadata": {}
        }

# --- Answer Agent Class ---
class AnswerAgent:
    """视觉问答助手
    
    职责：
    1. 接收用户问题和检索到的相似图像
    2. 使用Qwen2.5VL72BInstruct模型分析图像内容
    3. 基于图像内容回答用户问题
    4. 返回详细的回答和使用统计
    
    输入：
    - user_question (str): 用户提出的问题
    - retrieved_images (List[Dict]): 检索到的相似图像信息列表
    
    输出：
    - 包含回答内容、错误信息和使用统计的字典
    """
    
    def __init__(self):
        # 初始化Qwen模型配置
        self.vision_model_config = Qwen25VL72BInstruct()
        self.text_model_config = Qwen3_235B_A22B()
        self.vision_llm = None
        self.text_llm = None
        self.app = None
        self._initialize_llms()
        self._build_workflow()
    
    def _initialize_llms(self):
        """初始化LLM实例"""
        # 视觉模型（用于图像问答）
        self.vision_llm = ChatOpenAI(
            openai_api_base=self.vision_model_config.api_base,
            openai_api_key=self.vision_model_config.api_key,
            model_name=self.vision_model_config.model,
            streaming=False,
            temperature=0.1,  # 较低的温度以获得更准确的回答
            stream_usage=True,
            max_tokens=4096,
            extra_body={
                "vl_high_resolution_images": "True",
                "top_k": 1,
            }
        )
        
        # 文本模型（用于纯文本问答）
        self.text_llm = ChatOpenAI(
            openai_api_base=self.text_model_config.api_base,
            openai_api_key=self.text_model_config.api_key,
            model_name=self.text_model_config.model,
            streaming=False,
            temperature=0.1,
            stream_usage=True,
            max_tokens=4096,
            extra_body={
                "enable_thinking": False,  # 禁用思考模式
            }
        )
    
    def _build_workflow(self):
        """构建workflow图"""
        # 创建包装函数以传递llm实例
        async def invoke_answer_with_instance(state: AnswerState) -> AnswerState:
            return await invoke_answer_llm_node(state, self.vision_llm, self.text_llm)
        
        workflow = StateGraph(AnswerState)
        
        workflow.add_node("setup_inputs", setup_answer_inputs_node)
        workflow.add_node("invoke_llm_answer", invoke_answer_with_instance)
        
        workflow.set_entry_point("setup_inputs")
        workflow.add_edge("setup_inputs", "invoke_llm_answer")
        workflow.add_edge("invoke_llm_answer", END)
        
        self.app = workflow.compile()
    
    async def answer_question(self, user_question: str, retrieved_images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于检索到的图像回答用户问题
        
        Args:
            user_question: 用户提出的问题
            retrieved_images: 检索到的相似图像信息列表
            
        Returns:
            包含回答、错误信息和使用统计的字典
        """
        if not user_question or not user_question.strip():
            return {
                "error": "用户问题不能为空",
                "answer": "",
                "usage_metadata": {}
            }
        
        # 允许retrieved_images为空，支持纯文本问答
        if not retrieved_images:
            print("没有检索到相关图像，将进行纯文本问答")
            retrieved_images = []
        
        initial_state = AnswerState(
            user_question=user_question,
            retrieved_images=retrieved_images,
            image_urls=[],
            system_prompt="",
            user_prompt_text="",
            llm_response="",
            error_message="",
            usage_metadata={}
        )
        
        try:
            final_state = await self.app.ainvoke(initial_state)
            
            usage_metadata = final_state.get("usage_metadata", {})
            
            if final_state.get("error_message") and final_state["error_message"].strip():
                return {
                    "error": final_state["error_message"].strip(),
                    "answer": final_state.get("llm_response", ""),
                    "usage_metadata": usage_metadata
                }
            
            return {
                "error": None,
                "answer": final_state.get("llm_response", ""),
                "usage_metadata": usage_metadata
            }
            
        except Exception as e:
            return {
                "error": f"处理问答请求时出错: {e}",
                "answer": "",
                "usage_metadata": {}
            }

# 测试函数
async def _test_answer_agent():
    """测试问答助手"""
    agent = AnswerAgent()
    
    # 模拟检索结果
    mock_retrieved_images = [
        {
            'document': '测试图像描述',
            'distance': 0.2,
            'metadata': {
                'source_type': 'image',
                'original_filename': 'test.jpg',
                'image_path': '/path/to/test/image.jpg',
                'image_size_str': '800x600'
            },
            'id': 'test_id_1'
        }
    ]
    
    test_question = "这张图片显示了什么内容？"
    
    result = await agent.answer_question(test_question, mock_retrieved_images)
    
    print(f"问题: {test_question}")
    print(f"回答: {result.get('answer', '无回答')}")
    print(f"错误: {result.get('error', '无错误')}")
    print(f"使用统计: {result.get('usage_metadata', {})}")

if __name__ == "__main__":
    asyncio.run(_test_answer_agent())