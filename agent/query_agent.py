import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from utils.apis import Qwen3_235B_A22B

class QueryExpansionAgent:
    """基于Qwen3-235B-A22B的查询扩写智能代理"""
    
    def __init__(self):
        # 初始化Qwen模型
        qwen_config = Qwen3_235B_A22B()
        self.llm = ChatOpenAI(
            openai_api_base=qwen_config.api_base,
            openai_api_key=qwen_config.api_key,
            model_name=qwen_config.model,
            temperature=0.3,  # 稍微提高创造性以生成多样化的查询
            max_tokens=1000,
            streaming=False,
            extra_body={
                "enable_thinking": False,  # 禁用思考模式
            }
        )
        
        # 系统提示
        self.system_prompt = """
You are an expert query expansion specialist for multimodal RAG systems. Your task is to rewrite and expand user queries to improve retrieval accuracy in vision-based document search.

Given a user query, you must:
1. Generate 3-5 diverse English query variations that capture different aspects of the original intent
2. Each query should be semantically related but use different vocabulary and phrasing
3. Include synonyms, related concepts, and alternative expressions
4. Consider both technical and layman terms when applicable
5. Ensure queries are optimized for visual document retrieval (charts, diagrams, text in images)

IMPORTANT REQUIREMENTS:
- Output MUST be in English regardless of input language
- Return ONLY a JSON array of strings, no additional text
- Each query should be 5-20 words long
- Avoid redundant or overly similar queries
- Focus on actionable, specific queries that would match visual content

Example input: "给我讲解下transformer模型的结构"
Example output: ["explain transformer model architecture structure", "transformer neural network components diagram", "attention mechanism in transformer models", "encoder decoder transformer design", "transformer block structure visualization"]

Now process the user query:
"""
    
    async def expand_query(self, original_query: str) -> List[str]:
        """扩写查询，生成多个英文查询变体
        
        Args:
            original_query: 原始用户查询
            
        Returns:
            扩写后的查询列表
        """
        try:
            # 构建消息
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=original_query)
            ]
            
            # 调用LLM
            response = await self.llm.ainvoke(messages)
            llm_response = response.content.strip()
            
            # 解析JSON响应
            try:
                # 移除可能的markdown代码块标记
                if llm_response.startswith('```json'):
                    llm_response = llm_response[7:]
                if llm_response.endswith('```'):
                    llm_response = llm_response[:-3]
                
                expanded_queries = json.loads(llm_response)
                
                # 验证返回格式
                if isinstance(expanded_queries, list) and all(isinstance(q, str) for q in expanded_queries):
                    # 过滤空查询并限制数量
                    valid_queries = [q.strip() for q in expanded_queries if q.strip()]
                    return valid_queries[:5]  # 最多返回5个查询
                else:
                    print(f"Invalid response format: {llm_response}")
                    return [original_query]  # 返回原查询作为fallback
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {llm_response}")
                return [original_query]  # 返回原查询作为fallback
                
        except Exception as e:
            print(f"Query expansion error: {e}")
            return [original_query]  # 返回原查询作为fallback
    
    def expand_query_sync(self, original_query: str) -> List[str]:
        """同步版本的查询扩写
        
        Args:
            original_query: 原始用户查询
            
        Returns:
            扩写后的查询列表
        """
        return asyncio.run(self.expand_query(original_query))

# 便捷函数
def create_query_agent() -> QueryExpansionAgent:
    """创建查询扩写代理"""
    return QueryExpansionAgent()

def expand_query_quick(query: str) -> List[str]:
    """快速查询扩写函数"""
    agent = create_query_agent()
    return agent.expand_query_sync(query)

# 测试函数
async def test_query_expansion():
    """测试查询扩写代理"""
    agent = create_query_agent()
    
    test_queries = [
        "给我讲解下transformer模型的结构",
        "什么是注意力机制",
        "深度学习的优化算法有哪些",
        "CNN和RNN的区别",
        "解释一下RAG系统的工作原理"
    ]
    
    for query in test_queries:
        print(f"\n原始查询: {query}")
        print("-" * 50)
        expanded = await agent.expand_query(query)
        for i, exp_query in enumerate(expanded, 1):
            print(f"{i}. {exp_query}")

if __name__ == "__main__":
    asyncio.run(test_query_expansion())