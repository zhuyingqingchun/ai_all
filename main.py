import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from accelerate import Accelerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GLMAgent:
    def __init__(self, model_path: str = "./models/GLM-4.7-20B"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.agent_executor = None
        self.tools = self._initialize_tools()
        self.memory = []
        self.accelerator = Accelerator()
        
        # 配置内存使用
        torch.cuda.empty_cache()
        logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"设备 {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}GB")
            logger.info(f"  可用显存: {torch.cuda.memory_allocated(i) / 1e9:.2f}GB")
    
    def _initialize_tools(self) -> List[Tool]:
        def search_tool(query: str) -> str:
            """搜索工具：用于查询网络信息"""
            import requests
            from bs4 import BeautifulSoup
            try:
                url = f"https://cn.bing.com/search?q={query}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'lxml')
                results = soup.find_all('li', class_='b_algo')[:3]
                summary = f"搜索结果 for '{query}':\n"
                for i, result in enumerate(results, 1):
                    title = result.find('h2').text if result.find('h2') else "无标题"
                    url = result.find('a')['href'] if result.find('a') else "无链接"
                    summary += f"{i}. {title}\n   {url}\n"
                return summary
            except Exception as e:
                return f"搜索失败: {str(e)}"
        
        def calculate_tool(expression: str) -> str:
            """计算工具：用于执行数学计算"""
            try:
                result = eval(expression)
                return f"计算结果: {expression} = {result}"
            except Exception as e:
                return f"计算失败: {str(e)}"
        
        def weather_tool(city: str) -> str:
            """天气工具：查询城市天气"""
            import requests
            try:
                url = f"https://wttr.in/{city}?format=j1"
                response = requests.get(url, timeout=10)
                data = response.json()
                current = data['current_condition'][0]
                weather = f"{city} 天气:\n"
                weather += f"温度: {current['temp_C']}°C\n"
                weather += f"天气: {current['weatherDesc'][0]['value']}\n"
                weather += f"湿度: {current['humidity']}%\n"
                weather += f"风速: {current['windspeedKmph']} km/h"
                return weather
            except Exception as e:
                return f"天气查询失败: {str(e)}"
        
        tools = [
            Tool(
                name="Search",
                func=search_tool,
                description="搜索网络信息，用于回答需要最新信息的问题"
            ),
            Tool(
                name="Calculate",
                func=calculate_tool,
                description="执行数学计算，用于解决数学问题"
            ),
            Tool(
                name="Weather",
                func=weather_tool,
                description="查询城市天气信息"
            )
        ]
        return tools
    
    def load_model(self):
        """加载GLM-4.7-20B模型"""
        logger.info("正在加载模型...")
        start_time = time.time()
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 配置模型加载参数
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False,
            llm_int8_threshold=6.0
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="balanced",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            max_memory={0: "90GB", 1: "90GB", 2: "90GB", 3: "90GB"},
            quantization_config=quantization_config
        )
        
        # 使用accelerator优化
        self.model = self.accelerator.prepare(self.model)
        
        # 优化模型推理
        self.model.eval()
        
        # 清空缓存
        torch.cuda.empty_cache()
        
        logger.info(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"模型设备: {next(self.model.parameters()).device}")
    
    def _create_agent(self):
        """创建Agent"""
        # 系统提示
        system_prompt = """
你是一个智能助手，能够使用工具解决问题。请遵循以下步骤：
1. 分析用户问题
2. 决定是否需要使用工具
3. 如果需要，选择合适的工具并调用
4. 分析工具返回的结果
5. 生成最终回答

请在回答中保持友好、专业的语气，并确保回答准确、全面。
        """
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # 包装模型为LangChain的LLM
        from langchain.llms import HuggingFacePipeline
        from transformers import pipeline
        
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.95,
            device_map="auto",
            batch_size=1,
            num_return_sequences=1,
            do_sample=True
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # 创建工具调用Agent
        agent = create_tool_calling_agent(llm, self.tools, prompt)
        
        # 创建Agent执行器
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run(self, query: str) -> str:
        """运行Agent处理查询"""
        if not self.agent_executor:
            self._create_agent()
        
        # 记录开始时间和内存使用
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() / 1e9
        
        try:
            # 多步规划和反思闭环
            result = self.agent_executor.invoke({
                "input": query,
                "chat_history": self.memory
            })
            
            # 更新记忆
            self.memory.append(HumanMessage(content=query))
            self.memory.append(AIMessage(content=result['output']))
            
            # 限制记忆长度，防止内存溢出
            if len(self.memory) > 10:
                self.memory = self.memory[-10:]
            
            return result['output']
        finally:
            # 记录结束时间和内存使用
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() / 1e9
            logger.info(f"查询处理时间: {end_time - start_time:.2f}秒")
            logger.info(f"内存使用变化: {start_memory:.2f}GB -> {end_memory:.2f}GB")
            
            # 清空缓存
            torch.cuda.empty_cache()
    
    def clear_memory(self):
        """清空记忆"""
        self.memory = []

def download_model():
    """从国内镜像下载模型"""
    import os
    import subprocess
    
    model_dir = "./models/GLM-4.7-20B"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    # 使用huggingface-cli从国内镜像下载
    print("正在从国内镜像下载模型...")
    try:
        # 设置国内镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        # 下载模型
        subprocess.run([
            "huggingface-cli", "download",
            "THUDM/GLM-4.7-20B",
            "--local-dir", model_dir,
            "--local-dir-use-symlinks", "False"
        ], check=True)
        
        print("模型下载完成！")
    except Exception as e:
        print(f"模型下载失败: {str(e)}")
        sys.exit(1)

def main():
    """主函数"""
    # 下载模型
    download_model()
    
    # 初始化Agent
    agent = GLMAgent()
    agent.load_model()
    
    print("\nGLM-4.7-20B Agent 已启动！")
    print("输入 'exit' 退出\n")
    
    # 交互式对话
    while True:
        query = input("用户: ")
        if query.lower() == 'exit':
            break
        
        start_time = time.time()
        response = agent.run(query)
        end_time = time.time()
        
        print(f"\n助手: {response}")
        print(f"\n处理时间: {end_time - start_time:.2f}秒\n")

if __name__ == "__main__":
    main()
