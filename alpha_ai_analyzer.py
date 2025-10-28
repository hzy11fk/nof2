# 文件: alpha_ai_analyzer.py (V2 - 包含翻译功能)

import logging
import json
from openai import AsyncAzureOpenAI, AsyncOpenAI
from config import settings

class AlphaAIAnalyzer:
    """
    一个轻量化、专一化的AI模型交互客户端，专为 alpha_trader.py 服务。
    它的唯一职责是管理API连接并发送由 alpha_trader 构建好的Prompt。
    """
    def __init__(self, exchange, strategy_name: str):
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{strategy_name}]")
        
        self.client = None
        self.model_name = None
        self.provider_name = "N/A"

        try:
            provider = getattr(settings, 'AI_PROVIDER', '').lower()

            if provider == 'azure':
                self.logger.info("正在初始化 Async Azure OpenAI 客户端...")
                self.client = AsyncAzureOpenAI(
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_KEY,
                    api_version=getattr(settings, 'AZURE_API_VERSION', '2024-02-01'),
                )
                self.model_name = settings.AZURE_OPENAI_MODEL_NAME
                self.provider_name = "Azure OpenAI"
                self.logger.info(f"✅ Azure OpenAI 客户端已初始化，模型: {self.model_name}")

            elif provider in ['openai', 'deepseek']:
                effective_provider_name = "DeepSeek" if settings.OPENAI_API_BASE and "deepseek" in settings.OPENAI_API_BASE else "OpenAI"
                self.logger.info(f"正在初始化 Async OpenAI 兼容客户端 ({effective_provider_name})...")
                self.client = AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_API_BASE,
                )
                self.model_name = settings.OPENAI_MODEL_NAME
                self.provider_name = effective_provider_name
                self.logger.info(f"✅ {self.provider_name} 客户端已初始化，模型: {self.model_name}")
            
            else:
                self.logger.critical("❌ AI 初始化失败: 配置文件中未找到有效的 'AI_PROVIDER'。")

        except AttributeError as e:
            self.logger.critical(f"❌ AI 初始化失败: 配置文件中缺少凭据！错误: {e}")
            self.client = None

    async def test_connection(self) -> bool:
        """测试与配置的 AI API 的连接性。"""
        if not self.client:
            self.logger.critical("AI 客户端未初始化，跳过连接测试。")
            return False
            
        self.logger.info(f"正在测试与 {self.provider_name} 的连接...")
        try:
            await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                temperature=0.1
            )
            self.logger.info(f"✅ {self.provider_name} 连接测试成功！")
            return True
        except Exception as e:
            self.logger.critical(f"❌ {self.provider_name} 连接测试失败: {e}")
            return False

    async def get_ai_response(self, system_prompt: str, user_prompt_string: str) -> dict:
        """
        接收交易决策的Prompt，发送给AI，并返回解析后的JSON响应。
        """
        if not self.client:
            self.logger.error("无法获取AI响应，因为客户端未初始化。")
            return {}

        self.logger.info(f"正在向 {self.provider_name} 发送分析请求...")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_string}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            response_content = response.choices[0].message.content
            self.logger.info(f"成功接收到AI分析结果: {response_content}")
            return json.loads(response_content)

        except Exception as e:
            self.logger.error(f"调用 {self.provider_name} API 失败 (获取JSON): {e}", exc_info=True)
            return {}

    # [核心新增] 这个方法在您的旧文件中不存在，导致了报错
    async def get_translation_response(self, text_to_translate: str) -> str:
        """
        接收英文文本，调用AI进行翻译，并返回中文结果。
        这是一个独立的、轻量级的调用，不要求JSON格式。
        """
        if not self.client:
            self.logger.error("无法获取翻译响应，因为客户端未初始化。")
            return "[翻译失败: 客户端未初始化]"

        self.logger.info(f"正在向 {self.provider_name} 发送翻译请求...")
        
        try:
            system_prompt = "You are a professional translator. Translate the user's text into concise Chinese. Only return the translated text, without any extra explanations or quotes."
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_to_translate}
                ],
                temperature=0.1, # 翻译任务温度设低一些
            )
            
            translated_text = response.choices[0].message.content
            self.logger.info(f"成功接收到翻译结果: {translated_text}")
            return translated_text.strip()

        except Exception as e:
            self.logger.error(f"调用 {self.provider_name} API 失败 (翻译): {e}", exc_info=True)
            return f"[翻译失败: {e}]"
