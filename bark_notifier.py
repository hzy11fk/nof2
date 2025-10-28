# 文件: bark_notifier.py (新文件)

import logging
import aiohttp
from config import settings

async def send_bark_notification(title: str, body: str):
    """
    发送Bark通知。
    """
    if not settings.BARK_ENABLED or not settings.BARK_KEY:
        return

    # Bark API V2 版本支持更灵活的POST请求
    url = f"https://api.day.app/{settings.BARK_KEY}"
    payload = {
        "title": title,
        "body": body,
        "group": "AI Trader" # 可以为通知分组
    }
    
    logger = logging.getLogger("BarkNotifier")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("Bark 通知发送成功。")
                else:
                    response_text = await response.text()
                    logger.error(f"发送 Bark 通知失败: 状态码 {response.status}, 响应: {response_text}")
    except Exception as e:
        logger.error(f"发送 Bark 通知时发生网络错误: {e}", exc_info=True)
