"""图片代理模块 - 为 WebUI 提供轻量化的图片缓存服务"""

from .proxy_manager import get_proxy_manager, ProxyManager

__all__ = ["get_proxy_manager", "ProxyManager"]
