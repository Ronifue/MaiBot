"""代理图片管理器 - 管理图片缓存和转码队列"""

import os
import asyncio
import mimetypes
from typing import Optional, Set, TYPE_CHECKING
from src.common.logger import get_logger
from .transcoder import Transcoder

if TYPE_CHECKING:
    from src.common.database.database_model import Emoji

logger = get_logger("image_proxy.manager")

# 配置常量
PROXY_DIR = os.path.join("data", "emoji_proxy")
WEBP_QUALITY = 75
MAX_CONCURRENT_TRANSCODE = 2
TRANSCODE_INTERVAL = 0.3  # 秒
CLEANUP_INTERVAL = 3600  # 秒

# 支持转码的格式
SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}


class ProxyManager:
    """代理图片管理器"""

    _instance: Optional["ProxyManager"] = None

    def __new__(cls) -> "ProxyManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._transcoder = Transcoder(quality=WEBP_QUALITY)
        self._transcode_queue: asyncio.Queue = asyncio.Queue()
        self._pending_hashes: Set[str] = set()  # 防止重复入队
        self._pending_lock = asyncio.Lock()
        self._workers_started = False
        self._worker_tasks: list = []

        # 确保缓存目录存在
        os.makedirs(PROXY_DIR, exist_ok=True)
        logger.info(f"代理图片管理器已初始化，缓存目录: {PROXY_DIR}")

    def get_cache_path(self, emoji_hash: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(PROXY_DIR, f"{emoji_hash}.webp")

    def get_skip_marker_path(self, emoji_hash: str) -> str:
        """获取跳过标记文件路径"""
        return os.path.join(PROXY_DIR, f"{emoji_hash}.skip")

    def get_cached_proxy_path(self, emoji: "Emoji") -> Optional[str]:
        """获取已缓存的代理图片路径，不存在返回 None"""
        cache_path = self.get_cache_path(emoji.emoji_hash)
        return cache_path if os.path.exists(cache_path) else None

    def should_skip_transcode(self, emoji: "Emoji") -> bool:
        """检查是否应该跳过转码（原图更优）"""
        return os.path.exists(self.get_skip_marker_path(emoji.emoji_hash))

    async def enqueue_transcode(self, emoji: "Emoji") -> bool:
        """将转码任务加入队列（非阻塞）"""
        # 检查格式是否支持转码
        if emoji.format.lower() not in SUPPORTED_FORMATS:
            return False

        async with self._pending_lock:
            # 检查是否已在队列中或已有缓存
            if emoji.emoji_hash in self._pending_hashes:
                return False
            if self.get_cached_proxy_path(emoji) or self.should_skip_transcode(emoji):
                return False
            if not os.path.exists(emoji.full_path):
                return False

            self._pending_hashes.add(emoji.emoji_hash)
            await self._transcode_queue.put((emoji.emoji_hash, emoji.full_path))
            logger.debug(f"已加入转码队列: {emoji.emoji_hash[:8]}...")
            return True

    async def _worker(self, worker_id: int) -> None:
        """后台转码工作协程"""
        logger.debug(f"转码工作协程 #{worker_id} 已启动")
        while True:
            try:
                emoji_hash, source_path = await self._transcode_queue.get()
                try:
                    await self._do_transcode(emoji_hash, source_path)
                except Exception as e:
                    logger.error(f"转码任务异常: {e}")
                finally:
                    async with self._pending_lock:
                        self._pending_hashes.discard(emoji_hash)
                    self._transcode_queue.task_done()
                await asyncio.sleep(TRANSCODE_INTERVAL)
            except asyncio.CancelledError:
                logger.debug(f"转码工作协程 #{worker_id} 已停止")
                break
            except Exception as e:
                logger.error(f"转码工作协程 #{worker_id} 错误: {e}")
                await asyncio.sleep(1)

    async def _do_transcode(self, emoji_hash: str, source_path: str) -> None:
        """执行实际的转码操作"""
        if not os.path.exists(source_path):
            return

        cache_path = self.get_cache_path(emoji_hash)
        skip_path = self.get_skip_marker_path(emoji_hash)

        if os.path.exists(cache_path) or os.path.exists(skip_path):
            return

        success, original_size, webp_size = await self._transcoder.transcode_to_webp(
            source_path, cache_path
        )

        if not success:
            return

        # 压缩后更大，删除 WebP，创建跳过标记
        if webp_size >= original_size:
            logger.debug(f"压缩后体积更大，使用原图: {emoji_hash[:8]}...")
            try:
                os.remove(cache_path)
            except Exception:
                pass
            try:
                with open(skip_path, "w") as f:
                    f.write(f"original:{original_size},webp:{webp_size}")
            except Exception as e:
                logger.error(f"创建跳过标记失败: {e}")
        else:
            logger.debug(f"转码成功: {emoji_hash[:8]}... ({original_size} -> {webp_size})")

    async def start_workers(self) -> None:
        """启动后台转码工作协程"""
        if self._workers_started:
            return
        self._workers_started = True

        for i in range(MAX_CONCURRENT_TRANSCODE):
            self._worker_tasks.append(asyncio.create_task(self._worker(i)))
        self._worker_tasks.append(asyncio.create_task(self._cleanup_worker()))

        logger.info(f"已启动 {MAX_CONCURRENT_TRANSCODE} 个转码工作协程")

    async def stop_workers(self) -> None:
        """停止后台工作协程"""
        if not self._workers_started:
            return
        self._workers_started = False

        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("后台工作协程已停止")

    async def _cleanup_worker(self) -> None:
        """后台清理工作协程"""
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL)
                await self.cleanup_stale_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理任务错误: {e}")

    async def cleanup_stale_cache(self) -> int:
        """清理失效的缓存文件，返回清理数量"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._cleanup_stale_cache_sync)
        except Exception as e:
            logger.error(f"清理缓存任务调度错误: {e}")
            return 0

    def _cleanup_stale_cache_sync(self) -> int:
        """同步执行清理失效缓存文件"""
        try:
            from src.common.database.database_model import Emoji

            # 注意：这里涉及数据库查询和大量文件IO，必须在线程池中运行
            valid_hashes = {e.emoji_hash for e in Emoji.select(Emoji.emoji_hash)}
            cleaned_count = 0

            if not os.path.exists(PROXY_DIR):
                return 0

            for filename in os.listdir(PROXY_DIR):
                hash_part = filename.rsplit(".", 1)[0]
                if hash_part not in valid_hashes:
                    try:
                        os.remove(os.path.join(PROXY_DIR, filename))
                        cleaned_count += 1
                    except Exception:
                        pass

            if cleaned_count > 0:
                logger.info(f"已清理 {cleaned_count} 个失效缓存文件")
            return cleaned_count
        except Exception as e:
            logger.error(f"清理缓存错误: {e}")
            return 0

    async def get_proxy_or_original(self, emoji: "Emoji") -> tuple[str, str]:
        """获取代理图片路径或原图路径，返回 (file_path, media_type)"""
        proxy_path = self.get_cached_proxy_path(emoji)
        if proxy_path:
            return proxy_path, "image/webp"

        # 动态获取 MIME 类型
        mime_type, _ = mimetypes.guess_type(emoji.full_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        if self.should_skip_transcode(emoji):
            return emoji.full_path, mime_type

        await self.enqueue_transcode(emoji)
        return emoji.full_path, mime_type


def get_proxy_manager() -> ProxyManager:
    """获取全局代理图片管理器实例"""
    return ProxyManager()
