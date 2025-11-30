"""图片转码引擎 - 将各种格式转换为优化的 WebP"""

import os
import asyncio
from typing import Tuple
from PIL import Image
from src.common.logger import get_logger

logger = get_logger("image_proxy.transcoder")


# 配置常量
MAX_ANIMATION_FRAMES = 150  # 限制最大帧数防止内存爆炸


class Transcoder:
    """图片转码引擎"""

    def __init__(self, quality: int = 75):
        self.quality = quality

    async def transcode_to_webp(
        self, source_path: str, target_path: str
    ) -> Tuple[bool, int, int]:
        """将图片转码为 WebP 格式，返回 (success, original_size, webp_size)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._transcode_sync, source_path, target_path
            )
        except Exception as e:
            logger.error(f"转码失败 {source_path}: {e}")
            return False, 0, 0

    def _transcode_sync(
        self, source_path: str, target_path: str
    ) -> Tuple[bool, int, int]:
        """同步转码（在线程池中执行）"""
        temp_path = target_path + ".tmp"
        try:
            original_size = os.path.getsize(source_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            with Image.open(source_path) as img:
                is_animated = getattr(img, "is_animated", False)
                n_frames = getattr(img, "n_frames", 1)

                if is_animated and n_frames > 1:
                    if n_frames > MAX_ANIMATION_FRAMES:
                        logger.warning(
                            f"图片帧数 ({n_frames}) 超过限制 {MAX_ANIMATION_FRAMES}，将仅转码第一帧: {os.path.basename(source_path)}"
                        )
                        self._transcode_static(img, temp_path)
                    else:
                        self._transcode_animated(img, temp_path)
                else:
                    self._transcode_static(img, temp_path)

            webp_size = os.path.getsize(temp_path)

            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(temp_path, target_path)

            logger.debug(
                f"转码完成: {os.path.basename(source_path)} "
                f"({original_size} -> {webp_size}, -{100 - webp_size * 100 // original_size}%)"
            )
            return True, original_size, webp_size

        except Exception as e:
            logger.error(f"转码处理失败 {source_path}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return False, 0, 0

    def _transcode_static(self, img: Image.Image, target_path: str) -> None:
        """转码静态图片"""
        img = self._convert_image_mode(img)
        img.save(target_path, format="WEBP", quality=self.quality, method=4)

    def _transcode_animated(self, img: Image.Image, target_path: str) -> None:
        """转码动态图片（GIF → Animated WebP）"""
        frames = []
        durations = []

        for frame_num in range(img.n_frames):
            img.seek(frame_num)
            frame = self._convert_image_mode(img.copy())

            frames.append(frame)
            durations.append(img.info.get("duration", 100))

        if frames:
            frames[0].save(
                target_path,
                format="WEBP",
                save_all=True,
                append_images=frames[1:] if len(frames) > 1 else [],
                duration=durations,
                loop=img.info.get("loop", 0),
                quality=self.quality,
                method=4,
            )

    def _convert_image_mode(self, img: Image.Image) -> Image.Image:
        """转换图片模式以兼容 WebP"""
        if img.mode in ("P", "LA"):
            return img.convert("RGBA")
        elif img.mode not in ("RGB", "RGBA"):
            return img.convert("RGB")
        return img
