"""表情包管理 API 路由"""

from fastapi import APIRouter, HTTPException, Header, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Annotated
from src.common.logger import get_logger
from src.common.database.database_model import Emoji
from .token_manager import get_token_manager
import time
import os
import hashlib
from PIL import Image
import io

logger = get_logger("webui.emoji")

# 模块级别的类型别名（解决 B008 ruff 错误）
EmojiFile = Annotated[UploadFile, File(description="表情包图片文件")]
EmojiFiles = Annotated[List[UploadFile], File(description="多个表情包图片文件")]
DescriptionForm = Annotated[str, Form(description="表情包描述")]
EmotionForm = Annotated[str, Form(description="情感标签，多个用逗号分隔")]
IsRegisteredForm = Annotated[bool, Form(description="是否直接注册")]

# 创建路由器
router = APIRouter(prefix="/emoji", tags=["Emoji"])


class EmojiResponse(BaseModel):
    """表情包响应"""

    id: int
    full_path: str
    format: str
    emoji_hash: str
    description: str
    query_count: int
    is_registered: bool
    is_banned: bool
    emotion: Optional[str]  # 直接返回字符串
    record_time: float
    register_time: Optional[float]
    usage_count: int
    last_used_time: Optional[float]


class EmojiListResponse(BaseModel):
    """表情包列表响应"""

    success: bool
    total: int
    page: int
    page_size: int
    data: List[EmojiResponse]


class EmojiDetailResponse(BaseModel):
    """表情包详情响应"""

    success: bool
    data: EmojiResponse


class EmojiUpdateRequest(BaseModel):
    """表情包更新请求"""

    description: Optional[str] = None
    is_registered: Optional[bool] = None
    is_banned: Optional[bool] = None
    emotion: Optional[str] = None


class EmojiUpdateResponse(BaseModel):
    """表情包更新响应"""

    success: bool
    message: str
    data: Optional[EmojiResponse] = None


class EmojiDeleteResponse(BaseModel):
    """表情包删除响应"""

    success: bool
    message: str


class BatchDeleteRequest(BaseModel):
    """批量删除请求"""

    emoji_ids: List[int]


class BatchDeleteResponse(BaseModel):
    """批量删除响应"""

    success: bool
    message: str
    deleted_count: int
    failed_count: int
    failed_ids: List[int] = []


def verify_auth_token(authorization: Optional[str]) -> bool:
    """验证认证 Token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未提供有效的认证信息")

    token = authorization.replace("Bearer ", "")
    token_manager = get_token_manager()

    if not token_manager.verify_token(token):
        raise HTTPException(status_code=401, detail="Token 无效或已过期")

    return True


def emoji_to_response(emoji: Emoji) -> EmojiResponse:
    """将 Emoji 模型转换为响应对象"""
    return EmojiResponse(
        id=emoji.id,
        full_path=emoji.full_path,
        format=emoji.format,
        emoji_hash=emoji.emoji_hash,
        description=emoji.description,
        query_count=emoji.query_count,
        is_registered=emoji.is_registered,
        is_banned=emoji.is_banned,
        emotion=str(emoji.emotion) if emoji.emotion is not None else None,
        record_time=emoji.record_time,
        register_time=emoji.register_time,
        usage_count=emoji.usage_count,
        last_used_time=emoji.last_used_time,
    )


@router.get("/list", response_model=EmojiListResponse)
async def get_emoji_list(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    is_registered: Optional[bool] = Query(None, description="是否已注册筛选"),
    is_banned: Optional[bool] = Query(None, description="是否被禁用筛选"),
    format: Optional[str] = Query(None, description="格式筛选"),
    sort_by: Optional[str] = Query("usage_count", description="排序字段"),
    sort_order: Optional[str] = Query("desc", description="排序方向"),
    authorization: Optional[str] = Header(None),
):
    """
    获取表情包列表

    Args:
        page: 页码 (从 1 开始)
        page_size: 每页数量 (1-100)
        search: 搜索关键词 (匹配 description, emoji_hash)
        is_registered: 是否已注册筛选
        is_banned: 是否被禁用筛选
        format: 格式筛选
        sort_by: 排序字段 (usage_count, register_time, record_time, last_used_time)
        sort_order: 排序方向 (asc, desc)
        authorization: Authorization header

    Returns:
        表情包列表
    """
    try:
        verify_auth_token(authorization)

        # 构建查询
        query = Emoji.select()

        # 搜索过滤
        if search:
            query = query.where((Emoji.description.contains(search)) | (Emoji.emoji_hash.contains(search)))

        # 注册状态过滤
        if is_registered is not None:
            query = query.where(Emoji.is_registered == is_registered)

        # 禁用状态过滤
        if is_banned is not None:
            query = query.where(Emoji.is_banned == is_banned)

        # 格式过滤
        if format:
            query = query.where(Emoji.format == format)

        # 排序字段映射
        sort_field_map = {
            "usage_count": Emoji.usage_count,
            "register_time": Emoji.register_time,
            "record_time": Emoji.record_time,
            "last_used_time": Emoji.last_used_time,
        }

        # 获取排序字段，默认使用 usage_count
        sort_field = sort_field_map.get(sort_by, Emoji.usage_count)

        # 应用排序
        if sort_order == "asc":
            query = query.order_by(sort_field.asc())
        else:
            query = query.order_by(sort_field.desc())

        # 获取总数
        total = query.count()

        # 分页
        offset = (page - 1) * page_size
        emojis = query.offset(offset).limit(page_size)

        # 转换为响应对象
        data = [emoji_to_response(emoji) for emoji in emojis]

        return EmojiListResponse(success=True, total=total, page=page, page_size=page_size, data=data)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取表情包列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取表情包列表失败: {str(e)}") from e


@router.get("/{emoji_id}", response_model=EmojiDetailResponse)
async def get_emoji_detail(emoji_id: int, authorization: Optional[str] = Header(None)):
    """
    获取表情包详细信息

    Args:
        emoji_id: 表情包ID
        authorization: Authorization header

    Returns:
        表情包详细信息
    """
    try:
        verify_auth_token(authorization)

        emoji = Emoji.get_or_none(Emoji.id == emoji_id)

        if not emoji:
            raise HTTPException(status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包")

        return EmojiDetailResponse(success=True, data=emoji_to_response(emoji))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取表情包详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取表情包详情失败: {str(e)}") from e


@router.patch("/{emoji_id}", response_model=EmojiUpdateResponse)
async def update_emoji(emoji_id: int, request: EmojiUpdateRequest, authorization: Optional[str] = Header(None)):
    """
    增量更新表情包（只更新提供的字段）

    Args:
        emoji_id: 表情包ID
        request: 更新请求（只包含需要更新的字段）
        authorization: Authorization header

    Returns:
        更新结果
    """
    try:
        verify_auth_token(authorization)

        emoji = Emoji.get_or_none(Emoji.id == emoji_id)

        if not emoji:
            raise HTTPException(status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包")

        # 只更新提供的字段
        update_data = request.model_dump(exclude_unset=True)

        if not update_data:
            raise HTTPException(status_code=400, detail="未提供任何需要更新的字段")

        # emotion 字段直接使用字符串,无需转换

        # 如果注册状态从 False 变为 True，记录注册时间
        if "is_registered" in update_data and update_data["is_registered"] and not emoji.is_registered:
            update_data["register_time"] = time.time()

        # 执行更新
        for field, value in update_data.items():
            setattr(emoji, field, value)

        emoji.save()

        logger.info(f"表情包已更新: ID={emoji_id}, 字段: {list(update_data.keys())}")

        return EmojiUpdateResponse(
            success=True, message=f"成功更新 {len(update_data)} 个字段", data=emoji_to_response(emoji)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"更新表情包失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新表情包失败: {str(e)}") from e


@router.delete("/{emoji_id}", response_model=EmojiDeleteResponse)
async def delete_emoji(emoji_id: int, authorization: Optional[str] = Header(None)):
    """
    删除表情包

    Args:
        emoji_id: 表情包ID
        authorization: Authorization header

    Returns:
        删除结果
    """
    try:
        verify_auth_token(authorization)

        emoji = Emoji.get_or_none(Emoji.id == emoji_id)

        if not emoji:
            raise HTTPException(status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包")

        # 记录删除信息
        emoji_hash = emoji.emoji_hash

        # 执行删除
        emoji.delete_instance()

        logger.info(f"表情包已删除: ID={emoji_id}, hash={emoji_hash}")

        return EmojiDeleteResponse(success=True, message=f"成功删除表情包: {emoji_hash}")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"删除表情包失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除表情包失败: {str(e)}") from e


@router.get("/stats/summary")
async def get_emoji_stats(authorization: Optional[str] = Header(None)):
    """
    获取表情包统计数据

    Args:
        authorization: Authorization header

    Returns:
        统计数据
    """
    try:
        verify_auth_token(authorization)

        total = Emoji.select().count()
        registered = Emoji.select().where(Emoji.is_registered).count()
        banned = Emoji.select().where(Emoji.is_banned).count()

        # 按格式统计
        formats = {}
        for emoji in Emoji.select(Emoji.format):
            fmt = emoji.format
            formats[fmt] = formats.get(fmt, 0) + 1

        # 获取最常用的表情包（前10）
        top_used = Emoji.select().order_by(Emoji.usage_count.desc()).limit(10)
        top_used_list = [
            {
                "id": emoji.id,
                "emoji_hash": emoji.emoji_hash,
                "description": emoji.description,
                "usage_count": emoji.usage_count,
            }
            for emoji in top_used
        ]

        return {
            "success": True,
            "data": {
                "total": total,
                "registered": registered,
                "banned": banned,
                "unregistered": total - registered,
                "formats": formats,
                "top_used": top_used_list,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取统计数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计数据失败: {str(e)}") from e


@router.post("/{emoji_id}/register", response_model=EmojiUpdateResponse)
async def register_emoji(emoji_id: int, authorization: Optional[str] = Header(None)):
    """
    注册表情包（快捷操作）

    Args:
        emoji_id: 表情包ID
        authorization: Authorization header

    Returns:
        更新结果
    """
    try:
        verify_auth_token(authorization)

        emoji = Emoji.get_or_none(Emoji.id == emoji_id)

        if not emoji:
            raise HTTPException(status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包")

        if emoji.is_registered:
            raise HTTPException(status_code=400, detail="该表情包已经注册")

        # 注册表情包（如果已封禁，自动解除封禁）
        emoji.is_registered = True
        emoji.is_banned = False  # 注册时自动解除封禁
        emoji.register_time = time.time()
        emoji.save()

        logger.info(f"表情包已注册: ID={emoji_id}")

        return EmojiUpdateResponse(success=True, message="表情包注册成功", data=emoji_to_response(emoji))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"注册表情包失败: {e}")
        raise HTTPException(status_code=500, detail=f"注册表情包失败: {str(e)}") from e


@router.post("/{emoji_id}/ban", response_model=EmojiUpdateResponse)
async def ban_emoji(emoji_id: int, authorization: Optional[str] = Header(None)):
    """
    禁用表情包（快捷操作）

    Args:
        emoji_id: 表情包ID
        authorization: Authorization header

    Returns:
        更新结果
    """
    try:
        verify_auth_token(authorization)

        emoji = Emoji.get_or_none(Emoji.id == emoji_id)

        if not emoji:
            raise HTTPException(status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包")

        # 禁用表情包（同时取消注册）
        emoji.is_banned = True
        emoji.is_registered = False
        emoji.save()

        logger.info(f"表情包已禁用: ID={emoji_id}")

        return EmojiUpdateResponse(success=True, message="表情包禁用成功", data=emoji_to_response(emoji))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"禁用表情包失败: {e}")
        raise HTTPException(status_code=500, detail=f"禁用表情包失败: {str(e)}") from e


@router.get("/{emoji_id}/thumbnail")
async def get_emoji_thumbnail(
    emoji_id: int,
    token: Optional[str] = Query(None, description="访问令牌"),
    authorization: Optional[str] = Header(None),
):
    """
    获取表情包缩略图（优先返回优化后的 WebP 代理图片）

    Args:
        emoji_id: 表情包ID
        token: 访问令牌（通过 query parameter）
        authorization: Authorization header

    Returns:
        表情包图片文件（优先 WebP 代理图片，否则原图）
    """
    try:
        # 优先使用 query parameter 中的 token（用于 img 标签）
        if token:
            token_manager = get_token_manager()
            if not token_manager.verify_token(token):
                raise HTTPException(status_code=401, detail="Token 无效或已过期")
        else:
            # 如果没有 query token，则验证 Authorization header
            verify_auth_token(authorization)

        emoji = Emoji.get_or_none(Emoji.id == emoji_id)

        if not emoji:
            raise HTTPException(status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包")

        # 检查文件是否存在
        if not os.path.exists(emoji.full_path):
            raise HTTPException(status_code=404, detail="表情包文件不存在")

        # 使用代理图片管理器获取优化后的图片路径
        from src.webui.image_proxy import get_proxy_manager

        proxy_manager = get_proxy_manager()
        file_path, media_type = await proxy_manager.get_proxy_or_original(emoji)

        # 确定文件名（用于下载时的文件名）
        if media_type == "image/webp":
            filename = f"{emoji.emoji_hash}.webp"
        else:
            filename = f"{emoji.emoji_hash}.{emoji.format}"

        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
            headers={"Cache-Control": "public, max-age=86400"},  # 浏览器缓存 24 小时
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取表情包缩略图失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取表情包缩略图失败: {str(e)}") from e


@router.post("/batch/delete", response_model=BatchDeleteResponse)
async def batch_delete_emojis(request: BatchDeleteRequest, authorization: Optional[str] = Header(None)):
    """
    批量删除表情包

    Args:
        request: 包含emoji_ids列表的请求
        authorization: Authorization header

    Returns:
        批量删除结果
    """
    try:
        verify_auth_token(authorization)

        if not request.emoji_ids:
            raise HTTPException(status_code=400, detail="未提供要删除的表情包ID")

        deleted_count = 0
        failed_count = 0
        failed_ids = []

        for emoji_id in request.emoji_ids:
            try:
                emoji = Emoji.get_or_none(Emoji.id == emoji_id)
                if emoji:
                    emoji.delete_instance()
                    deleted_count += 1
                    logger.info(f"批量删除表情包: {emoji_id}")
                else:
                    failed_count += 1
                    failed_ids.append(emoji_id)
            except Exception as e:
                logger.error(f"删除表情包 {emoji_id} 失败: {e}")
                failed_count += 1
                failed_ids.append(emoji_id)

        message = f"成功删除 {deleted_count} 个表情包"
        if failed_count > 0:
            message += f"，{failed_count} 个失败"

        return BatchDeleteResponse(
            success=True,
            message=message,
            deleted_count=deleted_count,
            failed_count=failed_count,
            failed_ids=failed_ids,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"批量删除表情包失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量删除失败: {str(e)}") from e


# 表情包存储目录
EMOJI_REGISTERED_DIR = os.path.join("data", "emoji_registed")


class EmojiUploadResponse(BaseModel):
    """表情包上传响应"""

    success: bool
    message: str
    data: Optional[EmojiResponse] = None


@router.post("/upload", response_model=EmojiUploadResponse)
async def upload_emoji(
    file: EmojiFile,
    description: DescriptionForm = "",
    emotion: EmotionForm = "",
    is_registered: IsRegisteredForm = True,
    authorization: Optional[str] = Header(None),
):
    """
    上传并注册表情包

    Args:
        file: 表情包图片文件 (支持 jpg, jpeg, png, gif, webp)
        description: 表情包描述
        emotion: 情感标签，多个用逗号分隔
        is_registered: 是否直接注册，默认为 True
        authorization: Authorization header

    Returns:
        上传结果和表情包信息
    """
    try:
        verify_auth_token(authorization)

        # 验证文件类型
        if not file.content_type:
            raise HTTPException(status_code=400, detail="无法识别文件类型")

        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file.content_type}，支持: {', '.join(allowed_types)}",
            )

        # 读取文件内容
        file_content = await file.read()

        if not file_content:
            raise HTTPException(status_code=400, detail="文件内容为空")

        # 验证图片并获取格式
        try:
            with Image.open(io.BytesIO(file_content)) as img:
                img_format = img.format.lower() if img.format else "png"
                # 验证图片可以正常打开
                img.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的图片文件: {str(e)}") from e

        # 重新打开图片（verify后需要重新打开）
        with Image.open(io.BytesIO(file_content)) as img:
            img_format = img.format.lower() if img.format else "png"

        # 计算文件哈希
        emoji_hash = hashlib.md5(file_content).hexdigest()

        # 检查是否已存在相同哈希的表情包
        existing_emoji = Emoji.get_or_none(Emoji.emoji_hash == emoji_hash)
        if existing_emoji:
            raise HTTPException(
                status_code=409,
                detail=f"已存在相同的表情包 (ID: {existing_emoji.id})",
            )

        # 确保目录存在
        os.makedirs(EMOJI_REGISTERED_DIR, exist_ok=True)

        # 生成文件名
        timestamp = int(time.time())
        filename = f"emoji_{timestamp}_{emoji_hash[:8]}.{img_format}"
        full_path = os.path.join(EMOJI_REGISTERED_DIR, filename)

        # 如果文件已存在，添加随机后缀
        counter = 1
        while os.path.exists(full_path):
            filename = f"emoji_{timestamp}_{emoji_hash[:8]}_{counter}.{img_format}"
            full_path = os.path.join(EMOJI_REGISTERED_DIR, filename)
            counter += 1

        # 保存文件
        with open(full_path, "wb") as f:
            f.write(file_content)

        logger.info(f"表情包文件已保存: {full_path}")

        # 处理情感标签
        emotion_str = ",".join(e.strip() for e in emotion.split(",") if e.strip()) if emotion else ""

        # 创建数据库记录
        current_time = time.time()
        emoji = Emoji.create(
            full_path=full_path,
            format=img_format,
            emoji_hash=emoji_hash,
            description=description,
            emotion=emotion_str,
            query_count=0,
            is_registered=is_registered,
            is_banned=False,
            record_time=current_time,
            register_time=current_time if is_registered else None,
            usage_count=0,
            last_used_time=None,
        )

        logger.info(f"表情包已上传并注册: ID={emoji.id}, hash={emoji_hash}")

        return EmojiUploadResponse(
            success=True,
            message="表情包上传成功" + ("并已注册" if is_registered else ""),
            data=emoji_to_response(emoji),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"上传表情包失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}") from e


@router.post("/batch/upload")
async def batch_upload_emoji(
    files: EmojiFiles,
    emotion: EmotionForm = "",
    is_registered: IsRegisteredForm = True,
    authorization: Optional[str] = Header(None),
):
    """
    批量上传表情包

    Args:
        files: 多个表情包图片文件
        emotion: 共用的情感标签
        is_registered: 是否直接注册
        authorization: Authorization header

    Returns:
        批量上传结果
    """
    try:
        verify_auth_token(authorization)

        results = {
            "success": True,
            "total": len(files),
            "uploaded": 0,
            "failed": 0,
            "details": [],
        }

        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        os.makedirs(EMOJI_REGISTERED_DIR, exist_ok=True)

        for file in files:
            try:
                # 验证文件类型
                if file.content_type not in allowed_types:
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "filename": file.filename,
                            "success": False,
                            "error": f"不支持的文件类型: {file.content_type}",
                        }
                    )
                    continue

                # 读取文件内容
                file_content = await file.read()

                if not file_content:
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "filename": file.filename,
                            "success": False,
                            "error": "文件内容为空",
                        }
                    )
                    continue

                # 验证图片
                try:
                    with Image.open(io.BytesIO(file_content)) as img:
                        img_format = img.format.lower() if img.format else "png"
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "filename": file.filename,
                            "success": False,
                            "error": f"无效的图片: {str(e)}",
                        }
                    )
                    continue

                # 计算哈希
                emoji_hash = hashlib.md5(file_content).hexdigest()

                # 检查重复
                if Emoji.get_or_none(Emoji.emoji_hash == emoji_hash):
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "filename": file.filename,
                            "success": False,
                            "error": "已存在相同的表情包",
                        }
                    )
                    continue

                # 生成文件名并保存
                timestamp = int(time.time())
                filename = f"emoji_{timestamp}_{emoji_hash[:8]}.{img_format}"
                full_path = os.path.join(EMOJI_REGISTERED_DIR, filename)

                counter = 1
                while os.path.exists(full_path):
                    filename = f"emoji_{timestamp}_{emoji_hash[:8]}_{counter}.{img_format}"
                    full_path = os.path.join(EMOJI_REGISTERED_DIR, filename)
                    counter += 1

                with open(full_path, "wb") as f:
                    f.write(file_content)

                # 处理情感标签
                emotion_str = ",".join(e.strip() for e in emotion.split(",") if e.strip()) if emotion else ""

                # 创建数据库记录
                current_time = time.time()
                emoji = Emoji.create(
                    full_path=full_path,
                    format=img_format,
                    emoji_hash=emoji_hash,
                    description="",  # 批量上传暂不设置描述
                    emotion=emotion_str,
                    query_count=0,
                    is_registered=is_registered,
                    is_banned=False,
                    record_time=current_time,
                    register_time=current_time if is_registered else None,
                    usage_count=0,
                    last_used_time=None,
                )

                results["uploaded"] += 1
                results["details"].append(
                    {
                        "filename": file.filename,
                        "success": True,
                        "id": emoji.id,
                    }
                )

            except Exception as e:
                results["failed"] += 1
                results["details"].append(
                    {
                        "filename": file.filename,
                        "success": False,
                        "error": str(e),
                    }
                )

        results["message"] = f"成功上传 {results['uploaded']} 个，失败 {results['failed']} 个"
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"批量上传表情包失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量上传失败: {str(e)}") from e


# ==================== 图片代理缓存管理 API ====================


@router.get("/proxy/stats")
async def get_proxy_cache_stats(authorization: Optional[str] = Header(None)):
    """
    获取图片代理缓存统计信息

    Args:
        authorization: Authorization header

    Returns:
        缓存统计信息
    """
    try:
        verify_auth_token(authorization)

        from src.webui.image_proxy import get_proxy_manager
        from src.webui.image_proxy.proxy_manager import PROXY_DIR

        proxy_manager = get_proxy_manager()

        # 统计缓存文件
        webp_count = 0
        skip_count = 0
        total_cache_size = 0

        if os.path.exists(PROXY_DIR):
            for filename in os.listdir(PROXY_DIR):
                file_path = os.path.join(PROXY_DIR, filename)
                if filename.endswith(".webp"):
                    webp_count += 1
                    total_cache_size += os.path.getsize(file_path)
                elif filename.endswith(".skip"):
                    skip_count += 1

        # 统计表情包总数
        total_emojis = Emoji.select().count()

        # 计算缓存覆盖率
        cached_count = webp_count + skip_count
        coverage = (cached_count / total_emojis * 100) if total_emojis > 0 else 0

        return {
            "success": True,
            "data": {
                "total_emojis": total_emojis,
                "webp_cached": webp_count,
                "skipped_original_better": skip_count,
                "pending_transcode": total_emojis - cached_count,
                "cache_coverage_percent": round(coverage, 2),
                "cache_size_bytes": total_cache_size,
                "cache_size_mb": round(total_cache_size / 1024 / 1024, 2),
                "cache_directory": PROXY_DIR,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取代理缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取代理缓存统计失败: {str(e)}") from e


@router.post("/proxy/transcode-all")
async def trigger_transcode_all(authorization: Optional[str] = Header(None)):
    """
    触发批量转码所有未缓存的表情包（异步执行）

    Args:
        authorization: Authorization header

    Returns:
        任务启动结果
    """
    try:
        verify_auth_token(authorization)

        from src.webui.image_proxy import get_proxy_manager

        proxy_manager = get_proxy_manager()

        # 获取所有表情包并加入转码队列
        emojis = Emoji.select()
        enqueued_count = 0

        for emoji in emojis:
            if await proxy_manager.enqueue_transcode(emoji):
                enqueued_count += 1

        return {
            "success": True,
            "message": f"已将 {enqueued_count} 个表情包加入转码队列",
            "enqueued_count": enqueued_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"触发批量转码失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发批量转码失败: {str(e)}") from e


@router.post("/proxy/cleanup")
async def trigger_cache_cleanup(authorization: Optional[str] = Header(None)):
    """
    触发缓存清理（删除无效的缓存文件）

    Args:
        authorization: Authorization header

    Returns:
        清理结果
    """
    try:
        verify_auth_token(authorization)

        from src.webui.image_proxy import get_proxy_manager

        proxy_manager = get_proxy_manager()
        cleaned_count = await proxy_manager.cleanup_stale_cache()

        return {
            "success": True,
            "message": f"已清理 {cleaned_count} 个无效缓存文件",
            "cleaned_count": cleaned_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"触发缓存清理失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发缓存清理失败: {str(e)}") from e
