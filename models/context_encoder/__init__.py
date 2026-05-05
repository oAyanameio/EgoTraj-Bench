from .tbd_encoder_score import ContextEncoderScore

# 定义模块的公开接口：当使用 `from module import *` 时，只导出 ContextEncoderScore
__all__ = {
    "ContextEncoderScore": ContextEncoderScore,
}

def build_context_encoder(config, use_pre_norm):
    """工厂函数：根据配置构建上下文编码器模型

    Args:
        config: 模型配置对象，包含编码器的相关参数（如隐层维度、层数等）
        use_pre_norm: 是否在编码器输入层使用预归一化

    Returns:
        ContextEncoderScore: 构建好的上下文编码器模型实例
    """
    model = __all__[config.NAME](
        config=config, use_pre_norm=use_pre_norm
    )

    return model
