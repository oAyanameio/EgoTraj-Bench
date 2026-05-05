"""公共工具函数模块

本模块提供了两个基础的工具函数，用于处理值的存在性检查和默认值设置。
"""


def exists(x):
    """检查值是否存在（不为None）
    
    Args:
        x: 要检查的值
        
    Returns:
        bool: 如果x不为None则返回True，否则返回False
    """
    return x is not None


def default(val, d):
    """获取默认值
    
    如果val存在（不为None），则返回val；否则返回d，如果d是可调用对象则调用它
    
    Args:
        val: 要检查的值
        d: 默认值或可调用对象
        
    Returns:
        存在的val值或默认值d
    """
    if exists(val):
        return val
    return d() if callable(d) else d
