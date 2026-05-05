"""配置管理模块

本模块负责加载、管理和更新配置文件，提供配置对象的创建和操作功能。
"""

import yaml
import os
import glob
import numpy as np
from easydict import EasyDict
from .utils import create_logger


def noise_module_inherit_cfg(yml_dict):
    """处理噪声模块的配置继承
    
    如果模型使用 social_context_regen 模式，将 past_frames 和 future_frames 参数
    继承到模型配置中。
    
    Args:
        yml_dict: 配置字典对象
        
    Returns:
        更新后的配置字典对象
    """
    
    def inherit_cfg(model_cfg, cfg_dict):
        """将配置字典中的键值对继承到模型配置中
        
        Args:
            model_cfg: 模型配置对象
            cfg_dict: 要继承的配置字典
            
        Returns:
            更新后的模型配置对象
        """
        for key, value in cfg_dict.items():
            if key not in model_cfg:
                model_cfg[key] = value
        return model_cfg

    if yml_dict.MODEL.get("PAST_TRAJ_MODE", None) == "social_context_regen":
        past_frames = yml_dict.past_frames
        future_frames = yml_dict.future_frames

        cfg_dict = {
            "past_frames": past_frames,
            "future_frames": future_frames,
        }
        yml_dict.MODEL = inherit_cfg(yml_dict.MODEL, cfg_dict)

    return yml_dict


class Config:
    """配置类，用于管理模型训练和评估的配置
    
    该类负责加载YAML配置文件，创建必要的目录结构，管理配置参数，
    并提供访问和修改配置的方法。
    """
    
    def __init__(self, cfg_path, tag, train_mode=True):
        """初始化配置对象
        
        Args:
            cfg_path: 配置文件路径
            tag: 实验标签
            train_mode: 是否为训练模式
        """
        self.cfg_path = cfg_path
        # 从配置文件路径中提取配置名称
        self.cfg_name = (
            os.path.basename(cfg_path).replace(".yaml", "").replace(".yml", "")
        )
        self.tag = tag
        self.train_mode = train_mode
        # 查找配置文件
        files = glob.glob(cfg_path, recursive=True)
        # 确保配置文件存在
        assert len(files) == 1, "YAML file [{}] does not exist!".format(cfg_path)
        # 加载配置文件
        yml_dict_ = EasyDict(yaml.safe_load(open(files[0], "r")))
        
        if train_mode:
            self.yml_dict = yml_dict_
            # 扩展结果根目录路径
            self.results_root_dir = os.path.expanduser(
                self.yml_dict["results_root_dir"]
            )
        else:
            # 非训练模式下更新配置
            yml_dict_.cfg_path = cfg_path  # renew the config with the non-train mode
            yml_dict_.cfg_name = self.cfg_name
            yml_dict_.tag = tag
            yml_dict_.train_mode = train_mode
            self.yml_dict = yml_dict_
        
        # 计算项目根目录
        self.ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 处理噪声模块的配置继承
        self.yml_dict = noise_module_inherit_cfg(self.yml_dict)

    def create_dirs(self, tag_suffix=None):
        """创建必要的目录结构
        
        Args:
            tag_suffix: 标签后缀
            
        Returns:
            创建的日志记录器
        """
        # 确定标签
        tag = self.tag if tag_suffix is None else self.tag + tag_suffix
        
        if self.train_mode:
            # 训练模式下创建配置目录
            self.cfg_dir = "%s/%s/%s" % (self.results_root_dir, self.cfg_name, tag)
        else:
            # 非训练模式下从模型目录提取配置目录
            self.cfg_dir = os.path.dirname(self.cfg_path)

        # 创建模型、日志和样本目录
        self.model_dir = "%s/models" % self.cfg_dir
        self.log_dir = "%s/log" % self.cfg_dir
        self.sample_dir = "%s/samples" % self.cfg_dir
        # 模型文件路径模板
        self.model_path = os.path.join(self.model_dir, "model_%04d.p")

        # 创建目录（如果不存在）
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        # 获取所有模型文件
        self.model_files = glob.glob(os.path.join(self.model_dir, "model_*.p"))

        # 确定日志文件路径
        if self.train_mode:
            log_file = os.path.join(self.log_dir, "log.txt")
        else:
            log_file = os.path.join(
                self.log_dir, "log_eval_{:s}.txt".format(tag).replace("__", "_")
            )

        # 创建日志记录器
        logger = create_logger(log_file)
        self.logger = logger

        # 更新配置字典
        for key in sorted(dir(self)):
            if not key.startswith("__") and not callable(getattr(self, key)):
                if key in ["yml_dict", "logger"]:
                    continue
                if key not in self.yml_dict:
                    # 记录新添加的键
                    logger.info("New key {} ---> {}".format(key, getattr(self, key)))
                    self.yml_dict[key] = getattr(self, key)
                else:
                    # 记录已存在键的变更
                    orig_val = self.yml_dict[key]
                    new_val = getattr(self, key)
                    if orig_val != new_val:
                        logger.info(
                            "Existing key {} ---> {} from {}".format(
                                key, new_val, orig_val
                            )
                        )
                        self.yml_dict[key] = new_val

        if self.train_mode:
            # 保存更新后的配置文件
            os.system(
                "cp %s %s" % (self.cfg_path, self.cfg_dir)
            )  # 复制原始配置
            self.save_updated_yml()

        return logger

    def save_updated_yml(self):
        """Dump the current yml_dict to {cfg_name}_updated.yml in cfg_dir."""

        def easydict_to_dict(easydict_obj):
            """将EasyDict对象转换为普通字典
            
            Args:
                easydict_obj: EasyDict对象
                
            Returns:
                转换后的普通字典
            """
            result = {}
            for key, value in easydict_obj.items():
                if isinstance(value, EasyDict):
                    result[key] = easydict_to_dict(value)
                else:
                    result[key] = value
            return result

        # 转换EasyDict为普通字典
        nested_dict = easydict_to_dict(self.yml_dict)
        # 保存更新后的配置
        with open(
            os.path.join(self.cfg_dir, "{:s}_updated.yml".format(self.cfg_name)),
            "w",
        ) as f:
            yaml.dump(nested_dict, f)

    def get_last_epoch(self):
        """获取最后一个训练 epoch
        
        Returns:
            最后一个训练 epoch 的编号，如果没有模型文件则返回 None
        """
        model_files = glob.glob(os.path.join(self.model_dir, "model_*.p"))
        if len(model_files) == 0:
            return None
        else:
            # 获取最后一个模型文件
            model_file = os.path.basename(model_files[-1])
            # 从文件名中提取 epoch 编号
            epoch = int(os.path.splitext(model_file)[0].split("model_")[-1])
            return epoch

    def get_latest_ckpt(self):
        """获取最新的模型检查点路径
        
        Returns:
            最新模型检查点的路径，如果没有模型文件则返回 None
        """
        model_files = glob.glob(os.path.join(self.model_dir, "model_*.p"))
        if len(model_files) == 0:
            return None
        else:
            # 提取所有模型文件的 epoch 编号
            epochs = np.array(
                [int(os.path.splitext(f)[0].split("model_")[-1]) for f in model_files]
            )
            # 找到最大的 epoch 编号
            last_epoch = epochs.max()
            # 构建最新模型的路径
            fp = os.path.join(self.model_dir, "model_%04d.p" % last_epoch)
            return fp

    def __getattribute__(self, name):
        """获取属性值，优先从配置字典中获取
        
        Args:
            name: 属性名称
            
        Returns:
            属性值
        """
        try:
            yml_dict = super().__getattribute__("yml_dict")
        except AttributeError:
            # 如果 yml_dict 未设置，返回默认属性
            return super().__getattribute__(
                name
            )  # Return default attribute if yml_dict is not set
        if name in yml_dict:
            return yml_dict[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        """设置属性值，优先设置到配置字典中
        
        Args:
            name: 属性名称
            value: 属性值
        """
        try:
            yml_dict = super().__getattribute__("yml_dict")
        except AttributeError:
            return super().__setattr__(name, value)
        if name in yml_dict:
            yml_dict[name] = value
        else:
            return super().__setattr__(name, value)

    def get(self, name, default=None):
        """获取属性值，如果不存在则返回默认值
        
        Args:
            name: 属性名称
            default: 默认值
            
        Returns:
            属性值或默认值
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return default
