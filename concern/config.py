import importlib
from collections import OrderedDict

import anyconfig
import munch


class Config(object):
    def __init__(self):
        pass

    def load(self, conf):
        conf = anyconfig.load(conf) # 加载配置文件*.yaml
        return munch.munchify(conf) # gengerate munch objects 转换为munch格式

    def compile(self, conf, return_packages=False):
        packages = conf.get('package', [])  # 获取'package'相关dict  key value
        defines = {}

        for path in conf.get('import', []): # 获取'import'相关 key value
            parent_conf = self.load(path)
            #   递归找到下面的参数
            parent_packages, parent_defines = self.compile(
                parent_conf, return_packages=True)
            packages.extend(parent_packages) # 字典key为 list，添加新元素
            defines.update(parent_defines)   # 更新参数字典

        modules = []
        for package in packages:
            # 动态加载python模块
            module = importlib.import_module(package)
            modules.append(module)

        if isinstance(conf['define'], dict):
            conf['define'] = [conf['define']]
        for define in conf['define']:
            # 返回conf['define'] key value中'name'字典
            name = define.copy().pop('name')
            # name必须为str类型
            if not isinstance(name, str):
                raise RuntimeError('name must be str')
            # 返回字符串中除前缀的内容
            defines[name] = self.compile_conf(define, defines, modules)

        if return_packages:
            return packages, defines
        else:
            return defines
    # 用来将参数返回
    def compile_conf(self, conf, defines, modules):
        if isinstance(conf, (int, float)):
            return conf
        elif isinstance(conf, str):
            if conf.startswith('^'):
                return defines[conf[1:]]
            if conf.startswith('$'):
                return {'class': self.find_class_in_modules(conf[1:], modules)}
            return conf
        #   大多dict形式
        elif isinstance(conf, dict):
            if 'class' in conf:
                conf['class'] = self.find_class_in_modules(
                    conf['class'], modules)
            if 'base' in conf:
                base = conf.copy().pop('base')

                if not isinstance(base, str):
                    raise RuntimeError('base must be str')

                conf = {
                    **defines[base],
                    **conf,
                }
            return {key: self.compile_conf(value, defines, modules) for key, value in conf.items()}
        elif isinstance(conf, (list, tuple)):
            return [self.compile_conf(value, defines, modules) for value in conf]
        else:
            return conf

    def find_class_in_modules(self, cls, modules):
        if not isinstance(cls, str):
            raise RuntimeError('class name must be str')

        if cls.find('.') != -1:
            package, cls = cls.rsplit('.', 1)
            module = importlib.import_module(package)
            if hasattr(module, cls):
                return module.__name__ + '.' + cls

        for module in modules:
            if hasattr(module, cls):
                return module.__name__ + '.' + cls
        raise RuntimeError('class not found ' + cls)


class State:
    def __init__(self, autoload=True, default=None):
        self.autoload = autoload
        self.default = default
        
# 创建class并赋予它属性和对应值
class StateMeta(type):
    def __new__(mcs, name, bases, attrs):
        current_states = []
        for key, value in attrs.items():
            if isinstance(value, State):
                current_states.append((key, value))

        current_states.sort(key=lambda x: x[0])
        attrs['states'] = OrderedDict(current_states)
        new_class = super(StateMeta, mcs).__new__(mcs, name, bases, attrs)

        # Walk through the MRO
        states = OrderedDict()
        # 查看类的多继承关系，更新到字典中
        for base in reversed(new_class.__mro__):
            if hasattr(base, 'states'):
                states.update(base.states)
        new_class.states = states

        for key, value in states.items():
            setattr(new_class, key, value.default)

        return new_class


class Configurable(metaclass=StateMeta):
    def __init__(self, *args, cmd={}, **kwargs):
        self.load_all(cmd=cmd, **kwargs)

    @staticmethod
    def construct_class_from_config(args):
        cls = Configurable.extract_class_from_args(args)
        return cls(**args)
    # 静态函数，获取对象的属性值
    @staticmethod
    def extract_class_from_args(args):
        cls = args.copy().pop('class') # 只返回key
        package, cls = cls.rsplit('.', 1) # 从最右边最多进行一次拆分
        module = importlib.import_module(package) # 动态导入模块，主要导入checkpoint/learining_rate...
        cls = getattr(module, cls) # 获取modeule中cls的属性
        return cls
    
    # 获取所有的属性值
    def load_all(self, **kwargs):
        for name, state in self.states.items():
            if state.autoload:
                self.load(name, **kwargs)

    # 获取cmd字典中的value，并将其设定为state_name中新的属性
    def load(self, state_name, **kwargs):
        # FIXME: kwargs should be filtered
        # Args passed from command line
        # delete the dict and return the value
        cmd = kwargs.pop('cmd', dict())
        # 设置对象属性 setattr(object, name, value)
        if state_name in kwargs:
            setattr(self, state_name, self.create_member_from_config(
                (kwargs[state_name], cmd)))
        else:
            setattr(self, state_name, self.states[state_name].default)

    def create_member_from_config(self, conf):
        args, cmd = conf
        if args is None or isinstance(args, (int, float, str)):
            return args
        elif isinstance(args, (list, tuple)):
            return [self.create_member_from_config((subargs, cmd)) for subargs in args]
        # 根据train.py文件中的设置，args为dict
        elif isinstance(args, dict):
            if 'class' in args:
                cls = self.extract_class_from_args(args)
                return cls(**args, cmd=cmd)
            return {key: self.create_member_from_config((subargs, cmd)) for key, subargs in args.items()}
        else:
            return args
    # 存入属性
    def dump(self):
        state = {}
        state['class'] = self.__class__.__module__ + \
            '.' + self.__class__.__name__
        for name, value in self.states.items():
            obj = getattr(self, name)
            state[name] = self.dump_obj(obj)
        return state

    def dump_obj(self, obj):
        if obj is None:
            return None
        elif hasattr(obj, 'dump'):
            return obj.dump()
        elif isinstance(obj, (int, float, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self.dump_obj(value) for value in obj]
        elif isinstance(obj, dict):
            return {key: self.dump_obj(value) for key, value in obj.items()}
        else:
            return str(obj)

