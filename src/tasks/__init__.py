from contextlib import contextmanager
from functools import wraps

from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric, SumMetric

from src.tasks._base import TASK_REGISTRY
from src.tasks.cmlm import *


@contextmanager
def on_prediction_mode(pl_module: LightningModule, enable=True):
    if not enable:
        yield
        return

    # [upstream] PL>=2.0 把预测/测试钩子整体换了一轮名字，并去掉了 epoch_end 的入参：
    #     PL<2.0  : test_step / test_step_end / test_epoch_end   （test_epoch_end 收到 outputs）
    #     PL>=2.0 : test_step / on_test_epoch_end                （on_*_epoch_end 不收入参）
    # 哪些名字存在由当前 PL 决定，用「基类上有没有」筛。不能用 is not None：
    # PL 的 configuration_validator 用 callable(getattr(model, 'test_epoch_end', None)) 判定
    # 「是否还在用旧钩子」，而 TaskLitModule.predict_epoch_end 是个可调用的 raise 桩，
    # 一旦被 swap 到 test_epoch_end 上，校验器就会直接抛
    # NotImplementedError: Support for `test_epoch_end` has been removed in v2.0.0。
    _methods = []
    for _method in ['{}_step', '{}_step_end', '{}_epoch_end']:
        _test_method, _predict_method = _method.format('test'), _method.format('predict')
        if hasattr(LightningModule, _test_method) and hasattr(LightningModule, _predict_method):
            _methods.append((_test_method, _predict_method))

    # PL>=2.0 还改了 epoch_end 钩子的入参：PL 调 on_*_epoch_end() 时什么都不传，而
    # CMLM.on_predict_epoch_end(results) 需要「每个 batch 的 predict_step 返回值」才能落
    # test_tau*.fasta、算 scTM；PL 2.x 的 test 循环本身又不保留这些返回值（它的
    # EvaluationLoop 只留 _logged_outputs 这种指标字典）。所以这里自己攒：交换过去的
    # test_step（实际是 predict_step）每批返回值收进 _results，epoch 结束时补成
    # results 传给它。
    _collect = (not hasattr(LightningModule, 'test_epoch_end')
                and hasattr(LightningModule, 'on_test_epoch_end')
                and hasattr(LightningModule, 'on_predict_epoch_end'))
    _results = []
    _originals = []  # [(属性名, 交换前的对象)]，退出时逐个还原

    def _swap(_test_method, _predict_method):
        _test_method_obj = getattr(pl_module, _test_method, None)
        _predict_method_obj = getattr(pl_module, _predict_method, None)
        _originals.append((_test_method, _test_method_obj))
        _originals.append((_predict_method, _predict_method_obj))
        # swap test and predict method/hook
        setattr(pl_module, _test_method, _predict_method_obj)
        setattr(pl_module, _predict_method, _test_method_obj)

    def _swap_step_collecting():
        # test_step <-> predict_step，顺带把 predict_step 的返回值收集起来
        _test_step_obj = getattr(pl_module, 'test_step', None)
        _predict_step_obj = getattr(pl_module, 'predict_step', None)
        _originals.append(('test_step', _test_step_obj))
        _originals.append(('predict_step', _predict_step_obj))

        @wraps(_predict_step_obj)
        def _collecting_test_step(*args, **kwargs):
            _out = _predict_step_obj(*args, **kwargs)
            _results.append(_out)
            return _out

        setattr(pl_module, 'test_step', _collecting_test_step)
        setattr(pl_module, 'predict_step', _test_step_obj)

    def _swap_epoch_end_injecting_results():
        # on_test_epoch_end <-> on_predict_epoch_end，并补上 PL 2.x 不再传的 results
        _test_end_obj = getattr(pl_module, 'on_test_epoch_end', None)
        _predict_end_obj = getattr(pl_module, 'on_predict_epoch_end', None)
        _originals.append(('on_test_epoch_end', _test_end_obj))
        _originals.append(('on_predict_epoch_end', _predict_end_obj))

        @wraps(_predict_end_obj)
        def _test_epoch_end(*args, **kwargs):
            if not args and 'results' not in kwargs:
                kwargs['results'] = _results
            return _predict_end_obj(*args, **kwargs)

        setattr(pl_module, 'on_test_epoch_end', _test_epoch_end)
        setattr(pl_module, 'on_predict_epoch_end', _test_end_obj)

    for _test_method, _predict_method in _methods:
        if _test_method == 'test_step':
            _swap_step_collecting()
        else:
            _swap(_test_method, _predict_method)
    if _collect:
        _swap_epoch_end_injecting_results()

    try:
        yield
    finally:
        for _name, _obj in _originals:
            setattr(pl_module, _name, _obj)


class AutoMetric(nn.Module):
    _type_shortnames = dict(mean=MeanMetric, sum=SumMetric, max=MaxMetric, min=MinMetric, )

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter('_device', torch.zeros(1))

    @property
    def device(self):
        return self._device.device

    def update(self, name, value, type='mean', **kwds):
        if not hasattr(self, name):
            if isinstance(type, str):
                type = self._type_shortnames[type]
            setattr(self, name, type(**kwds))
            getattr(self, name).to(self.device)
        getattr(self, name).update(value)

    def compute(self, name):
        return getattr(self, name).compute()

    def reset(self, name):
        getattr(self, name).reset()

