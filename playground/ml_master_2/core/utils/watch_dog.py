"""看门狗模块：用于超时控制和强制中断"""
import ctypes
import threading

RUN_TIMEOUT_SECONDS = 4 * 60


# 必须继承自 BaseException，防止被底层的 except Exception: 吞噬
class GlobalTimeoutInterrupt(BaseException):
    """用于看门狗强制打断的全局超时异常"""
    pass


def _async_raise(target_tid, exception_type):
    """通过 C-API 向指定线程强制抛出异常"""
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(target_tid),
        ctypes.py_object(exception_type)
    )
    if ret == 0:
        raise ValueError("无效的线程 ID")
    elif ret > 1:
        # 如果返回值大于 1，说明状态异常，需要撤销操作
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(target_tid), None)
        raise SystemError("PyThreadState_SetAsyncExc 调用失败")


class TimeoutWatchdog:
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.cancel_event = threading.Event()
        self.main_thread_id = threading.get_ident()  # 记录启动看门狗的主线程 ID
        self._thread = None

    def start(self):
        """启动看门狗"""
        self._thread = threading.Thread(target=self._watch, daemon=True, name="TimeoutWatchdog")
        self._thread.start()

    def _watch(self):
        # 等待指定的超时时间，或者直到 stop() 被调用触发 event
        is_cancelled = self.cancel_event.wait(self.timeout_seconds)
        if not is_cancelled:
            # 时间到了，且没有被正常取消 -> 触发主线程中断！
            _async_raise(self.main_thread_id, GlobalTimeoutInterrupt)

    def stop(self):
        """主逻辑正常结束时，调用此方法取消看门狗"""
        self.cancel_event.set()
