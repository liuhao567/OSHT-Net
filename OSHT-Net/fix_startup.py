"""
修复 Python 3.10.19 (torch 2.3.1) 的 regex 编译 bug。
在所有脚本开头加 `import fix_startup` 即可。
"""
import sys
import re as _re_module

_original_compile = _re_module.compile

def _safe_compile(pattern, flags=0):
    try:
        return _original_compile(pattern, flags)
    except ValueError:
        # torch._dynamo 的巨量 SKIP_DIRS_RE 会触发 Python 3.10.19 的 regex bug
        # 降级为匹配空字符串的正则
        return _original_compile(r'(?!)', flags)

_re_module.compile = _safe_compile
