import typing
import typing_extensions

# lerobot 0.6.0はPython>=3.12を要求しているが、実際に使っている新機能は
# typingモジュールの一部の名前(Self/Unpack等、3.11以降で追加)のみで、
# 純粋な構文(PEP695ジェネリクス)以外はtyping_extensionsのバックポートで代替できる。
# 本番の採点環境がPython 3.10.12固定(公式README確定)なため、この形で吸収する。
for _name in ("Self", "Unpack", "override", "TypeVarTuple", "ParamSpec", "Required", "NotRequired"):
    if not hasattr(typing, _name) and hasattr(typing_extensions, _name):
        setattr(typing, _name, getattr(typing_extensions, _name))
