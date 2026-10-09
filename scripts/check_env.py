"""Muhit to'g'ri sozlanganini tekshiradi. Ishga tushirish: python scripts/check_env.py"""
import importlib
import sys

MIN_PYTHON = (3, 10)
PACKAGES = ["fastapi", "uvicorn", "sqlalchemy", "pydantic", "pandas", "numpy", "dotenv"]


def main() -> int:
    ok = True
    v = sys.version_info
    print(f"Python versiyasi: {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) < MIN_PYTHON:
        print(f"  XATO: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} yoki yangiroq kerak")
        ok = False

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"Virtual muhit faol: {'ha' if in_venv else 'YOQ (venv ni yoqing)'}")
    ok = ok and in_venv

    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  [OK] {name} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"  [YO'Q] {name}  ->  pip install -r requirements.txt")
            ok = False

    print("\nNatija:", "hammasi tayyor" if ok else "muammolar bor, yuqoridagilarni tuzating")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
