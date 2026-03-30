import sys
from pathlib import Path

# 将项目根目录和 src 目录添加到 Python 路径
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
