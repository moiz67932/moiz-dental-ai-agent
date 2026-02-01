import sys
import os
sys.path.append(os.getcwd())

try:
    from services import scheduling_service
    from tools import assistant_tools
    print('Imports successful')
except Exception as e:
    print(f'Import failed: {e}')
    import traceback
    traceback.print_exc()
