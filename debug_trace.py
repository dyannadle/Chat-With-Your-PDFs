import sys
import traceback

try:
    from chains.rag_chain import get_rag_chain
    print("SUCCESS")
except Exception as e:
    traceback.print_exc()
