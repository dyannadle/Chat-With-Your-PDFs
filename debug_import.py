import traceback
try:
    from chains.rag_chain import get_rag_chain
    with open('error2.txt', 'w', encoding='utf-8') as f:
        f.write('SUCCESS')
except Exception as e:
    with open('error2.txt', 'w', encoding='utf-8') as f:
        f.write(traceback.format_exc())
