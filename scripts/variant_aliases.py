"""Canoniza nomes de variante para os geradores de tabelas e figuras.

Em 2026-10-06 o seletor de landmarks do corpo principal passou de ``colnorm``
para ``random`` (ver docs/PLANO_FECHAMENTO.md, "PLANO DE MIGRAÇÃO"), e com ele
quatro nomes de variante mudaram nos JSONs canônicos. O rótulo de exibição na
tese não muda -- o modelo é o mesmo, só o seletor --, então os geradores
continuam indexando pelas chaves legadas e este módulo traduz na leitura.
O seletor verdadeiro de cada registro continua no JSON (campo ``variant``).

NÃO usar em generate_nystrom_selection_table.py / generate_appendix_selection_tables.py
(Apêndice C), onde o seletor é a variável de interesse, nem em scripts que
retreinam a partir de ``variant`` (plot_decision_surfaces.py).
"""

LEGACY_KEY = {
    "NystromLSSVMRandom": "NystromLSSVMColnorm",
    "ADMMNystromRandom": "ADMMNystromLSSVM",
    "FISTANystromRandom": "FISTANystrom",
    "FTTransformerCURRandom": "FTTransformerCURColnorm",
}


def canon(records):
    """Devolve os registros com ``variant`` traduzido para a chave legada."""
    out = []
    for r in records:
        v = r.get("variant")
        out.append({**r, "variant": LEGACY_KEY[v]} if v in LEGACY_KEY else r)
    return out
