#!/usr/bin/env python3
"""Normaliza o separador decimal das tabelas para vírgula (ABNT/CONMETRO).

A Resolução CONMETRO nº 12/1988, que adota o SI no Brasil, determina que a
parte inteira e a parte decimal de um número sejam separadas por **vírgula**.
A NBR 14724 delega a apresentação de tabelas às Normas de Apresentação Tabular
do IBGE (3ª ed., 1993), e o próprio modelo do IFCE grafa seus exemplos com
vírgula. Os geradores deste repositório, porém, emitem o ponto do Python — daí
esta passada única de normalização, executada sobre os `.tex` já gerados.

Substitui-se por ``{,}`` e não por ``,`` porque, em modo matemático, a vírgula
solta é tratada como pontuação e recebe espaço à direita (``$0,992$`` sai como
"0, 992"); as chaves suprimem esse espaçamento e são inócuas em modo texto,
de modo que a mesma forma serve aos dois contextos.

A operação é idempotente: ``0{,}992`` não contém mais ponto entre dígitos.

Uso:
    python scripts/normalize_decimals.py            # dry-run (padrão)
    python scripts/normalize_decimals.py --apply    # grava
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABELAS = ROOT.parent / "dissertacao-latex" / "tables"

# Ponto ladeado por dígitos — o separador decimal a converter.
DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")

# Trechos que nunca devem ser tocados: argumentos de comandos que carregam
# identificadores (onde um ponto pode ser parte do nome) e comentários.
PROTEGIDO = re.compile(
    r"(\\(?:label|ref|autoref|eqref|cite\w*|input|include|includegraphics)\s*\{[^}]*\})"
    r"|(?<!\\)%.*$",
    re.MULTILINE)

# Token numérico com mais de um ponto (ex.: "4.3.1") — provavelmente não é
# um decimal, e sim numeração ou versão. É preservado intacto e reportado,
# para decisão manual.
SUSPEITO = re.compile(r"\d+\.\d+\.[\d.]*\d")

# A varredura respeita, na mesma passada, os comandos com identificadores, os
# comentários e os tokens ambíguos.
INTOCAVEL = re.compile(f"{PROTEGIDO.pattern}|({SUSPEITO.pattern})", re.MULTILINE)


def normaliza(texto: str) -> tuple[str, int, list[str]]:
    """Devolve (texto convertido, nº de substituições, tokens suspeitos)."""
    suspeitos = SUSPEITO.findall(texto)
    partes, pos, n = [], 0, 0

    for m in INTOCAVEL.finditer(texto):
        trecho = texto[pos:m.start()]
        convertido, k = DECIMAL.subn("{,}", trecho)
        partes += [convertido, m.group(0)]   # o trecho protegido passa intacto
        n += k
        pos = m.end()

    convertido, k = DECIMAL.subn("{,}", texto[pos:])
    partes.append(convertido)
    return "".join(partes), n + k, suspeitos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="grava as alterações (sem esta opção, apenas simula)")
    ap.add_argument("--dir", type=Path, default=TABELAS)
    a = ap.parse_args()

    arquivos = sorted(a.dir.glob("*.tex"))
    if not arquivos:
        print(f"nenhum .tex em {a.dir}")
        return 1

    total, tocados, alertas = 0, 0, []
    for f in arquivos:
        original = f.read_text(encoding="utf-8")
        novo, n, suspeitos = normaliza(original)
        if suspeitos:
            alertas.append((f.name, suspeitos))
        if n:
            tocados += 1
            total += n
            print(f"  {f.name:42s} {n:4d} substituições")
            if a.apply:
                f.write_text(novo, encoding="utf-8")

    modo = "APLICADO" if a.apply else "DRY-RUN (nada gravado)"
    print(f"\n[{modo}] {total} substituições em {tocados}/{len(arquivos)} arquivos")

    if alertas:
        print("\n!! tokens com mais de um ponto (NÃO convertidos, conferir):")
        for nome, toks in alertas:
            print(f"   {nome}: {sorted(set(toks))}")
    else:
        print("\nNenhum token ambíguo (numeração/versão) encontrado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
