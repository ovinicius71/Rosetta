"""Avaliação opcional de LaTeX via SymPy: LaTeX → resultado.

Usado por POST /evaluate. Melhor esforço: nem toda expressão é avaliável (integrais
indefinidas retornam forma simbólica; expressões inválidas retornam erro amigável, nunca
HTTP 500).

Requer `antlr4-python3-runtime==4.11` (backend do sympy.parsing.latex).
"""

from __future__ import annotations

from .schemas import EvaluateResponse


def evaluate_latex(latex: str) -> EvaluateResponse:
    """Converte LaTeX → expressão SymPy e tenta simplificar/calcular.

    Regras:
      - sem símbolos livres (ex.: '1+1', '\\frac{3}{4}') → valor exato; se não-inteiro,
        inclui aproximação decimal ('3/4 = 0.75').
      - com símbolos livres (ex.: 'x^2 + x') → forma simplificada simbólica.
      - igualdades ('x + 1 = 3') → resolve para o símbolo, se possível.
    """
    latex = (latex or "").strip()
    if not latex:
        return EvaluateResponse(error="expressão vazia")
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        expr = parse_latex(latex)

        if isinstance(expr, sympy.Equality):
            free = sorted(expr.free_symbols, key=lambda s: s.name)
            if free:
                sols = sympy.solve(expr, free[0])
                if sols:
                    txt = ", ".join(f"{free[0]} = {sympy.nsimplify(s)}" for s in sols)
                    return EvaluateResponse(result=txt)
            # sem símbolos (ex.: '1+1=2') → verdade/falsidade
            return EvaluateResponse(result=str(sympy.simplify(expr)))

        simplified = sympy.simplify(expr)
        if not simplified.free_symbols:
            exact = sympy.nsimplify(simplified)
            if exact.is_Integer:
                return EvaluateResponse(result=str(exact))
            approx = sympy.N(simplified, 8)
            return EvaluateResponse(result=f"{exact} = {approx}")
        return EvaluateResponse(result=str(simplified))
    except Exception as e:  # noqa: BLE001 - resposta amigável, nunca 500
        return EvaluateResponse(error=f"não consegui avaliar: {e}")
