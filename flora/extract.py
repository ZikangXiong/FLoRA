from dataclasses import dataclass

from sympy import And, Implies, Or, Symbol, simplify_logic


@dataclass(frozen=True)
class TemporalAtom:
    """A simplified temporal expression over a single predicate.

    temporal_ops is outermost-first: ["G", "F"] means G(F(...)).
    """

    predicate: str
    negated: bool
    temporal_ops: list[str]

    def __str__(self) -> str:
        inner = f"~{self.predicate}" if self.negated else self.predicate
        # temporal_ops is outermost-first, so we wrap from innermost (last) to outermost (first)
        for op in reversed(self.temporal_ops):
            inner = f"{op}({inner})"
        return inner

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemporalAtom):
            return NotImplemented
        return (
            self.predicate == other.predicate
            and self.negated == other.negated
            and self.temporal_ops == other.temporal_ops
        )

    def __hash__(self) -> int:
        return hash((self.predicate, self.negated, tuple(self.temporal_ops)))


def _simplify_tokens(tokens: list[str]) -> list[str]:
    """Apply rewrite rules to a token list until convergence.

    Token list is innermost-first: ["P0", "G", "NEG", "F"] means F(¬G(P0)).

    Rules (applied to adjacent pairs, innermost token listed first):
        temporal_op, NEG  →  NEG, dual_op   (De Morgan: ¬G(x)=F(¬x), ¬F(x)=G(¬x))
        NEG, NEG          →  (removed)       (double negation)
        G, G              →  G               (idempotence)
        F, F              →  F               (idempotence)
    """
    dual = {"G": "F", "F": "G"}
    changed = True
    while changed:
        changed = False
        i = 1  # skip token 0 which is the predicate
        while i < len(tokens) - 1:
            a, b = tokens[i], tokens[i + 1]
            # De Morgan: push NEG inward past temporal op
            if a in dual and b == "NEG":
                tokens[i : i + 2] = ["NEG", dual[a]]
                changed = True
                # Don't advance; the new pair at i might trigger another rule
                continue
            # Double negation
            if a == "NEG" and b == "NEG":
                tokens[i : i + 2] = []
                changed = True
                continue
            # Idempotence
            if a == b and a in ("G", "F"):
                tokens[i : i + 2] = [a]
                changed = True
                continue
            i += 1
    return tokens


def compose_temporal(predicate: str, ops: list[str], negs: list[int]) -> TemporalAtom:
    """Compose temporal operators from K layers and simplify.

    Args:
        predicate: e.g. "P0"
        ops: list of K operators, each in {"id", "G", "F"}, bottom-up
        negs: list of K negation signs, each in {+1, -1}, bottom-up

    Returns:
        Simplified TemporalAtom.
    """
    tokens = [predicate]
    for k in range(len(ops)):
        if ops[k] != "id":
            tokens.append(ops[k])
        if negs[k] == -1:
            tokens.append("NEG")

    tokens = _simplify_tokens(tokens)

    # Parse result: tokens[0] is predicate, rest is NEG? then temporal ops
    negated = False
    temporal_ops_innermost_first: list[str] = []
    for tok in tokens[1:]:
        if tok == "NEG":
            negated = not negated
        else:
            temporal_ops_innermost_first.append(tok)

    # temporal_ops stored outermost-first
    temporal_ops = list(reversed(temporal_ops_innermost_first))
    return TemporalAtom(predicate, negated, temporal_ops)


def extract_formula(
    sel_weights: list[list[list[float]]],
    neg_weights: list[list[float]],
    prop_weights: list[list[float]],
    agg_weight: float,
    predicate_names: list[str],
) -> And | Or:
    """Extract interpretable formula from trained network weights.

    Args:
        sel_weights: K layers x N predicates x 3 (id/G/F logits)
        neg_weights: K layers x N predicates (negation logits, >=0 means positive)
        prop_weights: N//2 pairs x 3 (and/or/implies logits)
        agg_weight: scalar (>=0 means and, <0 means or)
        predicate_names: N predicate names

    Returns:
        Simplified sympy boolean formula.
    """
    K = len(sel_weights)
    N = len(predicate_names)
    op_names = ["id", "G", "F"]
    conn_names = ["and", "or", "implies"]

    # 1. Discretize
    ops = [
        [op_names[max(range(3), key=lambda j: sel_weights[k][i][j])] for i in range(N)]
        for k in range(K)
    ]
    negs = [[1 if neg_weights[k][i] >= 0 else -1 for i in range(N)] for k in range(K)]
    connectives = [
        conn_names[max(range(3), key=lambda j, row=row: row[j])] for row in prop_weights
    ]
    agg = "and" if agg_weight >= 0 else "or"

    # 2. Compose temporal atoms
    atoms = [
        compose_temporal(name, [ops[k][i] for k in range(K)], [negs[k][i] for k in range(K)])
        for i, name in enumerate(predicate_names)
    ]

    # 3. Build sympy formula
    symbols = {str(atom): Symbol(str(atom)) for atom in atoms}

    clauses = []
    for j in range(N // 2):
        left = symbols[str(atoms[2 * j])]
        right = symbols[str(atoms[2 * j + 1])]
        if connectives[j] == "and":
            clauses.append(And(left, right))
        elif connectives[j] == "or":
            clauses.append(Or(left, right))
        elif connectives[j] == "implies":
            clauses.append(Implies(left, right))

    if agg == "and":
        formula = And(*clauses)
    else:
        formula = Or(*clauses)

    # 4. Simplify
    return simplify_logic(formula, form="cnf")


def formula_to_str(formula) -> str:
    """Convert a sympy boolean formula to a human-readable temporal logic string.

    Sympy's str representation already uses &, |, ~ which map naturally
    to temporal logic notation. This function returns the sympy string as-is.
    """
    return str(formula)
