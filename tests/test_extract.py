import pytest
from sympy import And, Implies, Or, Symbol, simplify_logic

from flora.extract import TemporalAtom, compose_temporal, extract_formula, formula_to_str


# ---------------------------------------------------------------------------
# TemporalAtom basics
# ---------------------------------------------------------------------------


def test_temporal_atom_str_plain():
    atom = TemporalAtom("P0", False, [])
    assert str(atom) == "P0"


def test_temporal_atom_str_negated():
    atom = TemporalAtom("P0", True, [])
    assert str(atom) == "~P0"


def test_temporal_atom_str_ops():
    atom = TemporalAtom("P0", False, ["G", "F"])
    assert str(atom) == "G(F(P0))"


def test_temporal_atom_str_negated_with_ops():
    atom = TemporalAtom("P0", True, ["G", "F"])
    assert str(atom) == "G(F(~P0))"


def test_temporal_atom_equality():
    a = TemporalAtom("P0", True, ["G"])
    b = TemporalAtom("P0", True, ["G"])
    assert a == b


def test_temporal_atom_hash():
    a = TemporalAtom("P0", True, ["G"])
    b = TemporalAtom("P0", True, ["G"])
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# ---------------------------------------------------------------------------
# compose_temporal
# ---------------------------------------------------------------------------


def test_identity():
    atom = compose_temporal("P0", ["id", "id"], [1, 1])
    assert atom == TemporalAtom("P0", False, [])
    assert str(atom) == "P0"


def test_single_G():
    atom = compose_temporal("P0", ["G", "id"], [1, 1])
    assert atom == TemporalAtom("P0", False, ["G"])
    assert str(atom) == "G(P0)"


def test_single_F():
    atom = compose_temporal("P0", ["F", "id"], [1, 1])
    assert atom == TemporalAtom("P0", False, ["F"])
    assert str(atom) == "F(P0)"


def test_negation_pushes_through_G():
    # Layer 0: G(P0), neg=-1 → ¬G(P0) → F(¬P0)
    # Layer 1: id, neg=+1 → F(¬P0)
    atom = compose_temporal("P0", ["G", "id"], [-1, 1])
    assert atom == TemporalAtom("P0", True, ["F"])
    assert str(atom) == "F(~P0)"


def test_negation_pushes_through_F():
    # Layer 0: F(P0), neg=-1 → ¬F(P0) → G(¬P0)
    atom = compose_temporal("P0", ["F", "id"], [-1, 1])
    assert atom == TemporalAtom("P0", True, ["G"])
    assert str(atom) == "G(~P0)"


def test_nested_GF():
    # Layer 0: F(P0), neg=+1 → F(P0)
    # Layer 1: G(F(P0)), neg=+1 → G(F(P0))
    atom = compose_temporal("P0", ["F", "G"], [1, 1])
    assert atom == TemporalAtom("P0", False, ["G", "F"])
    assert str(atom) == "G(F(P0))"


def test_nested_FG():
    atom = compose_temporal("P0", ["G", "F"], [1, 1])
    assert atom == TemporalAtom("P0", False, ["F", "G"])
    assert str(atom) == "F(G(P0))"


def test_complex_with_double_negation():
    # Layer 0: G(P0), neg=+1 → G(P0)
    # Layer 1: F(G(P0)), neg=-1 → ¬F(G(P0))
    # Simplify: ¬F(G(P0)) → G(¬G(P0)) → G(F(¬P0))
    atom = compose_temporal("P0", ["G", "F"], [1, -1])
    assert atom == TemporalAtom("P0", True, ["G", "F"])
    assert str(atom) == "G(F(~P0))"


def test_idempotence_GG():
    atom = compose_temporal("P0", ["G", "G"], [1, 1])
    assert atom == TemporalAtom("P0", False, ["G"])
    assert str(atom) == "G(P0)"


def test_idempotence_FF():
    atom = compose_temporal("P0", ["F", "F"], [1, 1])
    assert atom == TemporalAtom("P0", False, ["F"])
    assert str(atom) == "F(P0)"


def test_double_negation():
    atom = compose_temporal("P0", ["id", "id"], [-1, -1])
    assert atom == TemporalAtom("P0", False, [])
    assert str(atom) == "P0"


def test_single_negation_no_op():
    atom = compose_temporal("P0", ["id", "id"], [-1, 1])
    assert atom == TemporalAtom("P0", True, [])
    assert str(atom) == "~P0"


def test_three_layers():
    # Layer 0: G, +1 → G(P0)
    # Layer 1: F, +1 → F(G(P0))
    # Layer 2: G, +1 → G(F(G(P0)))
    atom = compose_temporal("P0", ["G", "F", "G"], [1, 1, 1])
    assert atom == TemporalAtom("P0", False, ["G", "F", "G"])
    assert str(atom) == "G(F(G(P0)))"


def test_neg_with_id_then_G():
    # Layer 0: id, -1 → ¬P0
    # Layer 1: G, +1 → G(¬P0)
    # Tokens: ["P0", "NEG", "G"] — no adjacent patterns to simplify
    atom = compose_temporal("P0", ["id", "G"], [-1, 1])
    assert atom == TemporalAtom("P0", True, ["G"])
    assert str(atom) == "G(~P0)"


def test_neg_G_then_neg():
    # Layer 0: G, -1 → tokens: P0, G, NEG → F(¬P0) after simplify: P0, NEG, F
    # Layer 1: id, -1 → add NEG: P0, NEG, F, NEG
    # Simplify: F, NEG → NEG, G: P0, NEG, NEG, G → P0, G
    atom = compose_temporal("P0", ["G", "id"], [-1, -1])
    assert atom == TemporalAtom("P0", False, ["G"])
    assert str(atom) == "G(P0)"


# ---------------------------------------------------------------------------
# extract_formula
# ---------------------------------------------------------------------------


def test_simple_conjunction():
    """G(P0) ∧ F(P1)"""
    sel = [
        [[0, 10, 0], [0, 0, 10]],  # layer 0: P0→G, P1→F
        [[10, 0, 0], [10, 0, 0]],  # layer 1: both id
    ]
    neg = [[1.0, 1.0], [1.0, 1.0]]
    prop = [[10.0, 0, 0]]  # ∧
    agg = 1.0

    formula = extract_formula(sel, neg, prop, agg, ["P0", "P1"])
    gp0 = Symbol("G(P0)")
    fp1 = Symbol("F(P1)")
    expected = And(gp0, fp1)
    assert formula == expected


def test_simple_disjunction():
    """G(P0) ∨ F(P1)"""
    sel = [
        [[0, 10, 0], [0, 0, 10]],
        [[10, 0, 0], [10, 0, 0]],
    ]
    neg = [[1.0, 1.0], [1.0, 1.0]]
    prop = [[0, 10.0, 0]]  # ∨
    agg = 1.0

    formula = extract_formula(sel, neg, prop, agg, ["P0", "P1"])
    gp0 = Symbol("G(P0)")
    fp1 = Symbol("F(P1)")
    expected = Or(gp0, fp1)
    assert formula == expected


def test_implication():
    """G(P0) → F(P1), CNF form: ~G(P0) | F(P1)"""
    sel = [
        [[0, 10, 0], [0, 0, 10]],
        [[10, 0, 0], [10, 0, 0]],
    ]
    neg = [[1.0, 1.0], [1.0, 1.0]]
    prop = [[0, 0, 10.0]]  # →
    agg = 1.0

    formula = extract_formula(sel, neg, prop, agg, ["P0", "P1"])
    gp0 = Symbol("G(P0)")
    fp1 = Symbol("F(P1)")
    # Implies in CNF becomes Or(Not(left), right)
    expected = simplify_logic(Implies(gp0, fp1), form="cnf")
    assert formula == expected


def test_agg_or():
    """Two pairs aggregated with OR."""
    sel = [
        [[0, 10, 0], [0, 0, 10], [10, 0, 0], [10, 0, 0]],
        [[10, 0, 0], [10, 0, 0], [10, 0, 0], [10, 0, 0]],
    ]
    neg = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    prop = [[10.0, 0, 0], [10.0, 0, 0]]  # both ∧
    agg = -1.0  # ∨

    formula = extract_formula(sel, neg, prop, agg, ["P0", "P1", "P2", "P3"])
    gp0 = Symbol("G(P0)")
    fp1 = Symbol("F(P1)")
    p2 = Symbol("P2")
    p3 = Symbol("P3")
    expected = simplify_logic(Or(And(gp0, fp1), And(p2, p3)), form="cnf")
    assert formula == expected


def test_full_walkthrough():
    """
    4 predicates, K=2.
    P0: layer0: G,+1 → G(P0). layer1: id,+1 → G(P0)
    P1: layer0: F,+1 → F(P1). layer1: G,-1 → ¬G(F(P1)) → F(¬F(P1)) → F(G(¬P1))
        wait — ¬F(P1) = G(¬P1), so: tokens P1,F,G,NEG
        simplify: G,NEG → NEG,F: P1,F,NEG,F → F,NEG → NEG,G: P1,NEG,G,F
        So F(G(¬P1))... But spec says "G(F(¬P1))". Let me retrace carefully.

    P1 tokens: start ["P1"]
        layer 0: op=F, neg=+1 → ["P1", "F"]
        layer 1: op=G, neg=-1 → ["P1", "F", "G", "NEG"]
    Simplify ["P1", "F", "G", "NEG"]:
        i=1: F,G — no rule. i=2: G,NEG → NEG,F → ["P1", "F", "NEG", "F"]
        restart: i=1: F,NEG → NEG,G → ["P1", "NEG", "G", "F"]
        restart: i=1: NEG,G — no rule (NEG is not a temporal op). i=2: G,F — no rule.
    Result: ["P1", "NEG", "G", "F"] → negated=True, ops innermost-first=[G,F], outermost-first=[F,G]
    str: F(G(~P1))

    P2: layer0: id,-1 → ["P2", "NEG"]. layer1: F,+1 → ["P2", "NEG", "F"]
        No simplification needed. negated=True, ops=[F]. str: F(~P2)

    P3: layer0: G,+1 → ["P3", "G"]. layer1: id,-1 → ["P3", "G", "NEG"]
        Simplify: G,NEG → NEG,F → ["P3", "NEG", "F"]. negated=True, ops=[F]. str: F(~P3)

    Connectives: pair0 → (idx 2), pair1 ∧ (idx 0)
    Agg: ∧

    Formula: (G(P0) → F(G(~P1))) ∧ (F(~P2) ∧ F(~P3))
    """
    sel = [
        [[0, 10, 0], [0, 0, 10], [10, 0, 0], [0, 10, 0]],  # layer 0: G, F, id, G
        [[10, 0, 0], [0, 10, 0], [0, 0, 10], [10, 0, 0]],  # layer 1: id, G, F, id
    ]
    neg = [
        [1.0, 1.0, -1.0, 1.0],  # layer 0
        [1.0, -1.0, 1.0, -1.0],  # layer 1
    ]
    prop = [
        [0, 0, 10.0],  # pair 0: →
        [10.0, 0, 0],  # pair 1: ∧
    ]
    agg = 1.0
    names = ["P0", "P1", "P2", "P3"]

    formula = extract_formula(sel, neg, prop, agg, names)

    gp0 = Symbol("G(P0)")
    fgnp1 = Symbol("F(G(~P1))")
    fnp2 = Symbol("F(~P2)")
    fnp3 = Symbol("F(~P3)")

    expected = simplify_logic(
        And(Implies(gp0, fgnp1), And(fnp2, fnp3)), form="cnf"
    )
    assert formula == expected


# ---------------------------------------------------------------------------
# formula_to_str
# ---------------------------------------------------------------------------


def test_formula_to_str():
    gp0 = Symbol("G(P0)")
    fp1 = Symbol("F(P1)")
    formula = And(gp0, fp1)
    s = formula_to_str(formula)
    assert "G(P0)" in s
    assert "F(P1)" in s
