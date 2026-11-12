"""Apriori wrapper: support / confidence / lift, filtered."""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def mine_rules(onehot: pd.DataFrame, min_support=0.05, min_lift=1.2, min_conf=0.6, consequent_prefix=None,
               max_len=4):
    """With consequent_prefix, keep only rules whose consequent is ONLY the target item(s)
    (e.g. {debtor, low approval} -> {dropout}), which is the readable early-warning form."""
    fi = apriori(onehot.astype(bool), min_support=min_support, use_colnames=True, max_len=max_len)
    if fi.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    r = association_rules(fi, metric="confidence", min_threshold=min_conf)
    r = r[r["lift"] >= min_lift]
    if consequent_prefix:
        is_target = lambda s: all(str(i).startswith(consequent_prefix) for i in s)
        r = r[r["consequents"].apply(is_target) & ~r["antecedents"].apply(lambda s: any(is_target([i]) for i in s))]
    r = r.assign(antecedents=r["antecedents"].apply(lambda s: ", ".join(sorted(s))),
                 consequents=r["consequents"].apply(lambda s: ", ".join(sorted(s))))
    return r.sort_values(["lift", "support"], ascending=False)[["antecedents", "consequents", "support", "confidence", "lift"]]


def surprising_rules(onehot: pd.DataFrame, wanted: dict, min_support: float = 0.005) -> list:
    """Apriori rules that sound like common sense but that the data contradicts (lift below 1).

    wanted = {key: (antecedent_items, consequent_item, belief)}. Apriori runs on the items those rules use,
    then the rule whose 'if' side is exactly antecedent_items and 'then' side consequent_item is picked out.
    """
    items = sorted({i for a, c, _ in wanted.values() for i in [*a, c]})
    fi = apriori(onehot[items].astype(bool), min_support=min_support, use_colnames=True, max_len=4)
    found = association_rules(fi, metric="confidence", min_threshold=0.0)
    out = []
    for key, (antecedent, consequent, belief) in wanted.items():
        match = found[(found["antecedents"] == frozenset(antecedent)) & (found["consequents"] == frozenset([consequent]))]
        if match.empty:
            continue
        r = match.iloc[0]
        out.append(dict(key=key, belief=belief, antecedents=", ".join(antecedent), consequents=consequent,
                        support=float(r["support"]), confidence=float(r["confidence"]), lift=float(r["lift"]),
                        base_rate=float(onehot[consequent].mean())))
    return out
