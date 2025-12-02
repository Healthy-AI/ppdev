import itertools
import collections
import warnings
import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas.api.types import is_numeric_dtype

try:
    from .variables import COREVITAS_DATA
except ImportError:
    import os
    import pickle
    if not os.environ.get("COREVITAS_DATA_PATH"):
        raise ValueError(
            "The environment variable 'COREVITAS_DATA_PATH' is not set."
        )
    with open(os.environ("COREVITAS_DATA_PATH"), "rb") as f:
        COREVITAS_DATA = pickle.load(f)


# -----------------------------------------------------------------------------
# -- Drugs and variables ------------------------------------------------------
# -----------------------------------------------------------------------------

def get_drug_variables(drug):
    drug_variables = [v for v in COREVITAS_DATA.variables.keys() if "%s" in v]
    drug_variables.extend(["%s_since", "%s_until", "hx2%s"])
    DrugVariables = collections.namedtuple(
        "DrugVariables",
        [s.replace("%s", "").replace("_", "") for s in drug_variables]
    )
    return DrugVariables._make(map(lambda v: v % drug, drug_variables))


def get_drugs(drug_class=None, exclude=None):
    if exclude is not None:
        if drug_class is not None:
            raise ValueError(
                "Both `drug_class` and `exclude` cannot be specified at the "
                "same time."
            )
        drugs = []
        for drug_class in COREVITAS_DATA.drug_classes:
            if drug_class != exclude:
                drugs.extend(COREVITAS_DATA.drug_classes[drug_class])
        return drugs
    if drug_class is not None:
        return COREVITAS_DATA.drug_classes[drug_class]
    else:
        return list(itertools.chain.from_iterable(COREVITAS_DATA.drug_classes.values()))


def get_all_variables():
    all_variables = {}
    for v, i in COREVITAS_DATA.variables.items():
        if "%s" in v:
            for d in get_drugs():
                all_variables[v % d] = i
        else:
            all_variables[v] = i
    return all_variables


def is_unchanged(X, drug):
    drug_vars = get_drug_variables(drug)
    return (
        (X[drug_vars.add].eq(0) | X[drug_vars.add].isna())
        & (X[drug_vars.btwadd].eq(0) | X[drug_vars.btwadd].isna())
        & (X[drug_vars.disc].eq(0) | X[drug_vars.disc].isna())
        & (X[drug_vars.vindadd].eq(0) | X[drug_vars.vindadd].isna())
        & (X[drug_vars.vinddisc].eq(0) | X[drug_vars.vinddisc].isna())
    )


def add_at_visit(X, drug):
    drug_vars = get_drug_variables(drug)
    return (
        X[drug_vars.add].eq(1)
        & X[drug_vars.btwadd].eq(0)
        & X[drug_vars.disc].eq(0)
        & X[drug_vars.vindadd].eq(1)
        & pd.isna(X[drug_vars.vinddisc])
    )


def stop_at_visit(X, drug):
    drug_vars = get_drug_variables(drug)
    return (
        X[drug_vars.add].eq(0)
        & X[drug_vars.btwadd].eq(0)
        & X[drug_vars.disc].eq(-1)
        & X[drug_vars.vindadd].isna()
        & X[drug_vars.vinddisc].eq(1)
    )


def add_before_visit(X, drug):
    drug_vars = get_drug_variables(drug)
    return (
        X[drug_vars.add].eq(1)
        & X[drug_vars.btwadd].eq(0)
        & X[drug_vars.disc].eq(0)
        & X[drug_vars.vindadd].eq(2)
        & X[drug_vars.vinddisc].isna()
    )


def stop_before_visit(X, drug):
    drug_vars = get_drug_variables(drug)
    return (
        X[drug_vars.add].eq(0)
        & X[drug_vars.btwadd].eq(0)
        & X[drug_vars.disc].eq(-1)
        & X[drug_vars.vindadd].isna()
        & X[drug_vars.vinddisc].eq(2)
    )


def add_before_stop_at_visit(X, drug):
    drug_vars = get_drug_variables(drug)
    return (
        X[drug_vars.add].eq(1)
        & X[drug_vars.btwadd].eq(1)
        & X[drug_vars.disc].eq(-1)
        & X[drug_vars.vindadd].eq(2)
        & X[drug_vars.vinddisc].eq(1)
    )


def add_before_stop_before_visit(X, drug):
    drug_vars = get_drug_variables(drug)
    return (
        X[drug_vars.add].eq(1)
        & X[drug_vars.btwadd].eq(1)
        & X[drug_vars.disc].eq(-1)
        & X[drug_vars.vindadd].eq(2)
        & X[drug_vars.vinddisc].eq(2)
    )


def add_before_visit_any(X, drug):
    return (
        add_before_visit(X, drug)
        | add_before_stop_at_visit(X, drug)
        | add_before_stop_before_visit
    )


def stop_before_visit_any(X, drug):
    return (
        stop_before_visit(X, drug)
        | add_before_stop_before_visit(X, drug)
    )


def add_at_visit_any(X, drug):
    return add_at_visit(X, drug)


def stop_at_visit_any(X, drug):
    return (
        stop_at_visit(X, drug)
        | add_before_stop_at_visit(X, drug)
    )


def add_any(X, drug):
    return (
        add_at_visit(X, drug)
        | add_before_visit(X, drug)
        | add_before_stop_at_visit(X, drug)
        | add_before_stop_before_visit(X, drug)
    )


def stop_any(X, drug):
    return (
        stop_at_visit(X, drug)
        | stop_before_visit(X, drug)
        | add_before_stop_at_visit(X, drug)
        | add_before_stop_before_visit(X, drug)
    )


def _aggregate_init(X, c_init):
    assert X.loc[X.visitdate.isna(), c_init].isna().all(axis=1).all()
    init = X[c_init].max(axis=1)
    idxmax = init.replace(0, np.nan).groupby(X.id).idxmax()
    init = pd.Series(0.0, index=X.index).where(init.notna(), np.nan)
    init.loc[idxmax[idxmax.notna()]] = 1.
    init[X.visitdate.isna()] = np.nan
    return init


def assign_drug_class_variables(X):
    drug_classes = ["csdmard", "tnf", "nontnf", "jak"] + [None]
    
    # Aggregate `presX`.
    for drug_class in drug_classes:
        c = f"pres_{drug_class}" if drug_class is not None else "pres_dmard"
        c_pres = [f"pres_{d}" for d in get_drugs(drug_class)]
        pres = X[c_pres].max(axis=1)
        X[c] = pres.astype("float64")
    
    X["pres_bio"] = X[["pres_tnf", "pres_nontnf"]].max(axis=1)
    X["pres_biojak"] = X[["pres_tnf", "pres_nontnf", "pres_jak"]].max(axis=1)
    
    # Aggregate `initX`.
    for drug_class in drug_classes:
        c = f"init{drug_class}" if drug_class is not None else "initdmard"
        c_init = [f"init{d}" for d in get_drugs(drug_class)]
        init = _aggregate_init(X, c_init)
        X[c] = init
    
    X["initbio"] = _aggregate_init(X, ["inittnf", "initnontnf"])
    X["initbiojak"] = _aggregate_init(X, ["inittnf", "initnontnf", "initjak"])
    
    # Aggregate `hxX`.
    for drug_class in drug_classes:
        c = f"hx{drug_class}" if drug_class is not None else "hxdmard"
        c_hx = [f"hx{d}" for d in get_drugs(drug_class)]
        hx = X[c_hx].max(axis=1)
        X[c] = hx
    
    X["hxbio"] = X[["hxtnf", "hxnontnf"]].max(axis=1)
    X["hxbiojak"] = X[["hxtnf", "hxnontnf", "hxjak"]].max(axis=1)

    # Aggregate `hx2X`.
    for drug_class in drug_classes:
        c = f"hx2{drug_class}" if drug_class is not None else "hx2dmard"
        c_hx2 = [f"hx2{d}" for d in get_drugs(drug_class)]
        hx2 = X[c_hx2].max(axis=1)
        X[c] = hx2
    
    X["hx2bio"] = X[["hx2tnf", "hx2nontnf"]].max(axis=1)
    X["hx2biojak"] = X[["hx2tnf", "hx2nontnf", "hx2jak"]].max(axis=1)

    return X


def get_drug_periods(x):
    """Get the duration of all prescribed drugs.

    For each drug X, this function creates two columns: `X_since` and 
    `X_until`. These columns contain the start and stop dates of the drug
    whenever the drug is prescribed.
    """

    c_since = [f"{d}_since" for d in get_drugs()]
    c_until = [f"{d}_until" for d in get_drugs()]
    y = pd.DataFrame(np.nan, x.index, c_since + c_until, dtype="datetime64[ns]")
    
    c_dt = "date" if "date" in x.columns else "visitdate"

    for drug in get_drugs():
        drug_vars = get_drug_variables(drug)

        if x[drug_vars.pres].fillna(0).max() == 0:
            continue
        
        if x[drug_vars.pres].iat[0] == 1:
            adddt = x[drug_vars.adddt].iat[0]
            since = adddt if pd.notna(adddt) else x[c_dt].iat[0]
            y[drug_vars.since].iat[0] = since
        
        for idx, row in x.iterrows():
            if row[drug_vars.btwadd] == 1:
                if row[drug_vars.vinddisc] == 1:
                    y.at[idx, drug_vars.since] = row[drug_vars.adddt]
                    y.at[idx, drug_vars.until] = row[drug_vars.discdt]
                else:
                    continue
            if row[drug_vars.add] == 1:
                y.at[idx, drug_vars.since] = row[drug_vars.adddt]
            if row[drug_vars.disc] == -1:
                y.at[idx, drug_vars.until] = row[drug_vars.discdt]
        
        if x[drug_vars.pres].iat[-1] == 1:
            # `X_discdt` should always be missing when `X_pres=1` at the last visit.
            discdt = x[drug_vars.discdt].iat[-1]
            until = discdt if pd.notna(discdt) else x[c_dt].iat[-1]
            y[drug_vars.until].iat[-1] = until
        
        assert y[drug_vars.since].first_valid_index() is not None
        assert y[drug_vars.until].first_valid_index() is not None

    # Propagate dates forward.
    for drug in get_drugs():
        drug_vars = get_drug_variables(drug)

        idxs = y[drug_vars.since].first_valid_index()
        idxu = y[drug_vars.until].first_valid_index()
        while idxs is not None and idxu is not None:
            y.loc[idxs:idxu, drug_vars.since] = y.at[idxs, drug_vars.since]
            y.loc[idxs:idxu, drug_vars.until] = y.at[idxu, drug_vars.until]
            idxs = y[drug_vars.since].loc[idxu+1:].first_valid_index()
            idxu = y[drug_vars.until].loc[idxu+1:].first_valid_index()
            if not (idxs is None) == (idxu is None):
                if idxu is None:
                    assert y.at[idxs, drug_vars.since] > x.at[idxs, c_dt]
                if idxs is None:
                    assert y.at[idxu, drug_vars.until] <= x.at[idxu, c_dt]
        
        # Remove past start and stop dates.
        #
        # Check `presX` to avoid inserting `NaT` values in the last row of
        # `y` when `presX=1`.
        setna = y[drug_vars.until].le(x[c_dt]) & x[drug_vars.pres].ne(1)
        y.loc[setna, drug_vars.since] = np.datetime64("NaT")
        y.loc[setna, drug_vars.until] = np.datetime64("NaT")
    
    return y


def get_therapy_label(x):
    if x.pres_dmard == 0:
        return "No DMARD"
    if x.pres_tnf == 0 and x.pres_nontnf == 0 and x.pres_jak == 0:
        return "csDMARD therapy"
    if (
        (x.pres_tnf == 1 and x.pres_nontnf == 1)
        or (x.pres_tnf == 1 and x.pres_jak == 1)
        or (x.pres_nontnf == 1 and x.pres_jak == 1)
        or (x.pres_tnf == 1 and x.pres_nontnf == 1 and x.pres_jak == 1)
    ):
        return "Other therapy"
    n_dmards = x[[f"pres_{d}" for d in get_drugs()]].sum()
    cm = "combo" if n_dmards > 1 else "mono"
    if x.pres_tnf == 1:
        return f"TNF {cm}"
    if x.pres_nontnf == 1:
        return f"non-TNF {cm}"
    if x.pres_jak == 1:
        return f"JAK {cm}"
    raise ValueError(
        "Unknown combination of drugs.\n"
        f"{x[['pres_dmard', 'pres_tnf', 'pres_nontnf', 'pres_jak']]}"
    )


def assign_therapy_labels(X):
    X["therapy"] = X.apply(get_therapy_label, axis=1).astype("category")
    return X


def categorize_reason_code(c):
    if c in COREVITAS_DATA.reason_codes:
        return COREVITAS_DATA.reason_codes[c]["reason_group"]
    else:
        return c


def categorize_bp(row):
    """Classify blood pressure into categories.

    For details, see https://en.wikipedia.org/wiki/Blood_pressure.

    Parameters
    ----------
    row : pd.Series
        A row of the data frame. Must contain `seatedbp1` and `seatedbp2`.

    Returns
    -------
    bp_cat : str
        The blood pressure classification.
    """
    if row.seatedbp1 >= 140 or row.seatedbp2 >= 90:
        return "htn stage 2"
    elif row.seatedbp1 >= 130 or row.seatedbp2 >= 80:
        return "htn stage 1"
    elif 120 <= row.seatedbp1 < 130 and row.seatedbp2 < 80:
        return "elevated"
    elif row.seatedbp1 < 120 and row.seatedbp2 < 80:
        return "normal"
    else:
        return np.nan


# -----------------------------------------------------------------------------
# -- Data cleaning ------------------------------------------------------------
# -----------------------------------------------------------------------------

def select_subsequence(x, invalid, first=True):
    """Get a mask for the subsequence specified by `first`."""
    cum_invalid = invalid.loc[x.index].cumsum()
    if first:
        return cum_invalid.eq(0)
    else:
        num_invalid = cum_invalid.max()
        if num_invalid == 0:
            return pd.Series(False, index=x.index)
        else:
            mask = cum_invalid.eq(num_invalid)
            mask[mask.idxmax()] = False
            return mask


def split_on_invalid(X, invalid):
    """Split sequences on invalid rows."""

    mask1 = X.groupby("id", group_keys=False).progress_apply(
        select_subsequence,
        invalid=invalid,
        first=True,
    )
    X1 = X.loc[mask1].copy()

    mask2 = X.groupby("id", group_keys=False).progress_apply(
        select_subsequence,
        invalid=invalid,
        first=False,
    )
    X2 = X.loc[mask2].copy()

    X1["id"] += "_1"
    X2["id"] += "_2"

    return pd.concat([X1, X2]).sort_index()


# -----------------------------------------------------------------------------
# -- Non-registry events ------------------------------------------------------
# -----------------------------------------------------------------------------

def aggregate_events(s):
    if s.isna().all():
        return np.nan
    elif s.nunique() == 1:
        return s.dropna().iat[0]
    else:
        assert is_numeric_dtype(s)
        assert s.isin([-1, 0, np.nan]).all() or s.isin([0, 1, np.nan]).all()
        return s.replace(0, np.nan).dropna().iat[0]


def _get_pres_status(X, idx, c_dt):
    assert X.at[idx, "id"] == X.at[idx-1, "id"]
    pres = {}
    for drug in get_drugs():
        drug_vars = get_drug_variables(drug)
        if (
            X.at[idx-1, drug_vars.pres] == 1
            and X.at[idx, c_dt] < X.at[idx-1, drug_vars.until]
        ):
            pres[drug_vars.pres] = 1.
        elif (
            X.loc[idx, drug_vars.pres] == 1
            and X.at[idx, c_dt] >= X.at[idx, drug_vars.since]
        ):
            pres[drug_vars.pres] = 1.
        elif (
            X.at[idx, drug_vars.btwadd] == 1
            and X.at[idx, c_dt] >= X.at[idx, drug_vars.adddt]
            and X.at[idx, c_dt] < X.at[idx, drug_vars.discdt]
        ):
            pres[drug_vars.pres] = 1.
        else:
            pres[drug_vars.pres] = 0.
    return pres


def _get_hx2_status(X, idx):
    assert X.at[idx, "id"] == X.at[idx-1, "id"]
    hx2 = {}
    for drug in get_drugs():
        drug_vars = get_drug_variables(drug)
        if X.at[idx-1, drug_vars.hx] == 1:
            hx2[drug_vars.hx2] = 1.
        else:
            hx2[drug_vars.hx2] = X.at[idx-1, drug_vars.init]
    return hx2


def _get_row_for_added_drug(X, idx, drug):
    drug_vars = get_drug_variables(drug)
    add = pd.Series(None, index=X.columns, name=idx, dtype=object)
    add["id"] = X.loc[idx, "id"]
    add["date"] = X.loc[idx, drug_vars.adddt]
    add[drug_vars.pres] = 1.
    add[drug_vars.add] = 1.
    add[drug_vars.disc] = 0.
    add[drug_vars.hx2] = 0. if X.loc[idx, drug_vars.init] == 1 else 1.
    add[drug_vars.adddt] = X.loc[idx, drug_vars.adddt]
    # Update `presX` and `hx2X` for other drugs.
    pres = _get_pres_status(X, idx, drug_vars.adddt)
    hx2 = _get_hx2_status(X, idx)
    for k, v in (pres | hx2).items():
        if not k.endswith(drug):
            add[k] = v
    return add


def _get_row_for_stopped_drug(X, idx, drug):
    drug_vars = get_drug_variables(drug)
    disc = pd.Series(None, index=X.columns, name=idx, dtype=object)
    disc["id"] = X.loc[idx, "id"]
    disc["date"] = X.loc[idx, drug_vars.discdt]
    disc[drug_vars.pres] = 0.
    disc[drug_vars.add] = 0.
    disc[drug_vars.disc] = -1.
    disc[drug_vars.hx2] = 1.
    disc[drug_vars.discdt] = X.loc[idx, drug_vars.discdt]
    # Update `presX` and `hx2X` for other drugs.
    pres = _get_pres_status(X, idx, drug_vars.discdt)
    hx2 = _get_hx2_status(X, idx)
    for k, v in (pres | hx2).items():
        if not k.endswith(drug):
            disc[k] = v
    return disc


def extract_add_prior_to_visit(X):
    add_prior_to_visit = pd.Series(False, index=X.index)
    for drug in get_drugs():
        add_prior_to_visit |= add_before_visit(X, drug)
    
    # We do not consider changes before the first visit.
    add_prior_to_visit &= X.visit_order.gt(1)

    out = []
    n = add_prior_to_visit.sum()
    for idx, row in tqdm(X[add_prior_to_visit].iterrows(), total=n):
        added_drugs = [
            d for d in get_drugs() if (
                row[f"add{d}"] == 1
                and row[f"btwadd{d}"] == 0
                and row[f"disc{d}"] == 0
                and row[f"vindadd{d}"] == 2
                and pd.isna(row[f"vinddisc{d}"])
            )
        ]

        for drug in added_drugs:
            assert row[f"{drug}_adddt"] < row["visitdate"]
            add = _get_row_for_added_drug(X, idx, drug)
            out.append(add)

    return pd.DataFrame(out).astype(X.dtypes)


def extract_stop_prior_to_visit(X):
    stop_prior_to_visit = pd.Series(False, index=X.index)
    for drug in get_drugs():
        stop_prior_to_visit |= stop_before_visit(X, drug)

    # We do not consider changes before the first visit.
    stop_prior_to_visit &= X.visit_order.gt(1)

    out = []
    n = stop_prior_to_visit.sum()
    for idx, row in tqdm(X[stop_prior_to_visit].iterrows(), total=n):
        stopped_drugs = [
            d for d in get_drugs() if (
                row[f"add{d}"] == 0
                and row[f"btwadd{d}"] == 0
                and row[f"disc{d}"] == -1
                and pd.isna(row[f"vindadd{d}"])
                and row[f"vinddisc{d}"] == 2
            )
        ]

        for drug in stopped_drugs:
            assert row[f"{drug}_discdt"] < row["visitdate"]
            assert row[f"hx{drug}"] == 1
            disc = _get_row_for_stopped_drug(X, idx, drug)
            out.append(disc)

    return pd.DataFrame(out).astype(X.dtypes)


def extract_btwadd(X):
    # We do not consider changes before the first visit.
    c_btwadd = [f"btwadd{d}" for d in get_drugs()]
    assert X.loc[X.visit_order.gt(1), c_btwadd].sum(axis=1).max() <= 1
    btwadd = X[c_btwadd].max(axis=1).eq(1) & X.visit_order.gt(1)

    out = []
    n = btwadd.sum()
    for idx, row in tqdm(X[btwadd].iterrows(), total=n):
        is_btwadd = row[c_btwadd].eq(1)
        drug_btwadd = is_btwadd[is_btwadd].index.item().replace("btwadd", "")

        assert row[f"{drug_btwadd}_adddt"] < row["visitdate"]
        assert row[f"{drug_btwadd}_discdt"] <= row["visitdate"]
        assert row[f"hx{drug_btwadd}"] == 1

        add = _get_row_for_added_drug(X, idx, drug_btwadd)
        out.append(add)

        if row[f"vinddisc{drug_btwadd}"] == 2:
            # The drug was stopped prior to the visit.
            disc = _get_row_for_stopped_drug(X, idx, drug_btwadd)
            out.append(disc)

    return pd.DataFrame(out).astype(X.dtypes)


# -----------------------------------------------------------------------------
# -- Cohort selection ---------------------------------------------------------
# -----------------------------------------------------------------------------

def assign_stage(X):
    """Assign the stage of treatment.

    This function adds a column `stage` to the DataFrame `X` that indicates the
    stage of the treatment. The index visit (baseline) is stage 1, the first
    follow-up visit is stage 2, and so on. Visits before the index visit are
    assigned stage -1.
    """
    def update_stage(x):
        try:
            i = x[x == 1].index[0]
        except IndexError:
            return x
        j = x.index[-1]
        x.loc[i:j] = range(1, 2 + j-i)
        return x

    if "stage" in X.columns:
        X = X.drop(columns="stage")

    # Default stage is -1.
    X = X.assign(stage=-1)

    # Assign the stage -2 to the non-registry visits.
    is_registry_visit = X.visitdate.notna()
    X.loc[~is_registry_visit, "stage"] = -2

    # Assign the stage 1 to the index visits.
    is_baseline = X.hxbiojak.eq(0) & X.initbiojak.eq(1)
    assert X.loc[is_baseline, "visitdate"].notna().all()
    X.loc[is_baseline, "stage"] = 1

    # Update the stage for post-baseline registry visits.
    X.loc[is_registry_visit, "stage"] = \
        X[is_registry_visit].groupby("id", group_keys=False).stage.apply(update_stage)

    return X.astype({"stage": "float64"})


def update_exclude_reason(
    X,
    exclude,
    reason,
    c_reason="exclude_reason",
    only_if_missing=False,
):
    has_reason = X[c_reason].str.len() > 0

    mask = exclude & ~has_reason
    X.loc[mask, c_reason] = reason

    if not only_if_missing:
        mask = exclude & has_reason
        X.loc[mask, c_reason] = X.loc[mask, c_reason].str.cat(
            [reason] * sum(mask),
            sep=","
        )
    
    return X


def require_valid_therapies(
    x,
    exclude_invalid=True,
    exclude_reason=None,
    only_if_missing=False,
):
    if not x.stage.eq(1).any():
        return x
    idx_first_invalid = x[
        x.stage.ge(1) & x.therapy.eq("Other therapy")
    ].first_valid_index()
    if exclude_invalid:
        return x if idx_first_invalid is None else x.loc[:idx_first_invalid-1]
    else:
        exclude = x.index >= idx_first_invalid
        x.loc[exclude & x.stage.ne(-2), "stage"] = -1
        if exclude_reason is not None:
            x = update_exclude_reason(
                x,
                exclude,
                exclude_reason,
                only_if_missing=only_if_missing,
            )
        return x


def require_regular_followup(
    x,
    delta_min=90,
    delta_max=270,
    exclude_invalid=True,
    exclude_reason=None,
    only_if_missing=False,
):
    if not x.stage.eq(1).any():
        return x
    # Extract registry visits.
    y = x[x.visitdate.notna()]
    # Compute the number of days until the next registry visit.
    delta = y.visitdate.diff().dt.days.shift(-1)
    # Get the first invalid index (we include this index as well).
    idx_first_invalid = y[
        y.stage.ge(1) & (delta.lt(delta_min) | delta.gt(delta_max))
    ].first_valid_index()
    if exclude_invalid:
        return x.loc[:idx_first_invalid]
    else:
        exclude = x.index > idx_first_invalid
        x.loc[exclude & x.stage.ne(-2), "stage"] = -1
        if exclude_reason is not None:
            x = update_exclude_reason(
                x,
                exclude,
                exclude_reason,
                only_if_missing=only_if_missing,
            )
        return x


def require_recorded_outcome(
    x,
    exclude_invalid=True,
    exclude_reason=None,
    only_if_missing=False
):
    if not x.stage.eq(1).any():
        return x
    # Extract registry visits.
    y = x[x.visitdate.notna()]
    # Identify registry visits with missing outcome (no CDAI measured at the
    # following registry visit).
    lacks_outcome = y.cdai.shift(-1).isna()
    # Get the first invalid index (there will always be one).
    idx_first_invalid = y[y.stage.ge(1) & lacks_outcome].first_valid_index()
    if exclude_invalid:
        return x.loc[:idx_first_invalid-1]
    else:
        exclude = x.index >= idx_first_invalid
        x.loc[exclude & x.stage.ne(-2), "stage"] = -1
        if exclude_reason is not None:
            x = update_exclude_reason(
                x,
                exclude,
                exclude_reason,
                only_if_missing=only_if_missing,
            )
        return x


def require_fixed_followup(
    x,
    followup,
    exclude_invalid=True,
    exclude_reason=None,
    only_if_missing=False,
):
    if not x.stage.eq(1).any():
        return x
    visitdate_baseline = x.loc[x.stage.eq(1), "visitdate"].item()
    idx_first_invalid = x[
        (x.visitdate - visitdate_baseline).dt.days.gt(followup)
    ].first_valid_index()
    if idx_first_invalid is None:
        idx_first_invalid = x.index[x.stage.eq(1)].item() - 1
    else:
        idx_first_invalid -= 1
    if exclude_invalid:
        return x.loc[:idx_first_invalid]
    else:
        exclude = x.index > idx_first_invalid
        x.loc[exclude & x.stage.ne(-2), "stage"] = -1
        if exclude_reason is not None:
            x = update_exclude_reason(
                x,
                exclude,
                exclude_reason,
                only_if_missing=only_if_missing,
            )
        return x


def filter_cohort(X, c_stage="stage"):
    """Extract baseline and post-baseline visits, including non-registry visits."""
    mask = X[c_stage].gt(1)  # Post-baseline registry visists
    idx_postbaseline = mask[mask].index
    return X[
        X[c_stage].eq(1)  # Index visit
        | X.index.isin(idx_postbaseline)  # All post-baseline visits
    ]


# -----------------------------------------------------------------------------
# -- Visualization ------------------------------------------------------------
# -----------------------------------------------------------------------------

def _get_drug_periods_for_plotting(x):
    drug_periods = collections.defaultdict(list)

    c_dt = "date" if "date" in x.columns else "visitdate"

    y = x.reset_index(drop=True).reset_index()
    
    for drug in get_drugs():
        drug_vars = get_drug_variables(drug)

        for idx in y.groupby(drug_vars.since).index.first():
            start_date = y.at[idx, drug_vars.since]
            end_date = y.at[idx, drug_vars.until]
            
            if start_date == x[c_dt].iat[0] and pd.isna(y.at[idx, drug_vars.adddt]):
                drug_periods[drug].append(
                    (start_date.replace(month=1, day=1), start_date, "dotted")
                )
            
            drug_periods[drug].append((start_date, end_date, "solid"))

            if end_date == x[c_dt].iat[-1] and pd.isna(y.at[idx, drug_vars.discdt]):
                drug_periods[drug].append(
                    (end_date, end_date.replace(month=12, day=31), "dotted")
                )
    
    return drug_periods


def visualize_therapies(X, patient_id, figsize=(10, 6), idx_highlight=None):
    x = X[X.id == patient_id]
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)

    y_values = {drug: i for i, drug in enumerate(get_drugs())}
    
    colors = {drug: "tab:blue" for drug in get_drugs("csdmard")}
    colors.update({drug: "tab:orange" for drug in get_drugs("tnf")})
    colors.update({drug: "tab:green" for drug in get_drugs("nontnf")})
    colors.update({drug: "tab:red" for drug in get_drugs("jak")})

    for drug, periods in _get_drug_periods_for_plotting(x).items():
        for start, end, ls in periods:
            if pd.isna(start) or pd.isna(end):
                warnings.warn(
                    f"Patient {patient_id} has missing date(s) for {drug}."
                )
                continue
            ax.plot(
                [start, end], [y_values[drug]]*2, marker="|", linestyle=ls,
                color=colors[drug], markersize=10, linewidth=5,
            )

    for idx, row in x.iterrows():
        for drug in get_drugs():
            if row[f"pres_{drug}"] == 1:
                ax.scatter(
                    row.date, y_values[drug], s=75, facecolor="white",
                    color=colors[drug], zorder=100, linewidth=2,
                )
        if idx_highlight is None:
            color = "black"
        else:
            color = "black" if idx != idx_highlight else "tab:red"
        linestyle = "--" if pd.notna(row.visitdate) else ":"
        ax.axvline(
            x=row.date, color=color, linestyle=linestyle, linewidth=1, 
            alpha=0.8, zorder=-100
        )
        edgecolor = "tab:green" if row.get("stage") == 1 else "none"
        ax.text(
            row.date, len(y_values)/2, row.therapy, rotation=90, 
            va="center", ha="center", fontsize=10,
            bbox=dict(facecolor="white", alpha=1, edgecolor=edgecolor),
        )

    ax.set_title(f"Patient {patient_id}\n")
    
    ax.grid(True, alpha=0.5)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator([1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks(x.visitdate)
    ax2.set_xticklabels(
        x.visitdate.dt.strftime("%Y-%m-%d"), rotation=30, ha="left",
    )

    ax.set_yticks(list(y_values.values()))
    ax.set_yticklabels(list(y_values.keys()))
    for ytick in ax.get_yticklabels():
        drug = ytick.get_text()
        ytick.set_color(colors[drug])

    fig.autofmt_xdate()

    return fig, ax


def get_therapy_summary(X, patient_id, drug_vars=None):
    x = X[X.id == patient_id].copy()

    c_pres = [f"pres_{d}" for d in get_drugs()]
    pres = x[c_pres].max().eq(1)

    c_btwadd = [f"btwadd{d}" for d in get_drugs()]
    btwadd = x[c_btwadd].max().eq(1)
    btwadd.index = pres.index
    
    used_drugs = pres | btwadd
    used_drugs = used_drugs[used_drugs].index.str.replace("pres_", "")
    
    info = []
    if drug_vars is None:
        drug_vars = [
            "pres_%s", "add%s", "btwadd%s", "disc%s", "vindadd%s",
            "vinddisc%s", "%s_adddt", "%s_discdt",
        ]
        if all([f"{drug}_since" in X.columns for drug in used_drugs]):
            drug_vars.append("%s_since")
        if all([f"{drug}_until" in X.columns for drug in used_drugs]):
            drug_vars.append("%s_until")
    for drug in used_drugs:
        info += list(map(lambda v: v % drug, drug_vars))
    
    if not "index" in x.columns:
        x = x.reset_index()
    index = ["index"]
    index.append("date" if "date" in x.columns else "visitdate")
    if "stage" in x.columns:
        index.append("stage")
    if "therapy" in x.columns:
        index.append("therapy")
    x = x.set_index(index)

    date_vars = [
        v for v in info 
        if v.endswith("dt") or v.endswith("since") or v.endswith("until")
    ]
    x[date_vars] = x[date_vars].apply(lambda s: s.dt.strftime("%Y-%m-%d"), 0)

    drug_vars = map(lambda v: v % "X", drug_vars)
    labels = pd.MultiIndex.from_product([used_drugs, drug_vars])
    x = x[info].set_axis(labels, axis=1)

    return x


def _get_sort_order(grouped_trajs):
    first_therapies = grouped_trajs.therapy.first().value_counts(ascending=True)

    sort_order = {t: {} for t, c in first_therapies.items() if c > 0}

    def recursively_add(d, i):
        this = grouped_trajs.nth(i)
        next = grouped_trajs.nth(i+1)

        for k, v in d.items():
            # Find next therapy for patients with current treatment "k".
            ids_k = this.loc[this.therapy.eq(k), "id"]
            after_k = next.loc[next.id.isin(ids_k), "therapy"]
            
            # Add therapies with non-zero occurrence.
            for t, n in after_k.value_counts(ascending=True).items():
                if n > 0:
                    if i + 2 < grouped_trajs.size().max():
                        v.update({t: {}})
                        recursively_add(v, i+1)
                    else:
                        v.update({t: None})
    
    # For each therapy, recursively sort the subsequent therapies.
    recursively_add(sort_order, 0)

    return sort_order


def visualize_sequential_therapies(
    input_trajs,
    traj_th=10,
    scale=50,
    fac=1,
    ax=None,
    figsize=(6, 4),
    default_len=200,
    color_mapper=None,
    show_errors=True,
    show_group_counts=True,
    error_type="std",
    exclude_trajectory=lambda x: False,
):  
    assert error_type in {"std", "iqr"}

    # Make trajectorires hashable by converting them into strings.
    trajs_grouped = input_trajs.groupby("id")
    trajs_str = trajs_grouped.therapy.agg(lambda t: " => ".join(t))
    unique_trajs = set(trajs_str)

    # Count the number of occurrences of each trajectory.
    traj_counts = {k: 0 for k in unique_trajs}
    traj_to_ids = {k: [] for k in unique_trajs}
    for pid, traj_str in trajs_str.items():
        traj_counts[traj_str] += 1
        traj_to_ids[traj_str].append(pid)
    
    # Keep only trajectories that occur more than `traj_th` times in data.
    traj_counts = {
        k: v for k, v in traj_counts.items()
        if v > traj_th and not exclude_trajectory(k)
    }
    
    # Regroup the trajectories to limit the number of recursions.
    s = trajs_str.map(lambda t: t in traj_counts)
    eligible_ids = s[s].index
    input_trajs = input_trajs[input_trajs.id.isin(eligible_ids)]
    grouped_trajs = input_trajs.groupby("id")
    
    # Sort the trajectories.
    
    sort_order = _get_sort_order(grouped_trajs)
    
    def sorter(item):
        s = []
        so = sort_order
        for t in item[0].split(" => "):
            s.append(list(so).index(t))
            so = so[t]
        return tuple(s)
    
    traj_counts = dict(sorted(traj_counts.items(), key=sorter))
    
    # Extract therapies included in the trajectories.
    therapies = input_trajs.therapy.value_counts().index.tolist()
    
    # Plot the trajectories.

    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=figsize)

    if show_group_counts:
        first_therapies = grouped_trajs.therapy.first().value_counts(ascending=True)
        first_therapies_ylim = {
            k: [None, None] for k, c in first_therapies.items() if c > 0
        }

    if color_mapper is None:
        import seaborn as sns
        colors = sns.color_palette("colorblind", n_colors=len(therapies))

    counts = np.array(list(traj_counts.values()))
    if np.ptp(counts) > 0:
        heights = 1 + scale * (counts - np.min(counts)) / np.ptp(counts)
    else:
        heights = counts

    y0 = 0

    for i, (traj_str, n_trajs) in enumerate(traj_counts.items()):
        traj_str_split = traj_str.split(" => ")
        traj_length = len(traj_str_split)

        # Compute statistics of the duration of each therapy.
        mask = input_trajs.id.isin(traj_to_ids[traj_str])
        trajs = input_trajs.loc[mask, ["id", "days_between_visits"]]
        ranks = trajs.groupby("id").cumcount()
        days_btw_visits = trajs.groupby(ranks).days_between_visits
        if error_type == "std":
            duration_m = days_btw_visits.mean()
            duration_e = days_btw_visits.std().iloc[:traj_length].values
        elif error_type == "iqr":
            duration_m = days_btw_visits.median()
            duration_q1 = days_btw_visits.quantile(q=0.25).iloc[:traj_length]
            duration_q3 = days_btw_visits.quantile(q=0.75).iloc[:traj_length]
            duration_e = np.vstack(
                ((duration_m - duration_q1),(duration_q3 - duration_m))
            )

        x0 = 0
        x = []
        height = heights[i]
        for j, t in enumerate(traj_str_split):
            c = color_mapper[t] if color_mapper else colors[therapies.index(t)]
            width = duration_m.iloc[j]
            if np.isnan(width):
                width = default_len
                kwargs = {"hatch": "///", "ec": "white"}
            else:
                kwargs = {"ec": "white"}
            ax.fill_between(
                np.linspace(x0, x0 + width),
                y0, y0 + height,
                step="pre",
                alpha=0.8,
                fc=c,
                lw=0.5,
                **kwargs,
            )
            x0 += width
            x += [x0]
            if show_group_counts and j == 0:
                if first_therapies_ylim[t][0] is None:
                    first_therapies_ylim[t][0] = y0
                first_therapies_ylim[t][1] = y0 + height
        
        bbox = dict(ec="white", fc="white", alpha=0.8, pad=1.5)
        ax.text(
            10, y0 + height / 2, str(n_trajs), ha="left", va="center",
            bbox=bbox,
        )
        
        if show_errors:
            y = np.linspace(y0, y0 + height, traj_length)[::-1]
            ax.errorbar(x, y, xerr=duration_e, fmt=".", c="k", elinewidth=0.5)

        if i + 1 < len(traj_counts):
            next_traj_str = list(traj_counts)[i+1]
            if next_traj_str and (next_traj_str.split(" => ")[0] == traj_str_split[0]):
                y0 += height + fac
            else:
                y0 += height + 3 * fac
    
    if show_group_counts:
        for t, (ymin, ymax) in first_therapies_ylim.items():
            c = color_mapper[t] if color_mapper else colors[therapies.index(t)]
            if ymin is not None and ymax is not None:
                bbox["ec"] = c
                count = first_therapies[t]
                ax.vlines(-100, ymin, ymax, color=c, lw=5, zorder=-100)
                ax.text(
                    -100, ymin + (ymax - ymin) / 2,
                    str(count),
                    ha="center",
                    va="center",
                    bbox=bbox,
                )

    import matplotlib.patches as mpatches
    if color_mapper:
        artists = [mpatches.Patch(color=color_mapper[d]) for d in therapies]
    else:
        artists = [mpatches.Patch(color=colors[i]) for i in range(len(therapies))]
    ax.legend(artists, therapies, loc="upper right", title="Therapy")
    
    ax.set_xlabel("Therapy duration (days)")
    ax.set_yticks([], [])
    
    return ax


# -----------------------------------------------------------------------------
# -- Other --------------------------------------------------------------------
# -----------------------------------------------------------------------------

def convert_dates(
    dates,
    base_date="1960-01-01",
    format="%Y-%m-%d",
    days_per_month = 30.4,
):
    """Convert dates on the %td format to the given string format."""
    base_date = datetime.datetime.strptime(base_date, format)
    converted_dates = []
    for d in dates:
        if pd.isnull(d):
            converted_dates.append(d)
        else:
            months = int(d)
            days = round(days_per_month * (d%1))
            new_date = base_date + relativedelta(months=months, days=days)
            converted_dates.append(new_date.strftime(format))
    return pd.Series(converted_dates, index=dates.index, name=dates.name)


def get_subsequences(data, n, return_grouped=True, allow_equality=True):
    """Get subsequences from grouped data.

    Parameters
    ----------
    data : pandas DataFrame
        A DataFrame which contains the column `id`.
    n : int
        The length of the subsequences.
    return_grouped : bool, default=True
        If True, return a DataFrameGroupBy object.
    allow_equality : bool, default=True
        If True, subsequences of length `n` are included. Otherwise, only
        subsequences of length > `n` are included.
    
    Returns
    -------
    subsequences
        Subsequences of length `n` or `n+1`, either as DataFrameGroupBy or 
        DataFrame.
    """
    grouped_data = data.groupby("id")
    mask = grouped_data.size().ge(n) if allow_equality else grouped_data.size().gt(n)
    eligible_ids = mask[mask].index
    subsequences = pd.concat(
        [grouped_data.get_group(pid).head(n) for pid in eligible_ids]
    )
    return subsequences.groupby("id") if return_grouped else subsequences
