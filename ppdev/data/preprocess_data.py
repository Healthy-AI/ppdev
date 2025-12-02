import warnings

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm

from ppdev.data import utils as utils

# Disable warnings. 
warnings.simplefilter(action="ignore", category=FutureWarning)

# Show progress bars.
tqdm.pandas()


def preprocess_data(raw_data):
    """Preprocess raw registry data.
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw registry data.
    
    Returns
    -------
    preprocessed_data : pd.DataFrame
        Preprocessed registry data.
    """
    # -------------------------------------------------------------------------
    # -- Select variables. ----------------------------------------------------
    # -------------------------------------------------------------------------

    # Select variables that exist in all versions of the registry.
    all_vars = {}
    for v, i in utils.get_all_variables().items():
        if v in raw_data.columns and i["version"] == "all":
            all_vars[v] = i
    data = raw_data.loc[:, all_vars.keys()]

    # Convert dates from %td format to %Y-%m-%d format.
    dtvars = [v for v, i in all_vars.items() if i["dtype"].startswith("datetime")]
    dtvars_to_convert = [v for v in dtvars if is_numeric_dtype(data[v])]
    converted_dates = data[dtvars_to_convert].apply(utils.convert_dates)
    data = data.drop(columns=dtvars_to_convert)
    data = pd.concat([data, converted_dates], axis=1)

    # Remove spurious dates.
    kwargs = dict(other=np.nan, inplace=True)
    data.where(data[dtvars].fillna("1900-01-01") <= "2100-01-01", **kwargs)
    data.where(data[dtvars].fillna("2100-01-01") >= "1900-01-01", **kwargs)

    # Update the data types of all variables.
    dtypes = {v: i["dtype"] for v, i in all_vars.items()}
    data = data.astype(dtypes)

    # -------------------------------------------------------------------------
    # -- Exclude rows with invalid drug information. --------------------------
    # -------------------------------------------------------------------------

    invalid_drug_info = pd.Series(False, index=data.index)

    data_prev = data.groupby("id").shift(1)
    prev_visitdate = data_prev.visitdate.fillna("1900-01-01")
    visit_order = data.groupby("id").cumcount() + 1

    # Identify rows with invalid combinations of `addX` and `discX`.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        invalid_drug_info |= ~(
            utils.is_unchanged(data, drug)
            | utils.add_at_visit(data, drug)
            | utils.stop_at_visit(data, drug)
            | utils.add_before_visit(data, drug)
            | utils.stop_before_visit(data, drug)
            | utils.add_before_stop_at_visit(data, drug)
            | utils.add_before_stop_before_visit(data, drug)
        )
    print(f"Invalid combinations of `addX` and `discX`: {invalid_drug_info.mean():.4f}")

    # Identify rows with invalid `pres_X`.
    # - When `discX=-1`, we should have `pres_X=0`.
    # - When `addX=1` and `discX=0`, we should have `pres_X=1`.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        invalid_drug_info |= (
            utils.stop_any(data, drug) 
            & data[drug_vars.pres].ne(0)
        )
        invalid_drug_info |= (
            (utils.add_at_visit(data, drug) | utils.add_before_visit(data, drug)) 
            & data[drug_vars.pres].ne(1)
        )
    print(f"Invalid `pres_X`: {invalid_drug_info.mean():.4f}")

    # Check that `initX` and `addX` are consistent.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        assert data.loc[data[drug_vars.init].eq(1), drug_vars.add].eq(1).all()
        assert data.loc[data[drug_vars.add].eq(0), drug_vars.init].eq(0).all()

    # Identify rows with invalid combinations of `hxX` and `initX`.
    # - When `presX=1` and `addX=0`, we should have `hxX=1`.
    # - When `addX=1` and `hxX=0`, we should have `initX=1`.
    # - When `btwaddX=1`, we should have `hxX=1`.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        invalid_drug_info |= (
            data[drug_vars.pres].eq(1)
            & data[drug_vars.add].eq(0)
            & data[drug_vars.hx].ne(1)
            & visit_order.gt(1)  # Ignore the first visit
        )
        invalid_drug_info |= (
            data[drug_vars.pres].eq(1)  # Redundant but kept for clarity
            & data[drug_vars.add].eq(1)
            & data[drug_vars.hx].ne(1)
            & data[drug_vars.init].ne(1)
            & visit_order.gt(1)  # Ignore the first visit
        )
        #invalid_drug_info |= (
        #    data[drug_vars.btwadd].eq(1)
        #    & data[drug_vars.hx].ne(1)
        #)
    print(f"Invalid combinations of `hxX` and `initX`: {invalid_drug_info.mean():.4f}")

    # Set `hxX` and `initX` equal to 1 in specific cases.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        data.loc[
            data[drug_vars.pres].eq(1)
            & data[drug_vars.add].eq(0)
            & data[drug_vars.hx].ne(1)
            & visit_order.eq(1),
            drug_vars.hx
        ] = 1.
        data.loc[
            data[drug_vars.pres].eq(1)
            & data[drug_vars.add].eq(1)
            & data[drug_vars.hx].ne(1)
            & data[drug_vars.init].ne(1)
            & visit_order.eq(1),
            drug_vars.init
        ] = 1.
        data.loc[
            data[drug_vars.btwadd].eq(1)
            & data[drug_vars.hx].ne(1),
            drug_vars.hx
        ] = 1.
    
    # Check that `X_adddt` and `X_discdt` are not missing for drugs added or
    # stopped at a registry visit.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        assert data.loc[utils.add_at_visit_any(data, drug), drug_vars.adddt].notna().all()
        assert data.loc[utils.stop_at_visit_any(data, drug), drug_vars.discdt].notna().all()

    # Identify rows with missing dates for drugs added or stopped prior to a
    # registry visit.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        invalid_drug_info |= (
            (utils.add_before_visit(data, drug) & data[drug_vars.adddt].isna())
            | (utils.stop_before_visit(data, drug) & data[drug_vars.discdt].isna())
            | (utils.add_before_stop_at_visit(data, drug) & data[drug_vars.adddt].isna())
            | (
                utils.add_before_stop_before_visit(data, drug)
                & (data[drug_vars.adddt].isna() | data[drug_vars.discdt].isna())
            )
        )
    print(f"Missing dates for drugs added or stopped prior to a registry visit: {invalid_drug_info.mean():.4f}")
    
    # Identify rows where the reported start (stop) date for any drug added
    # (stopped) prior to a registry visit is either before the previous visit date 
    # or after the current visit date. Also, identify rows where
    # `X_adddt>=discdt` when `btwaddX=1`.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        invalid_drug_info |= (
            data[drug_vars.vindadd].eq(2)  # Drug added prior to the visit
            & (
                (data[drug_vars.adddt] <= prev_visitdate)
                | (data[drug_vars.adddt] >= data.visitdate)
            )
        )
        invalid_drug_info |= (
            data[drug_vars.vinddisc].eq(2)  # Drug stopped prior to the visit
            & (
                (data[drug_vars.discdt] <= prev_visitdate)
                | (data[drug_vars.discdt] >= data.visitdate)
            )
        )
        invalid_drug_info |= (
            data[drug_vars.btwadd].eq(1)
            & data[drug_vars.adddt].ge(data[drug_vars.discdt])
        )
    print(f"Invalid dates for drugs added and/or stopped prior to a registry visit: {invalid_drug_info.mean():.4f}")

    # Identify rows where multiple drugs are added prior to a registry visit.
    c_btwadd = [f"btwadd{d}" for d in utils.get_drugs()]
    invalid_drug_info |= data[c_btwadd].sum(axis=1).gt(1)
    print(f"Multiple drugs added prior to a registry visit: {invalid_drug_info.mean():.4f}")

    # Identify unexplained interruptions in treatment.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        invalid_drug_info |= (
            data_prev[drug_vars.pres].eq(1)
            & data[drug_vars.pres].ne(1)  # 0 or NaN
            & data[drug_vars.disc].ne(-1)  # 0 or NaN
        )
    print(f"Unexplained interruptions in treatment: {invalid_drug_info.mean():.4f}")

    # Identify unexplained restarts in treatment.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        invalid_drug_info |= (
            data_prev[drug_vars.pres].eq(0)
            & data[drug_vars.pres].eq(1)
            & data[drug_vars.add].ne(1)  # 0 or NaN
        )
    print(f"Unexplained restarts of treatment: {invalid_drug_info.mean():.4f}")

    # Exclude rows with invalid drug information.
    data = utils.split_on_invalid(data, invalid_drug_info)

    # -------------------------------------------------------------------------
    # -- Update start and stop dates. -----------------------------------------
    # -------------------------------------------------------------------------

    # Replace `X_adddt` with `visitdate` when the drug is added at a registry visit.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        add_at_visit = utils.add_at_visit_any(data, drug)
        data.loc[add_at_visit, drug_vars.adddt] = data.loc[add_at_visit, "visitdate"]

    # Replace `X_discdt` with `visitdate` when the drug is stopped at a registry visit.
    for drug in utils.get_drugs():
        drug_vars = utils.get_drug_variables(drug)
        stop_at_visit = utils.stop_at_visit_any(data, drug)
        data.loc[stop_at_visit, drug_vars.discdt] = data.loc[stop_at_visit, "visitdate"]

    # -------------------------------------------------------------------------
    # -- Insert non-registry events. ------------------------------------------
    # -------------------------------------------------------------------------

    # Add columns `X_since` and `X_until` to indicate the duration of drug use.
    drug_periods = data.groupby("id", group_keys=False).progress_apply(
        utils.get_drug_periods,
    )
    data = pd.concat([data, drug_periods], axis=1)

    # Assign a column `visit_order` to enumerate the registry visits for each
    # patient.
    data["visit_order"] = data.groupby("id").cumcount() + 1.

    # Add a column `date` to record the dates of non-registry visits.
    data.insert(1, "date", data.visitdate)

    # Add a column `hx2X` to indicate whether a patient has a history of using 
    # drug X, including any initiations before the current registry visit. The 
    # original column `hxX` remains unchanged, as it defines the index visit.
    c_hx = [f"hx{d}" for d in utils.get_drugs()]
    c_hx2 = [f"hx2{d}" for d in utils.get_drugs()]
    data[c_hx2] = data[c_hx].copy()
    add_before_visit = []
    for drug in utils.get_drugs():
        add_before_visit += [utils.add_before_visit(data, drug)]
    add_before_visit = pd.concat(add_before_visit, axis=1).astype(float)
    data[c_hx2] = (data[c_hx] + add_before_visit.values).clip(upper=1)
    data[c_hx2] = data[c_hx2].astype("boolean")

    # Collect non-registry events.
    # 
    # Note 1: We do not include any non-registry events prior to the first
    # registry visit, as they often date far back in time.
    #
    # Note 2: We do not update the `initX` variable for non-registry events.
    # This means that all non-registry events have `initX=NaN`.
    add_prior = utils.extract_add_prior_to_visit(data)
    stop_prior = utils.extract_stop_prior_to_visit(data)
    btwadd = utils.extract_btwadd(data)
    data = pd.concat([data, add_prior, stop_prior, btwadd])
    data = data.reset_index()  # Insert index as column
    data = data.sort_values(by=["index", "date"])  # Sort rows
    data = data.set_index("index")  # Reset index

    # Aggregate non-registry events that occur on the same day.
    #
    # Note: We group the selected data by `index` and `date`, which means that
    # only events that occur on the same day are aggregated. Registry events
    # are left unchanged.
    has_date_duplicate = data.groupby("id").date.diff().dt.days.shift(-1).eq(0)
    has_date_duplicate.fillna(False, inplace=True)  # For cuDF compatibility
    assert data[has_date_duplicate].visitdate.isna().all()
    aggregate = data.loc[
        data[has_date_duplicate].index.unique()
    ].groupby(["index", "date"]).agg(utils.aggregate_events)
    aggregate = aggregate.reset_index(level="date")
    data = data.drop(data[has_date_duplicate].index.unique())
    data = pd.concat([data, aggregate])
    data = data.sort_values(by=["index", "date"])  # Sort rows

    # Add drug class variables.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        data = utils.assign_drug_class_variables(data)

    # Label therapies.
    drug_classes = ["dmard", "csdmard", "tnf", "nontnf", "jak", "bio", "biojak"]
    c_pres = [f"pres_{dc}" for dc in drug_classes]
    data[c_pres] = data[c_pres].replace(np.nan, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        data = utils.assign_therapy_labels(data)

    # Exclude registry visits where the therapy was classified as "Other therapy".
    #
    # Note: We also exclude all non-registry visits before a registry visit that
    # is excluded.
    indices_to_exclude = data.index[
        data.visitdate.notna()
        & data.therapy.eq("Other therapy")
    ]
    exclude = data.index.isin(indices_to_exclude)
    data = data.reset_index(drop=False)  # Insert index as column
    exclude = pd.Series(exclude, index=data.index)
    data = utils.split_on_invalid(data, exclude)
    data = data.set_index("index")  # Reset index
    data = data.sort_values(by=["index", "date"])  # Sort rows

    # Update the columns `X_since` and `X_until`.
    #
    # Note: We cannot group `data` and apply `utils.get_drug_periods` unless
    # `data` has unique index values.
    c_since = [f"{d}_since" for d in utils.get_drugs()]
    c_until = [f"{d}_until" for d in utils.get_drugs()]
    data = data.drop(columns=c_since + c_until)
    data_grouped = data.reset_index(drop=True).groupby("id", group_keys=False)
    drug_periods = data_grouped.progress_apply(utils.get_drug_periods)
    drug_periods = drug_periods.set_index(data.index)
    data = pd.concat([data, drug_periods], axis=1)

    # Update the data types of all variables.
    dtypes = {v: i["dtype"] for v, i in all_vars.items()}
    data = data.astype(dtypes)

    # -------------------------------------------------------------------------
    # -- Add features. --------------------------------------------------------
    # -------------------------------------------------------------------------

    # Ignore performance warnings.
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

    # Add the year of the visit.
    data["year"] = data.date.dt.year.rename("year").astype("float64")

    # Categorize reasons for drug changes.
    c_rsnchg = [c for c in data.columns if c.startswith("rsnchg")]
    c_rsnchg_cat = [f"{c}_cat" for c in c_rsnchg]
    data[c_rsnchg_cat] = data[c_rsnchg].map(
        utils.categorize_reason_code,
        na_action="ignore",
    )
    to_replace = {".": np.nan}
    data[c_rsnchg_cat] = data[c_rsnchg_cat].replace(to_replace).astype("category")
        
    # Categorize BMI.
    data["bmi_cat"] = pd.cut(
        data.bmi,
        bins=[0, 18.4, 24.9, 29.9, np.inf],
        labels=["underweight", "healthy weight", "overweight", "obesity"],
        include_lowest=True,
    )

    # Categorize CDAI.
    bins = [0, 2.8, 10, 22, np.inf]
    data["cdai_cat"] = pd.cut(
        data.cdai,
        bins=bins,
        labels=["remission", "low", "moderate", "high"],
        include_lowest=True,
    )

    # Categorize blood pressure.
    data["bp_cat"] = data.apply(utils.categorize_bp, axis=1).astype("category")
    
    # Add columns `rf_outcome`, `ccp_outcome`, and `tb_outcome` to record the 
    # results of RF, CCP antibody, and PPD tests.

    to_replace = {"False": "Negative", "True": "Positive"}
    rf_outcome = data.rfpos.astype("string").replace(to_replace)
    ccp_outcome = data.ccppos.astype("string").replace(to_replace)
    tb_outcome = data.ppd.astype("string").replace(to_replace)
    
    rf_outcome = rf_outcome.fillna("Not recorded").astype("category")
    ccp_outcome = ccp_outcome.fillna("Not recorded").astype("category")
    tb_outcome = tb_outcome.fillna("Not recorded").astype("category")

    data["rf_outcome"] = rf_outcome
    data["ccp_outcome"] = ccp_outcome
    data["tb_outcome"] = tb_outcome

    # Aggregate indicators for infections and comorbidities.

    def nonzero_sum(X, min_count=0):
        X = X.sum(axis=1, min_count=min_count)
        X = X.where(X.isna(), X > 0)
        return X.astype("boolean")
    
    def get_hxcomor_columns(c_comor):
        c_hxcomor = [c.replace("comor_", "hx") for c in c_comor]
        return [c for c in c_hxcomor if c in data.columns]

    infection = ["hospinf", "ivinf"]
    data["infection"] = nonzero_sum(data[infection])

    comor_metabolic = ["comor_hld", "comor_diabetes"]
    hxcomor_metabolic = get_hxcomor_columns(comor_metabolic)
    data["comor_metabolic"] = nonzero_sum(data[comor_metabolic])
    data["hxcomor_metabolic"] = nonzero_sum(data[hxcomor_metabolic])

    comor_cvd = [
        "comor_htn_hosp", "comor_htn", "comor_revasc", "comor_ven_arrhythm",
        "comor_mi", "comor_acs", "comor_unstab_ang", "comor_cor_art_dis",
        "comor_chf_hosp", "comor_chf_nohosp", "comor_stroke", "comor_tia",
        "comor_card_arrest", "comor_oth_clot", "comor_pulm_emb",
        "comor_pef_art_dis", "comor_pat_event", "comor_urg_par", "comor_pi",
        "comor_carotid", "comor_other_cv",
    ]
    hxcomor_cvd = get_hxcomor_columns(comor_cvd)
    data["comor_cvd"] = nonzero_sum(data[comor_cvd])
    data["hxcomor_cvd"] = nonzero_sum(data[hxcomor_cvd])

    comor_respiratory = ["comor_copd", "comor_asthma", "comor_fib"]
    hxcomor_respiratory = get_hxcomor_columns(comor_respiratory)
    data["comor_respiratory"] = nonzero_sum(data[comor_respiratory])
    data["hxcomor_respiratory"] = nonzero_sum(data[hxcomor_respiratory])

    comor_dil = ["comor_drug_ind_sle"]
    hxcomor_dil = get_hxcomor_columns(comor_dil)
    data["comor_dil"] = nonzero_sum(data[comor_dil])
    data["hxcomor_dil"] = nonzero_sum(data[hxcomor_dil])

    comor_cancer = [
        "comor_bc", "comor_lc", "comor_lymphoma", "comor_skin_cancer_squa",
        "comor_skin_cancer_mel", "comor_oth_cancer"
    ]
    hxcomor_cancer = get_hxcomor_columns(comor_cancer)
    data["comor_cancer"] = nonzero_sum(data[comor_cancer])
    data["hxcomor_cancer"] = nonzero_sum(data[hxcomor_cancer])

    comor_gi_liver = [
        "comor_ulcer", "comor_bowel_perf", "comor_hepatic_wbiop",
        "comor_hepatic_nobiop"
    ]
    hxcomor_gi_liver = get_hxcomor_columns(comor_gi_liver)
    data["comor_gi_liver"] = nonzero_sum(data[comor_gi_liver])
    data["hxcomor_gi_liver"] = nonzero_sum(data[hxcomor_gi_liver])

    comor_musculoskeletal = ["sec_sjog", "jt_deform"]
    to_replace = {"no": 0., "yes": 1., "new": 1.}
    data["comor_musculoskeletal"] = nonzero_sum(
        data[comor_musculoskeletal].replace(to_replace).astype(float)
    )

    comor_other = [
        "comor_psoriasis", "comor_depression", "comor_fm", "comor_oth_neuro",
        "comor_hemorg_hosp", "comor_hemorg_nohosp", "comor_oth_cond"
    ]
    hxcomor_other = get_hxcomor_columns(comor_other)
    data["comor_other"] = nonzero_sum(data[comor_other])
    data["hxcomor_other"] = nonzero_sum(data[hxcomor_other])

    return data
