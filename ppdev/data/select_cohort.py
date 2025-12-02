import warnings

import pandas as pd
from tqdm import tqdm

from ppdev.data import utils as utils

# Show progress bars.
tqdm.pandas()

# Define the columns that are critical for the cohort selection process.
CRITICAL_COLUMNS = [
    "id",
    "visitdate",
    "date",
    "hxbiojak",
    "initbiojak",
    "therapy",
    "cdai",
]
CRITICAL_COLUMNS += [f"btwadd{d}" for d in utils.get_drugs()]
CRITICAL_COLUMNS += [f"{d}_adddt" for d in utils.get_drugs()]


def select_cohort(
    data,
    max_days_since_biojakinit=180,
    max_days_until_biojakinit=30,
    exclude_other_therapies=True,
    require_recorded_outcome=True,
    require_regular_followup=True,
    followup_deltamin=30,
    followup_deltamax=270,
    require_fixed_followup=True,
    followup_total=1080,
):
    # Ignore performance warnings.
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    
    # Initialize the cohort data with the preprocessed data.
    cohort_data = data[CRITICAL_COLUMNS].copy()

    # Add a column `stage` to indicate the treatment stage. The index visit is 
    # labeled as stage 1, the first follow-up visit as stage 2, and so on. 
    # Visits before the index visit are assigned stage -1. The index visit is 
    # defined by `hxbiojak=0` and `initbiojak=1`.
    cohort_data = utils.assign_stage(cohort_data)

    # Add a column `stage2` which is a copy of `stage`. While `stage` is updated
    # during the cohort selection process, `stage2` remains unchanged.
    cohort_data["stage2"] = cohort_data.stage.copy()

    # Add a column `exclude_reason` to specify the reason for excluding a 
    # patient from the cohort.
    exclude_reason = pd.Series("", cohort_data.index, dtype=str, name="exclude_reason")
    cohort_data = pd.concat([cohort_data, exclude_reason], axis=1)

    # Exclude patients with history of b/tsDMARD use at registry enrollment.
    index_first_visit = data.groupby("id").head(1).index
    first_registry_visits = data[
        data.index.isin(index_first_visit)
        & data.visitdate.notna()
    ]
    assert first_registry_visits.visitdate.notna().all()
    eligible_ids = first_registry_visits[first_registry_visits.hxbiojak.eq(0)].id
    exclude = ~cohort_data.id.isin(eligible_ids)
    assert not cohort_data.loc[exclude, "stage"].eq(1).any()
    exclude_reason = "History of b/tsDMARDs at registry enrollment"
    cohort_data.loc[exclude, "exclude_reason"] = exclude_reason

    # Exclude patients who never start any b/tsDMARD therapy.
    num_biojakinits = cohort_data.groupby("id").initbiojak.sum()
    eligible_ids = num_biojakinits[num_biojakinits.ge(1)].index
    exclude = ~cohort_data.id.isin(eligible_ids)
    assert not cohort_data.loc[exclude, "stage"].eq(1).any()
    exclude_reason = "No b/tsDMARD initiated"
    cohort_data = utils.update_exclude_reason(
        cohort_data,
        exclude,
        exclude_reason,
        only_if_missing=True,
    )

    # If the first b/tsDMARD is initiated at a non-registry visit and 
    # discontinued before or at the subsequent registry index, `hxbiojak=1` 
    # when `initbiojak=1`. In this case, the patient will not have a valid 
    # index visit by our definition.

    c_biojak_btwadd = [f"btwadd{d}" for d in utils.get_drugs(exclude="csdmard")]
    assert cohort_data.loc[cohort_data.stage.eq(1), c_biojak_btwadd].max().max() == 0
    
    has_index_visit = cohort_data.groupby("id").stage.max().gt(0)
    eligible_ids = has_index_visit[has_index_visit].index
    exclude = ~cohort_data.id.isin(eligible_ids)
    assert not cohort_data.loc[exclude, "stage"].eq(1).any()
    exclude_reason = (
        "First b/tsDMARD initiated at a non-registry visit and discontinued "
        "before or at the subsequent registry visit"
    )
    cohort_data = utils.update_exclude_reason(
        cohort_data,
        exclude,
        exclude_reason,
        only_if_missing=True,
    )

    # Exclude patients whose first b/tsDMARD therapy begins too far from the 
    # index visit.

    c_biojak_adddt = [f"{d}_adddt" for d in utils.get_drugs(exclude="csdmard")]
    days_since_biojakinit = cohort_data.visitdate - cohort_data[c_biojak_adddt].min(axis=1)
    days_until_biojakinit = cohort_data[c_biojak_adddt].max(axis=1) - cohort_data.visitdate

    eligible_mask = (
        cohort_data.stage.eq(1)
        & days_since_biojakinit.dt.days.le(max_days_since_biojakinit)
        & days_until_biojakinit.dt.days.le(max_days_until_biojakinit)
    )
    eligible_mask.fillna(False, inplace=True)  # For cuDF compatibility
    eligible_ids = cohort_data.loc[eligible_mask, "id"]
    exclude = ~cohort_data.id.isin(eligible_ids)
    cohort_data.loc[exclude & cohort_data.stage.ne(-2), "stage"] = -1
    exclude_reason = (
        f"First b/tsDMARD initiated more than {max_days_since_biojakinit} days before "
        f"or more than {max_days_until_biojakinit} days after the registry visit"
    )
    cohort_data = utils.update_exclude_reason(
        cohort_data,
        exclude,
        exclude_reason,
        only_if_missing=True,
    )

    # Exclude all visits that follow a visit where the patient receives a 
    # therapy classified as "Other therapy".
    if exclude_other_therapies:
        exclude_reason = "Therapy classified as 'Other therapy'"
        cohort_data = cohort_data.groupby("id", group_keys=False).progress_apply(
            utils.require_valid_therapies,
            exclude_invalid=False,
            exclude_reason=exclude_reason,
        )
    
    # Exclude all visits that follow a visit where CDAI is not measured at the 
    # follow-up.
    if require_recorded_outcome:
        exclude_reason = "No CDAI recorded"
        cohort_data = cohort_data.groupby("id", group_keys=False).progress_apply(
            utils.require_recorded_outcome,
            exclude_invalid=False,
            exclude_reason=exclude_reason,
        )
    
    # Exclude all visits that follow a visit where the follow-up is either too
    # close or too far.
    if require_regular_followup:
        exclude_reason = (
            f"No follow-up within {followup_deltamin}--"
            f"{followup_deltamax} days"
        )
        cohort_data = cohort_data.groupby("id", group_keys=False).progress_apply(
            utils.require_regular_followup,
            delta_min=followup_deltamin,
            delta_max=followup_deltamax,
            exclude_invalid=False, 
            exclude_reason=exclude_reason,
        )
    
    # Require a fixed amount of follow-up.
    if require_fixed_followup:
        exclude_reason = f"Not {followup_total} days of follow-up"
        cohort_data = cohort_data.groupby("id", group_keys=False).progress_apply(
            utils.require_fixed_followup,
            followup=followup_total,
            exclude_invalid=False, 
            exclude_reason=exclude_reason,
        )

    # Exclude patients who have no registry follow-ups after the index visit.
    has_followup = utils.filter_cohort(cohort_data).groupby("id").size().gt(1)
    ineligible_ids = has_followup[~has_followup].index
    exclude = cohort_data.id.isin(ineligible_ids)
    cohort_data.loc[exclude & cohort_data.stage.ne(-2), "stage"] = -1
    exclude_reason = "No follow-up visits"
    cohort_data = utils.update_exclude_reason(
        cohort_data,
        exclude,
        exclude_reason,
        only_if_missing=True,
    )

    return cohort_data[["stage", "stage2", "exclude_reason"]]
