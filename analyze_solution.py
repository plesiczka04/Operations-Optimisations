import csv
import os
from typing import Dict, Any, Optional

# Must match epsilon_p used in the model / main.py
EPSILON_P_DEFAULT = 0.001


def analyze_solution_csv(
    path: str,
    epsilon_p: float = EPSILON_P_DEFAULT,
    run_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a single solution CSV produced by model.solve_and_report.

    Computes:
    - #total aircraft, #current aircraft, #requests
    - #accepted requests, #rejected requests, acceptance rate
    - total arrival delay (requests), total departure delay (all)
    - objective decomposition (rejection, arrival, departure, epsilon_p term)
    - reconstructed total objective

    The classification "request (future)" vs "current" follows the same rule as
    your sensitivity script:
        is_future = (Penalty_Reject > 0) or (Penalty_ArrivalDelay > 0)

    Parameters
    ----------
    path : str
        Path to a single solution CSV.
    epsilon_p : float
        Epsilon positioning penalty (must match the one used in optimization).
    run_tag : Optional[str]
        Optional label to carry into the output (e.g., "baseline", "case_A").

    Returns
    -------
    Dict[str, Any]
        Dictionary with aggregated metrics.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Solution CSV not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in solution file {path!r}.")

    n_total = len(rows)
    n_current = 0
    n_requests = 0
    n_accepted_requests = 0
    n_rejected_requests = 0

    total_D_Arr = 0.0
    total_D_Dep = 0.0

    obj_rej = 0.0
    obj_arr = 0.0
    obj_dep = 0.0
    obj_eps = 0.0

    for row in rows:
        accepted = float(row["Accepted"])
        accepted = 1.0 if accepted >= 0.5 else 0.0  # robust 0/1

        p_rej = float(row["Penalty_Reject"])
        p_arr = float(row["Penalty_ArrivalDelay"])
        p_dep = float(row["Penalty_DepartureDelay"])

        d_arr = float(row["D_Arr"])
        d_dep = float(row["D_Dep"])

        x = float(row["X"])
        y = float(row["Y"])

        is_future = (p_rej > 0.0) or (p_arr > 0.0)

        if is_future:
            n_requests += 1
            if accepted > 0.5:
                n_accepted_requests += 1
            else:
                n_rejected_requests += 1

            # Arrival delay is meaningful for request aircraft in your definition
            total_D_Arr += d_arr

            # Objective components for request aircraft
            obj_rej += p_rej * (1.0 - accepted)
            obj_arr += p_arr * d_arr
            obj_eps += epsilon_p * (x + y)
        else:
            n_current += 1

        # Departure delay cost applies (in your original script) to all aircraft
        total_D_Dep += d_dep
        obj_dep += p_dep * d_dep

    acceptance_rate = 1.0 if n_requests == 0 else (n_accepted_requests / n_requests)
    objective = obj_rej + obj_arr + obj_dep + obj_eps

    return {
        "tag": run_tag if run_tag is not None else os.path.splitext(os.path.basename(path))[0],
        "file": os.path.basename(path),
        "n_total": n_total,
        "n_current": n_current,
        "n_requests": n_requests,
        "n_accepted": n_accepted_requests,
        "n_rejected": n_rejected_requests,
        "acceptance_rate": acceptance_rate,
        "total_D_Arr": total_D_Arr,
        "total_D_Dep": total_D_Dep,
        "obj_rej": obj_rej,
        "obj_arr": obj_arr,
        "obj_dep": obj_dep,
        "obj_eps": obj_eps,
        "objective": objective,
    }
