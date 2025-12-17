import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_initial_hangar_state(
    t2_path: str,
    hangar_width: float,
    hangar_length: float,
    t1_path: str = None,
    model_dims: dict = None,
):
    """
    Plot the initial state of the hangar using T2.

    Parameters
    ----------
    t2_path : str
        Path to T2.csv (must contain at least columns: 'c', 'M_ID', 'Init_X', 'Init_Y').
    hangar_width : float
        Hangar width (HW).
    hangar_length : float
        Hangar length (HL).
    t1_path : str, optional
        Path to T1.csv, where model sizes are stored.
        Expected columns: 'M_ID', 'Width', 'Length'.
    model_dims : dict, optional
        Mapping {M_ID: (Width, Length)}. If provided, this overrides t1_path.

    Notes
    -----
    - This function performs *no* checks (no overlap, no boundary checks, etc.).
    - Origin is assumed at (0, 0) in the bottom-left corner of the hangar.
    """

    # Read T2 (positions and model ids)
    t2 = pd.read_csv(t2_path)

    # If no external mapping is provided, but T1 is, build mapping from T1
    if model_dims is None and t1_path is not None:
        t1 = pd.read_csv(t1_path)

        required_cols = {"m", "W", "L"}
        if not required_cols.issubset(t1.columns):
            raise ValueError(
                f"T1 file must contain columns {required_cols}, "
                f"but has {set(t1.columns)}"
            )

        model_dims = {
            row["m"]: (row["W"], row["L"])
            for _, row in t1.iterrows()
        }

    # Check if T2 already has Width/Length (fallback option)
    has_size_cols_in_t2 = {"W", "L"}.issubset(t2.columns)

    # Default rectangle size if nothing else is available
    default_W, default_L = 10.0, 10.0

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw hangar boundary
    hangar = Rectangle((0, 0), hangar_width, hangar_length,
                       fill=False, linewidth=2)
    ax.add_patch(hangar)

    # Plot each aircraft
    for _, row in t2.iterrows():
        x = row["Init_X"]
        y = row["Init_Y"]
        m_id = row["M_ID"]

        # Priority:
        # 1) Width/Length in T2
        # 2) Mapping from T1 (model_dims)
        # 3) Default size
        if has_size_cols_in_t2:
            W = row["W"]
            L = row["L"]
        elif model_dims is not None and m_id in model_dims:
            W, L = model_dims[m_id]
        else:
            W, L = default_W, default_L

        rect = Rectangle((x, y), W, L, alpha=0.4)
        ax.add_patch(rect)

        # Label with aircraft ID (column 'c')
        ac_id = row["c"]
        ax.text(x + W / 2, y + L / 2, ac_id,
                ha="center", va="center", fontsize=8)

    ax.set_xlim(0, hangar_width)
    ax.set_ylim(0, hangar_length)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5)

    plt.title("Initial Hangar State (from T2 and T1)")
    plt.tight_layout()
    plt.show()


plot_initial_hangar_state(
    t2_path="Sensitivity/Sensitivity_Scenario/T2.csv",
    t1_path="Sensitivity/Sensitivity_Scenario/T1.csv",
    hangar_width=150.0,
    hangar_length=100.0,
)
