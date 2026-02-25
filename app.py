import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# --- PAGE CONFIG ---
st.set_page_config(page_title="LP Solver", layout="wide")

# --- UI STYLING ---
st.markdown("""
    <style>
    a.header-anchor { display: none; }
    [data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 800; color: #00d4ff; }
    
    /* Force tables to be scrollable on mobile */
    .stTable { overflow-x: auto !important; display: block; }
    </style>
""", unsafe_allow_html=True)

st.title("Linear Programming Solver")
st.caption("Quantitative Methods - Simplex Method Result")
st.divider()

# --- 1. OBJECTIVE SECTION ---
st.subheader("Objective Function")
obj_cols = st.columns([1.5, 2, 2])
with obj_cols[0]:
    mode = st.selectbox("Goal", ["Maximize", "Minimize"])
with obj_cols[1]:
    c1 = st.number_input("Coefficient of X", value=3000.0)
with obj_cols[2]:
    c2 = st.number_input("Coefficient of Y", value=5000.0)

# --- 2. CONSTRAINTS SECTION ---
st.subheader("Constraints")
num_constraints = st.number_input("Total Constraints", 1, 10, 4)

constraints_data = []

# Header Row (Visible on PC, helpful for context)
h_cols = st.columns([2, 2, 1.5, 2])
h_cols[0].caption("X Coefficient")
h_cols[1].caption("Y Coefficient")
h_cols[2].caption("Relation")
h_cols[3].caption("RHS / Limit")

for i in range(num_constraints):
    # These columns will stay horizontal on PC but stack on narrow mobile screens
    cols = st.columns([2, 2, 1.5, 2])
    
    val_x = cols[0].number_input(f"x{i}", label_visibility="collapsed", value=10.0, key=f"x{i}")
    val_y = cols[1].number_input(f"y{i}", label_visibility="collapsed", value=10.0, key=f"y{i}")
    rel = cols[2].selectbox(f"r{i}", ["<=", ">=", "="], label_visibility="collapsed", key=f"rel{i}")
    rhs = cols[3].number_input(f"b{i}", label_visibility="collapsed", value=100.0, key=f"rhs{i}")
    
    constraints_data.append({"x": val_x, "y": val_y, "rel": rel, "rhs": rhs})

st.write("")
col_solve, col_reset = st.columns([4, 1])
with col_solve:
    solve_btn = st.button("Calculate Optimal Solution", type="primary", use_container_width=True)
with col_reset:
    if st.button("Reset", use_container_width=True):
        st.rerun()

st.divider()

# --- 3. RESULTS ---
if solve_btn:
    c = np.array([c1, c2])
    c_scipy = -c if mode == "Maximize" else c
    
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for con in constraints_data:
        row = [con["x"], con["y"]]
        if con["rel"] == "<=":
            A_ub.append(row); b_ub.append(con["rhs"])
        elif con["rel"] == ">=":
            A_ub.append([-v for v in row]); b_ub.append(-con["rhs"])
        else:
            A_eq.append(row); b_eq.append(con["rhs"])

    iteration_logs = []
    def callback(res):
        iteration_logs.append({"Z": res.fun, "X": res.x[0], "Y": res.x[1]})

    res = linprog(c_scipy, A_ub=np.array(A_ub) if A_ub else None, 
                  b_ub=np.array(b_ub) if b_ub else None, 
                  A_eq=np.array(A_eq) if A_eq else None, 
                  b_eq=np.array(b_eq) if b_eq else None, 
                  method='simplex', callback=callback)

    if res.success:
        final_z = -res.fun if mode == "Maximize" else res.fun
        
        # Results metrics
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Optimal {mode[:3]}. Z", f"{final_z:,.2f}")
        m2.metric("Final X", f"{res.x[0]:.3f}")
        m3.metric("Final Y", f"{res.x[1]:.3f}")

        st.write("")
        with st.expander("Show Iteration History Details", expanded=True):
            st.info("""
            **Technical Note:** The iteration path is determined by the solver's pivot selection rules. 
            While this path can differ from manual textbook methods (like Big-M), the solver will 
            always arrive at the same correct optimal result.
            """)
            
            iter_df = pd.DataFrame(iteration_logs)
            if mode == "Maximize": iter_df["Z"] = -iter_df["Z"]
            iter_df.columns = ["Z", "X", "Y"]
            iter_df = iter_df.drop_duplicates().reset_index(drop=True)
            iter_df.index.name = "Step"

            def bold_last_row(row):
                return ['font-weight: bold' if row.name == len(iter_df) - 1 else '' for _ in row]

            st.table(iter_df.style.apply(bold_last_row, axis=1).format("{:.3f}"))
    else:
        st.error(f"Solver Error: {res.message}")
