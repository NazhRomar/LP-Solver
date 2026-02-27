import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# --- PAGE CONFIG ---
st.set_page_config(page_title="LP Solver", layout="wide")

# --- INITIALIZE RESET COUNTER ---
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

def reset_state():
    """Increments the counter to force Streamlit to render brand new widgets."""
    st.session_state.reset_counter += 1

# --- UI STYLING ---
st.markdown("""
    <style>
    a.header-anchor { display: none; }
    [data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 800; color: #00d4ff; }
    
    /* MOBILE STACKING FIX */
    @media (max-width: 700px) {
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 15px;
        }
        [data-testid="stWidgetLabel"] p {
            display: block !important;
        }
    }

    .stTable { overflow-x: auto !important; display: block; }
    </style>
""", unsafe_allow_html=True)

st.title("Linear Programming Solver")
st.caption("Quantitative Methods - Simplex Method Result")
st.divider()

# --- 1. OBJECTIVE SECTION ---
st.subheader("Objective Function")
obj_cols = st.columns([1.5, 2, 2])

# Keys include the reset_counter to allow for a true wipe
with obj_cols[0]:
    mode = st.selectbox("Goal", ["Maximize", "Minimize"], key=f"goal_{st.session_state.reset_counter}")
with obj_cols[1]:
    c1 = st.number_input("Coefficient of X", value=0.0, key=f"obj_x_{st.session_state.reset_counter}")
with obj_cols[2]:
    c2 = st.number_input("Coefficient of Y", value=0.0, key=f"obj_y_{st.session_state.reset_counter}")

# --- 2. CONSTRAINTS SECTION ---
st.subheader("Constraints")
num_constraints = st.number_input("Total Constraints", 1, 10, 4, key=f"num_con_{st.session_state.reset_counter}")

constraints_data = []

# Desktop Headers
h_cols = st.columns([2, 2, 1.5, 2])
h_cols[0].caption("X Coefficient")
h_cols[1].caption("Y Coefficient")
h_cols[2].caption("Relation")
h_cols[3].caption("RHS / Limit")

for i in range(num_constraints):
    cols = st.columns([2, 2, 1.5, 2])
    
    val_x = cols[0].number_input(f"X Coeff {i+1}", label_visibility="collapsed", value=0.0, key=f"x{i}_{st.session_state.reset_counter}")
    val_y = cols[1].number_input(f"Y Coeff {i+1}", label_visibility="collapsed", value=0.0, key=f"y{i}_{st.session_state.reset_counter}")
    rel = cols[2].selectbox(f"Rel {i+1}", ["<=", ">=", "="], label_visibility="collapsed", key=f"rel{i}_{st.session_state.reset_counter}")
    rhs = cols[3].number_input(f"Limit {i+1}", label_visibility="collapsed", value=0.0, key=f"rhs{i}_{st.session_state.reset_counter}")
    
    constraints_data.append({"x": val_x, "y": val_y, "rel": rel, "rhs": rhs})

st.write("")
col_solve, col_reset = st.columns([4, 1])

with col_solve:
    solve_btn = st.button("Calculate Optimal Solution", type="primary", use_container_width=True)

with col_reset:
    st.button("Reset All Fields", use_container_width=True, on_click=reset_state)

st.divider()

# --- 3. RESULTS ---
if solve_btn:
    # 1. Check if the Objective Function is empty
    if c1 == 0.0 and c2 == 0.0:
        st.error("Validation Error: Objective Function coefficients cannot both be zero.")
    else:
        c = np.array([c1, c2])
        c_scipy = -c if mode == "Maximize" else c
        
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        valid_constraints = 0
        
        # 2. Filter out empty constraints to prevent matrix crashes
        for con in constraints_data:
            if con["x"] == 0.0 and con["y"] == 0.0:
                continue
            
            valid_constraints += 1
            row = [con["x"], con["y"]]
            
            if con["rel"] == "<=":
                A_ub.append(row); b_ub.append(con["rhs"])
            elif con["rel"] == ">=":
                A_ub.append([-v for v in row]); b_ub.append(-con["rhs"])
            else:
                A_eq.append(row); b_eq.append(con["rhs"])

        # 3. Ensure at least one constraint has data
        if valid_constraints == 0:
            st.error("Validation Error: Please enter at least one valid constraint.")
        else:
            iteration_logs = []
            def callback(res):
                iteration_logs.append({"Z": res.fun, "X": res.x[0], "Y": res.x[1]})

            # 4. Try running the math solver
            try:
                res = linprog(c_scipy, A_ub=np.array(A_ub) if len(A_ub) > 0 else None, 
                              b_ub=np.array(b_ub) if len(b_ub) > 0 else None, 
                              A_eq=np.array(A_eq) if len(A_eq) > 0 else None, 
                              b_eq=np.array(b_eq) if len(b_eq) > 0 else None, 
                              method='simplex', callback=callback)

                if res.success:
                    final_z = -res.fun if mode == "Maximize" else res.fun
                    # Fix the negative zero floating-point quirk
                    if final_z == 0.0:
                        final_z = 0.0
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric(f"Optimal {mode[:3]}. Z", f"{final_z:,.2f}")
                    m2.metric("Final X Value", f"{res.x[0]:.3f}")
                    m3.metric("Final Y Value", f"{res.x[1]:.3f}")

                    st.write("")
                    with st.expander("Show Iteration History Details", expanded=True):
                        st.info("""
                        **Technical Note:** The iteration path is determined by the solver's pivot selection rules, which prioritize numerical stability. 
                        While this path can differ from manual textbook methods (like Big-M or Two-Phase) due to different tie-breaking 
                        logic, the solver will always arrive at the same correct optimal result.
                        """)
                        
                        iter_df = pd.DataFrame(iteration_logs)
                        
                        # 5. Handle the 0-Step edge case gracefully
                        if not iter_df.empty:
                            if mode == "Maximize": iter_df["Z"] = -iter_df["Z"]
                            iter_df.columns = ["Objective Value (Z)", "X Position", "Y Position"]
                            iter_df = iter_df.drop_duplicates().reset_index(drop=True)
                            iter_df.index.name = "Step"

                            def bold_last_row(row):
                                return ['font-weight: bold' if row.name == len(iter_df) - 1 else '' for _ in row]

                            st.table(iter_df.style.apply(bold_last_row, axis=1).format("{:.3f}"))
                        else:
                            st.success("The algorithm found the optimal solution immediately (0 iterations required).")
                            instant_df = pd.DataFrame([{
                                "Objective Value (Z)": final_z, 
                                "X Position": res.x[0], 
                                "Y Position": res.x[1]
                            }])
                            instant_df.index.name = "Step"
                            st.table(instant_df.style.format("{:.3f}"))
                else:
                    st.error(f"Solver Error: {res.message}")
            except Exception as e:
                st.error(f"Mathematical Error: {str(e)}. Please check your inputs.")
