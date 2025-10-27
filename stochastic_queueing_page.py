"""
Stochastic Processes & Queueing Theory Analysis
Add this as a new page to your existing Streamlit DOE application
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
import itertools


# ========================================
# HELPER FUNCTIONS
# ========================================

def generate_queue_template():
    """Generate a template CSV for queueing data upload"""
    template = pd.DataFrame({
        'arrival_time': [0.0, 1.5, 3.2, 4.8, 6.1, 8.3, 10.2, 12.5],
        'service_time': [2.1, 1.8, 2.5, 1.2, 2.8, 1.5, 2.0, 1.9],
        'customer_id': [1, 2, 3, 4, 5, 6, 7, 8]
    })
    return template


@st.cache_data
def analyze_uploaded_queue_data(df):
    """Analyze uploaded queueing data (long format)"""
    df = df.sort_values('arrival_time').reset_index(drop=True)
    
    # Calculate interarrival times
    df['interarrival_time'] = df['arrival_time'].diff()
    
    # Calculate waiting times and system times
    df['service_start'] = 0.0
    df['service_end'] = 0.0
    df['wait_time'] = 0.0
    df['system_time'] = 0.0
    
    for i in range(len(df)):
        if i == 0:
            df.loc[i, 'service_start'] = df.loc[i, 'arrival_time']
        else:
            df.loc[i, 'service_start'] = max(df.loc[i, 'arrival_time'], 
                                              df.loc[i-1, 'service_end'])
        
        df.loc[i, 'wait_time'] = df.loc[i, 'service_start'] - df.loc[i, 'arrival_time']
        df.loc[i, 'service_end'] = df.loc[i, 'service_start'] + df.loc[i, 'service_time']
        df.loc[i, 'system_time'] = df.loc[i, 'service_end'] - df.loc[i, 'arrival_time']
    
    # Calculate metrics
    lambda_hat = 1 / df['interarrival_time'].mean()  # Arrival rate
    mu_hat = 1 / df['service_time'].mean()  # Service rate
    rho = lambda_hat / mu_hat  # Utilization
    
    metrics = {
        'lambda': lambda_hat,
        'mu': mu_hat,
        'rho': rho,
        'avg_wait': df['wait_time'].mean(),
        'avg_service': df['service_time'].mean(),
        'avg_system': df['system_time'].mean(),
        'max_wait': df['wait_time'].max(),
        'max_system': df['system_time'].max(),
        'n_customers': len(df)
    }
    
    return df, metrics


def mm1_theoretical(lambda_rate, mu_rate):
    """Calculate M/M/1 queue theoretical metrics"""
    rho = lambda_rate / mu_rate
    
    if rho >= 1:
        return {
            'system': 'UNSTABLE',
            'rho': rho,
            'L': np.inf,
            'Lq': np.inf,
            'W': np.inf,
            'Wq': np.inf,
            'P0': 0
        }
    
    L = rho / (1 - rho)  # Average number in system
    Lq = (rho**2) / (1 - rho)  # Average number in queue
    W = 1 / (mu_rate - lambda_rate)  # Average time in system
    Wq = rho / (mu_rate - lambda_rate)  # Average waiting time
    P0 = 1 - rho  # Probability system is empty
    
    return {
        'system': 'M/M/1',
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'P0': P0
    }


def mmc_theoretical(lambda_rate, mu_rate, c):
    """Calculate M/M/c queue theoretical metrics"""
    rho = lambda_rate / (c * mu_rate)
    
    if rho >= 1:
        return {
            'system': 'UNSTABLE',
            'c': c,
            'rho': rho,
            'L': np.inf,
            'Lq': np.inf,
            'W': np.inf,
            'Wq': np.inf,
            'P0': 0
        }
    
    # Calculate P0 (Erlang C formula)
    sum_term = sum([(lambda_rate/mu_rate)**n / np.math.factorial(n) for n in range(c)])
    last_term = ((lambda_rate/mu_rate)**c / np.math.factorial(c)) * (1 / (1 - rho))
    P0 = 1 / (sum_term + last_term)
    
    # Probability of waiting (Erlang C)
    C = (((lambda_rate/mu_rate)**c / np.math.factorial(c)) * (1 / (1 - rho))) * P0
    
    Lq = C * rho / (1 - rho)
    L = Lq + lambda_rate / mu_rate
    Wq = Lq / lambda_rate
    W = Wq + 1 / mu_rate
    
    return {
        'system': f'M/M/{c}',
        'c': c,
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'P0': P0,
        'C': C
    }


@st.cache_data
def simulate_queue(lambda_rate, mu_rate, n_servers=1, n_customers=1000, seed=None):
    """Monte Carlo simulation of M/M/c queue"""
    rng = np.random.default_rng(seed)
    
    # Generate arrivals (Poisson process)
    interarrival_times = rng.exponential(1/lambda_rate, n_customers)
    arrival_times = np.cumsum(interarrival_times)
    
    # Generate service times (Exponential)
    service_times = rng.exponential(1/mu_rate, n_customers)
    
    # Simulate queue
    service_start = np.zeros(n_customers)
    service_end = np.zeros(n_customers)
    server_busy_until = np.zeros(n_servers)
    
    for i in range(n_customers):
        # Find earliest available server
        earliest_server = np.argmin(server_busy_until)
        earliest_available = server_busy_until[earliest_server]
        
        # Customer starts service when they arrive OR when server is free
        service_start[i] = max(arrival_times[i], earliest_available)
        service_end[i] = service_start[i] + service_times[i]
        
        # Update server availability
        server_busy_until[earliest_server] = service_end[i]
    
    wait_times = service_start - arrival_times
    system_times = service_end - arrival_times
    
    # Calculate queue length over time (snapshot method)
    max_time = service_end[-1]
    time_points = np.linspace(0, max_time, 10000)
    queue_length = np.zeros(len(time_points))
    
    for i, t in enumerate(time_points):
        in_system = np.sum((arrival_times <= t) & (service_end > t))
        queue_length[i] = in_system
    
    results = {
        'arrival_times': arrival_times,
        'service_times': service_times,
        'wait_times': wait_times,
        'system_times': system_times,
        'service_start': service_start,
        'service_end': service_end,
        'time_points': time_points,
        'queue_length': queue_length,
        'avg_wait': np.mean(wait_times),
        'avg_system': np.mean(system_times),
        'avg_queue_length': np.mean(queue_length),
        'max_wait': np.max(wait_times),
        'max_system': np.max(system_times),
        'utilization': np.sum(service_times) / (n_servers * max_time)
    }
    
    return results


@st.cache_data
def compare_queue_designs(lambda_rate, mu_rate, max_servers=5, n_customers=1000, seed=None):
    """Compare different queue configurations"""
    results = []
    
    for c in range(1, max_servers + 1):
        # Theoretical
        theoretical = mmc_theoretical(lambda_rate, mu_rate, c)
        
        # Simulation
        sim = simulate_queue(lambda_rate, mu_rate, n_servers=c, 
                            n_customers=n_customers, seed=seed)
        
        results.append({
            'servers': c,
            'rho_theoretical': theoretical['rho'],
            'W_theoretical': theoretical['W'] if theoretical['W'] != np.inf else None,
            'Wq_theoretical': theoretical['Wq'] if theoretical['Wq'] != np.inf else None,
            'L_theoretical': theoretical['L'] if theoretical['L'] != np.inf else None,
            'W_simulated': sim['avg_system'],
            'Wq_simulated': sim['avg_wait'],
            'L_simulated': sim['avg_queue_length'],
            'utilization_simulated': sim['utilization'],
            'stable': theoretical['rho'] < 1
        })
    
    return pd.DataFrame(results)


# ========================================
# MARKOV CHAIN FUNCTIONS
# ========================================

@st.cache_data
def analyze_markov_chain(P, n_steps=10, initial_state=None):
    """Analyze a discrete-time Markov chain"""
    n_states = P.shape[0]
    
    # Validate transition matrix
    if not np.allclose(P.sum(axis=1), 1):
        st.error("Transition matrix rows must sum to 1")
        return None
    
    # Initial distribution
    if initial_state is None:
        pi_0 = np.ones(n_states) / n_states
    else:
        pi_0 = np.zeros(n_states)
        pi_0[initial_state] = 1
    
    # Calculate state distributions over time
    distributions = [pi_0]
    for _ in range(n_steps):
        distributions.append(distributions[-1] @ P)
    
    # Find steady-state (if exists)
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    steady_state_idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
    steady_state = np.real(eigenvectors[:, steady_state_idx])
    steady_state = steady_state / steady_state.sum()
    
    return {
        'P': P,
        'distributions': np.array(distributions),
        'steady_state': steady_state,
        'n_steps': n_steps
    }


# ========================================
# VISUALIZATION FUNCTIONS
# ========================================

def plot_queue_timeline(df):
    """Plot Gantt-style timeline of queue activities"""
    fig = go.Figure()
    
    for i, row in df.iterrows():
        customer = row['customer_id']
        
        # Waiting time (red)
        if row['wait_time'] > 0:
            fig.add_trace(go.Scatter(
                x=[row['arrival_time'], row['service_start']],
                y=[customer, customer],
                mode='lines',
                line=dict(color='red', width=8),
                name='Waiting' if i == 0 else '',
                showlegend=(i == 0),
                legendgroup='wait'
            ))
        
        # Service time (green)
        fig.add_trace(go.Scatter(
            x=[row['service_start'], row['service_end']],
            y=[customer, customer],
            mode='lines',
            line=dict(color='green', width=8),
            name='In Service' if i == 0 else '',
            showlegend=(i == 0),
            legendgroup='service'
        ))
    
    fig.update_layout(
        title='Queue Timeline (Gantt Chart)',
        xaxis_title='Time',
        yaxis_title='Customer ID',
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_queue_length_over_time(time_points, queue_length):
    """Plot queue length evolution"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=queue_length,
        mode='lines',
        line=dict(color='blue'),
        name='Queue Length',
        fill='tozeroy'
    ))
    
    fig.add_hline(y=np.mean(queue_length), 
                  line_dash="dash", 
                  line_color="red",
                  annotation_text=f"Mean: {np.mean(queue_length):.2f}")
    
    fig.update_layout(
        title='Number of Customers in System Over Time',
        xaxis_title='Time',
        yaxis_title='Customers in System',
        height=450
    )
    
    return fig


def plot_comparison_table(comparison_df):
    """Create comparison visualization for multiple server configs"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Wait Time', 'Average System Time', 
                       'Average Queue Length', 'Server Utilization'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    stable = comparison_df['stable']
    
    # Wait Time
    fig.add_trace(go.Bar(
        x=comparison_df['servers'], 
        y=comparison_df['Wq_simulated'],
        name='Simulated',
        marker_color=['green' if s else 'red' for s in stable]
    ), row=1, col=1)
    
    # System Time
    fig.add_trace(go.Bar(
        x=comparison_df['servers'], 
        y=comparison_df['W_simulated'],
        name='Simulated',
        marker_color=['green' if s else 'red' for s in stable],
        showlegend=False
    ), row=1, col=2)
    
    # Queue Length
    fig.add_trace(go.Bar(
        x=comparison_df['servers'], 
        y=comparison_df['L_simulated'],
        name='Simulated',
        marker_color=['green' if s else 'red' for s in stable],
        showlegend=False
    ), row=2, col=1)
    
    # Utilization
    fig.add_trace(go.Bar(
        x=comparison_df['servers'], 
        y=comparison_df['utilization_simulated'],
        name='Simulated',
        marker_color=['green' if s else 'red' for s in stable],
        showlegend=False
    ), row=2, col=2)
    
    fig.update_xaxes(title_text="Number of Servers", row=1, col=1)
    fig.update_xaxes(title_text="Number of Servers", row=1, col=2)
    fig.update_xaxes(title_text="Number of Servers", row=2, col=1)
    fig.update_xaxes(title_text="Number of Servers", row=2, col=2)
    
    fig.update_yaxes(title_text="Wq", row=1, col=1)
    fig.update_yaxes(title_text="W", row=1, col=2)
    fig.update_yaxes(title_text="L", row=2, col=1)
    fig.update_yaxes(title_text="ρ", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True)
    
    return fig


def plot_markov_evolution(result):
    """Plot state probability evolution for Markov chain"""
    distributions = result['distributions']
    n_states = distributions.shape[1]
    
    fig = go.Figure()
    
    for state in range(n_states):
        fig.add_trace(go.Scatter(
            x=np.arange(len(distributions)),
            y=distributions[:, state],
            mode='lines+markers',
            name=f'State {state}'
        ))
    
    fig.update_layout(
        title='State Probability Evolution',
        xaxis_title='Time Step',
        yaxis_title='Probability',
        height=450
    )
    
    return fig


# ========================================
# NEW PAGE: GUIDE & GLOSSARY
# ========================================

def guide_and_glossary_page():
    """A new page to guide users and define terms."""
    st.title("Guide & Glossary")
    st.markdown("By **Leonardo H. Talero-Sarmiento**")
    
    st.markdown("""
    Welcome to the Stochastic Processes & Queueing Theory analyzer. This tool is
    designed to help you understand and optimize systems where "customers"
    (people, data packets, work items) arrive and wait for "service".
    """)
    
    st.header("How to Use This Tool")
    
    st.subheader("1. Guide & Glossary (This Page)")
    st.markdown("""
    Start here! This page explains the key concepts and defines all the
    acronyms and technical terms used in the other modules.
    """)
    
    st.subheader("2. Upload Queue Data")
    st.markdown("""
    Use this mode if you have **your own historical data** of arrivals and services.
    
    - **What it does:** Calculates the key performance metrics from your data
      (like average wait time, server utilization, etc.).
    - **How to use:**
        1.  Download the template CSV to see the required format.
        2.  Prepare your data (must have `arrival_time` and `service_time`).
        3.  Upload your CSV file.
        4.  The app will instantly analyze your data, show descriptive statistics,
            and compare it to the theoretical M/M/1 model.
    """)
    
    st.subheader("3. Simulate Queue Systems")
    st.markdown("""
    Use this mode to **model and test hypothetical queue systems**. This is
    perfect for "what-if" analysis (e.g., "What happens if I add another server?").
    
    - **What it does:** Simulates thousands of customer arrivals based on parameters
      you set (arrival rate, service rate) and compares different designs
      (e.g., 1 server vs. 2 servers vs. 3 servers).
    - **How to use:**
        1.  Go to the sidebar and set your **System Parameters**:
            - **Arrival Rate (λ):** How many customers arrive per unit of time?
            - **Service Rate (μ):** How many customers can one server handle per
              unit of time?
        2.  Click the "Run Simulation & Comparison" button.
        3.  Analyze the results to find the best balance of wait time and
            server utilization.
    """)

    st.subheader("4. Markov Chain Analysis")
    st.markdown("""
    Use this mode for **probabilistic state-based systems**. This is useful for
    modeling things like weather patterns, machine reliability (Working vs. Broken),
    or customer loyalty (Brand A vs. Brand B).
    
    - **What it does:** Calculates the long-term probabilities (steady-state)
      of a system given its transition rules.
    - **How to use:**
        1.  In the sidebar, select the **Number of States** (e.g., 2 states:
            "Working" and "Broken").
        2.  Enter the **Transition Matrix (P)**. Each cell `P(i, j)` is the
            probability of moving *from* state `i` *to* state `j` in one time step.
            **Each row must sum to 1.0!**
        3.  Click "Analyze Markov Chain".
        4.  The tool will show you the long-run probability of being in any
            given state (the "Steady-State Distribution").
    """)
    
    st.subheader("5. Economic Analysis")
    st.markdown("""
    Use this mode to **find the most cost-effective queue design**.
    
    - **What it does:** Balances the **cost of servers** (e.g., salaries)
      against the **cost of customer waiting** (e.g., lost business,
      customer dissatisfaction).
    - **How to use:**
        1.  Set the system parameters (λ and μ) in the sidebar.
        2.  Enter your **Cost Parameters**:
            - **Cost per Server:** How much does it cost to operate one server
              (per hour)?
            - **Customer Wait Cost:** How much does it cost your business for
              a customer to wait (per hour)?
        3.  Click "Calculate Optimal Configuration".
        4.  The tool will output a table showing the total cost for each design
            and recommend the one with the **lowest total cost**.
    """)
    
    st.header("Glossary of Terms & Acronyms")
    
    with st.expander("Expand to see all definitions..."):
        st.subheader("Queueing Theory (Kendall's Notation: A/B/c)")
        st.markdown("""
        Queueing theory is described using **Kendall's Notation (A/B/c)**, which
        defines the system's properties.
        - **A (Arrival Process):** Describes how customers arrive.
            - **M:** Markovian or Poisson process. This means arrivals are random
              and independent. The time *between* arrivals follows an
              **exponential distribution**.
        - **B (Service Process):** Describes how long service takes.
            - **M:** Markovian. Service times are random and independent,
              following an **exponential distribution**.
        - **c (Servers):** The number of parallel servers.
        
        **M/M/1:** A system with Poisson arrivals, exponential service times,
        and **one server**.
        
        **M/M/c:** A system with Poisson arrivals, exponential service times,
        and **'c' parallel servers**.
        """)
        
        st.subheader("Core Metrics (The Greeks)")
        st.markdown(r"""
        - **λ (Lambda): Arrival Rate**
            - **Definition:** The average number of customers arriving per
              unit of time (e.g., 10 customers/hour).
            - **Calculation:** $\lambda = 1 / (\text{Average Interarrival Time})$
        
        - **μ (Mu): Service Rate (per server)**
            - **Definition:** The average number of customers one server can
              process per unit of time (e.g., 12 customers/hour).
            - **Calculation:** $\mu = 1 / (\text{Average Service Time})$
        
        - **ρ (Rho): System Utilization**
            - **Definition:** The proportion of time that servers are busy.
              It's a measure of system "business".
            - **Calculation (M/M/1):** $\rho = \lambda / \mu$
            - **Calculation (M/M/c):** $\rho = \lambda / (c \times \mu)$
            - **Rule:** If **$\rho \ge 1$**, the system is **UNSTABLE**. Arrivals
              are happening faster than they can be served, and the queue will
              grow to infinity.
        """)

        st.subheader("Performance Metrics (Little's Law)")
        st.markdown(r"""
        These metrics tell you how well your system is performing.
        
        - **L: Average Customers in System**
            - **Definition:** The average number of customers either waiting
              *or* being served.
            - **M/M/1 Formula:** $L = \lambda / (\mu - \lambda) = \rho / (1 - \rho)$
        
        - **Lq: Average Customers in Queue**
            - **Definition:** The average number of customers *only* waiting
              in the line.
            - **M/M/1 Formula:** $L_q = \lambda^2 / (\mu(\mu - \lambda)) = \rho^2 / (1 - \rho)$
        
        - **W: Average Time in System**
            - **Definition:** The average total time a customer spends from
              arrival to service completion (waiting + service).
            - **M/M/1 Formula:** $W = 1 / (\mu - \lambda) = L / \lambda$
        
        - **Wq: Average Time in Queue**
            - **Definition:** The average time a customer spends *only* waiting
              in line (before service begins).
            - **M/M/1 Formula:** $W_q = \lambda / (\mu(\mu - \lambda)) = L_q / \lambda$
        """)
        
        st.subheader("Markov Chain Terms")
        st.markdown(r"""
        - **State:** A specific condition the system can be in (e.g., "Sunny",
          "Broken").
        - **Transition Matrix (P):** A square matrix where `P(i, j)` is the
          probability of moving from state `i` to state `j` in one time step.
          All rows must sum to 1.
        - **State Distribution (π(t)):** A vector showing the probability
          of being in each state at time `t`.
            - $\pi(0)$ is the initial state.
            - $\pi(t+1) = \pi(t) \times P$
        - **Steady-State Distribution (π):** The long-run probability
          distribution that the system settles into, regardless of the
          starting state (assuming the chain is regular).
            - It is the vector $\pi$ that solves the equation: $\pi = \pi \times P$
        - **Expected Return Time:** For a given state `i`, this is the
          average number of steps it takes to return to state `i` after
          leaving it.
            - **Calculation:** $1 / \pi_i$ (where $\pi_i$ is the steady-state
              probability of state `i`).
        """)


# ========================================
# MAIN PAGE FUNCTION
# ========================================

def stochastic_queueing_page():
    st.title("Stochastic Processes & Queueing Theory Analysis")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    
    st.markdown("""
    This page provides tools for analyzing **queueing systems** and **Markov chains**.
    Select a mode from the sidebar to begin.
    
    - **Upload Queue Data**: Analyze your own historical data.
    - **Simulate Queue Systems**: Model and compare hypothetical M/M/c systems.
    - **Markov Chain Analysis**: Analyze discrete-time probabilistic systems.
    
    For a full explanation of terms, please see the **"Guide & Glossary"** page.
    """)
    
    # Sidebar mode selection
    st.sidebar.header("Analysis Mode")
    mode = st.sidebar.radio(
        "Select Mode",
        ["Upload Queue Data", "Simulate Queue Systems", "Markov Chain Analysis"]
    )
    
    # ========================================
    # MODE 1: UPLOAD DATA
    # ========================================
    if mode == "Upload Queue Data":
        st.header("Upload & Analyze Queue Data")
        
        st.markdown("""
        **Required format (CSV):**
        - `arrival_time`: Time when customer arrives
        - `service_time`: Duration of service
        - `customer_id`: Unique identifier (optional)
        """)
        
        # Download template
        template = generate_queue_template()
        st.download_button(
            "Download Template CSV",
            data=template.to_csv(index=False),
            file_name="queue_data_template.csv",
            mime="text/csv"
        )
        
        # File upload
        uploaded_file = st.file_uploader("Upload your queue data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = ['arrival_time', 'service_time']
                if not all(col in df_raw.columns for col in required_cols):
                    st.error(f"Missing required columns: {required_cols}")
                    return
                
                if 'customer_id' not in df_raw.columns:
                    df_raw['customer_id'] = range(1, len(df_raw) + 1)
                
                # Analyze
                df_analyzed, metrics = analyze_uploaded_queue_data(df_raw)
                
                st.subheader("Descriptive Statistics")
                with st.expander("What do these metrics mean?"):
                    st.markdown(r"""
                    These metrics are calculated *directly from your data*.
                    - **Arrival Rate (λ):** The average number of customers
                      arriving per unit of time.
                      ($\lambda = 1 / \text{Avg. Interarrival Time}$)
                    - **Service Rate (μ):** The average number of customers
                      *one server* can process per unit of time.
                      ($\mu = 1 / \text{Avg. Service Time}$)
                    - **Utilization (ρ):** The proportion of time the server
                      is busy. ($\rho = \lambda / \mu$). If $\rho \ge 1$,
                      your system is unstable!
                    - **Avg Wait Time (Wq):** The average time spent in the
                      queue before service.
                    - **Avg System Time (W):** The average total time
                      (Wait + Service).
                    """)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Arrival Rate (λ)", f"{metrics['lambda']:.3f}/unit time")
                with col2:
                    st.metric("Service Rate (μ)", f"{metrics['mu']:.3f}/unit time")
                with col3:
                    st.metric("Utilization (ρ)", f"{metrics['rho']:.3f}")
                    if metrics['rho'] >= 1:
                        st.error("System is UNSTABLE (ρ ≥ 1)")
                with col4:
                    st.metric("Customers", metrics['n_customers'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Wait Time", f"{metrics['avg_wait']:.2f}")
                    st.caption(f"Max: {metrics['max_wait']:.2f}")
                with col2:
                    st.metric("Avg Service Time", f"{metrics['avg_service']:.2f}")
                with col3:
                    st.metric("Avg System Time", f"{metrics['avg_system']:.2f}")
                    st.caption(f"Max: {metrics['max_system']:.2f}")
                
                # Theoretical comparison (M/M/1)
                st.subheader("M/M/1 Theoretical Comparison")
                st.markdown("""
                This compares your observed data to the **theoretical M/M/1 model**
                (assuming Poisson arrivals and exponential service times)
                using your data's calculated λ and μ.
                
                If your observed values are very different from the theoretical
                ones, it suggests your system is *not* M/M/1.
                """)
                theoretical = mm1_theoretical(metrics['lambda'], metrics['mu'])
                
                comparison = pd.DataFrame({
                    'Metric': ['Wq (Avg Wait)', 'W (Avg System)', 'Utilization'],
                    'Observed': [metrics['avg_wait'], metrics['avg_system'], metrics['rho']],
                    'M/M/1 Theory': [
                        theoretical['Wq'] if theoretical['Wq'] != np.inf else np.nan,
                        theoretical['W'] if theoretical['W'] != np.inf else np.nan,
                        theoretical['rho']
                    ]
                })
                st.dataframe(comparison)
                
                # Visualizations
                st.subheader("Visualizations")
                
                tab1, tab2, tab3 = st.tabs(["Timeline", "Distributions", "Data Table"])
                
                with tab1:
                    st.markdown("""
                    This chart shows the journey of each customer.
                    - **Red Line:** Time spent waiting in the queue.
                    - **Green Line:** Time spent in service.
                    """)
                    fig_timeline = plot_queue_timeline(df_analyzed)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                with tab2:
                    st.markdown("""
                    These histograms show the *distribution* of times.
                    - **M/M/1 systems** have **Exponential** distributions.
                    - If your data does not look exponential (e.g., it looks
                      like a Normal/Bell curve), your system is not "M/M".
                    """)
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_wait = go.Figure()
                        fig_wait.add_trace(go.Histogram(
                            x=df_analyzed['wait_time'],
                            nbinsx=20,
                            name='Wait Time'
                        ))
                        fig_wait.update_layout(
                            title='Wait Time Distribution',
                            xaxis_title='Wait Time',
                            yaxis_title='Frequency',
                            height=400
                        )
                        st.plotly_chart(fig_wait, use_container_width=True)
                    
                    with col2:
                        fig_service = go.Figure()
                        fig_service.add_trace(go.Histogram(
                            x=df_analyzed['service_time'],
                            nbinsx=20,
                            name='Service Time'
                        ))
                        fig_service.update_layout(
                            title='Service Time Distribution',
                            xaxis_title='Service Time',
                            yaxis_title='Frequency',
                            height=400
                        )
                        st.plotly_chart(fig_service, use_container_width=True)
                
                with tab3:
                    st.dataframe(df_analyzed)
                    st.download_button(
                        "Download Analyzed Data",
                        data=df_analyzed.to_csv(index=False),
                        file_name="queue_analysis_results.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # ========================================
    # MODE 2: SIMULATE QUEUES
    # ========================================
    elif mode == "Simulate Queue Systems":
        st.header("Queue System Simulation & Comparison")
        st.markdown("""
        This tool uses **Monte Carlo simulation** to model M/M/c queueing systems.
        It generates thousands of random customer arrivals and services based on
        your parameters to predict system performance.
        """)
        
        st.sidebar.subheader("System Parameters")
        lambda_rate = st.sidebar.number_input("Arrival Rate (λ)", 0.1, 100.0, 5.0, 0.5)
        mu_rate = st.sidebar.number_input("Service Rate (μ)", 0.1, 100.0, 6.0, 0.5)
        n_customers = st.sidebar.slider("Customers to Simulate", 100, 5000, 1000, 100)
        seed = st.sidebar.number_input("Random Seed", 0, 9999, 42, 1)
        
        st.sidebar.subheader("Comparison Settings")
        max_servers = st.sidebar.slider("Max Servers to Compare", 1, 10, 5, 1)
        
        # Quick stability check
        rho_1 = lambda_rate / mu_rate
        st.sidebar.markdown("---")
        st.sidebar.metric("M/M/1 Utilization", f"{rho_1:.3f}")
        if rho_1 >= 1:
            st.sidebar.error("Single server is UNSTABLE")
        else:
            st.sidebar.success("Single server is stable")
        
        # Run comparison
        if st.button("Run Simulation & Comparison"):
            # This button press triggers the main (and slow) calculation.
            # Because the function is cached, this is fast after the first run.
            comparison_df = compare_queue_designs(
                lambda_rate, mu_rate, max_servers, n_customers, seed
            )
            
            # Store the cached results in session state to persist them
            # across other widget interactions (like the selectbox)
            st.session_state.comparison_df = comparison_df
        
        
        # Only show the results area if the simulation has been run and
        # the results are in session state.
        if 'comparison_df' in st.session_state:
            
            # Retrieve the results
            comparison_df = st.session_state.comparison_df

            st.subheader("Performance Comparison")
            st.markdown(f"""
            This table compares the simulated performance for systems with
            1 server up to {max_servers} servers.
            - **Green bars** indicate a STABLE system ($\rho < 1$).
            - **Red bars** indicate an UNSTABLE system ($\rho \ge 1$).
            """)
            
            # Display table
            st.dataframe(comparison_df.style.format({
                'rho_theoretical': '{:.3f}',
                'W_theoretical': '{:.3f}',
                'Wq_theoretical': '{:.3f}',
                'L_theoretical': '{:.3f}',
                'W_simulated': '{:.3f}',
                'Wq_simulated': '{:.3f}',
                'L_simulated': '{:.3f}',
                'utilization_simulated': '{:.3f}'
            }))
            
            # Visualizations
            fig_comparison = plot_comparison_table(comparison_df)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Recommendation
            st.subheader("Recommendation")
            stable_configs = comparison_df[comparison_df['stable']]
            
            if len(stable_configs) == 0:
                st.error("No stable configuration found. Increase service rate or reduce arrival rate.")
            else:
                # Find best config (minimize wait time among stable)
                best = stable_configs.loc[stable_configs['Wq_simulated'].idxmin()]
                
                st.success(f"""
                **Recommended Configuration: {int(best['servers'])} server(s)**
                
                - Average Wait Time: {best['Wq_simulated']:.3f} time units
                - Average System Time: {best['W_simulated']:.3f} time units
                - Average Queue Length: {best['L_simulated']:.2f} customers
                - Utilization: {best['utilization_simulated']:.1%}
                """)
                
                # Cost-benefit analysis
                st.markdown("**Trade-off Analysis:**")
                st.markdown(f"- Going from 1 to {int(best['servers'])} servers reduces wait time by "
                          f"{(1 - best['Wq_simulated']/comparison_df.loc[0, 'Wq_simulated'])*100:.1f}%")
            
            # Download results
            st.download_button(
                "Download Comparison Data",
                data=comparison_df.to_csv(index=False),
                file_name="queue_comparison_results.csv",
                mime="text/csv"
            )
            
            # Detailed single simulation
            st.subheader("Detailed Simulation (Select Configuration)")
            st.markdown("Select one of the configurations above to see its detailed simulation.")
            
            # This selectbox changing *used* to trigger a re-run.
            # Now, it just re-runs this small block of code.
            selected_servers = st.selectbox(
                "Number of Servers",
                comparison_df['servers'].tolist(),
                index=0
            )
            
            # This `simulate_queue` function is ALSO cached.
            # It was already run inside the `compare_queue_designs` function,
            # so Streamlit finds the result in cache and returns it instantly.
            sim_results = simulate_queue(lambda_rate, mu_rate, 
                                        n_servers=int(selected_servers), 
                                        n_customers=n_customers, 
                                        seed=seed)
            
            fig_queue_length = plot_queue_length_over_time(
                sim_results['time_points'],
                sim_results['queue_length']
            )
            st.plotly_chart(fig_queue_length, use_container_width=True)
            
            # Distributions
            col1, col2 = st.columns(2)
            with col1:
                fig_wait_dist = go.Figure()
                fig_wait_dist.add_trace(go.Histogram(
                    x=sim_results['wait_times'],
                    nbinsx=30,
                    name='Wait Time'
                ))
                fig_wait_dist.update_layout(
                    title='Wait Time Distribution',
                    xaxis_title='Wait Time',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig_wait_dist, use_container_width=True)
            
            with col2:
                fig_system_dist = go.Figure()
                fig_system_dist.add_trace(go.Histogram(
                    x=sim_results['system_times'],
                    nbinsx=30,
                    name='System Time'
                ))
                fig_system_dist.update_layout(
                    title='System Time Distribution',
                    xaxis_title='System Time',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig_system_dist, use_container_width=True)
    
    # ========================================
    # MODE 3: MARKOV CHAINS
    # ========================================
    elif mode == "Markov Chain Analysis":
        st.header("Discrete-Time Markov Chain Analysis")
        
        st.markdown("""
        Analyze the evolution of a discrete-time Markov chain given a
        **transition matrix (P)**.
        A Markov chain models a system that moves between different **states**
        (e.g., "Sunny", "Cloudy", "Rainy") based on fixed probabilities.
        """)
        
        st.sidebar.subheader("Chain Configuration")
        n_states = st.sidebar.slider("Number of States", 2, 6, 3, 1)
        n_steps = st.sidebar.slider("Time Steps to Simulate", 5, 100, 20, 5)
        
        # Manual matrix input
        st.subheader("Transition Matrix (P)")
        st.markdown(f"""
        Enter the transition probabilities. `P(i, j)` is the probability of
        moving **FROM state i TO state j**.
        
        **IMPORTANT: Each row MUST sum to 1.0.**
        """)
        
        P = np.zeros((n_states, n_states))
        
        cols = st.columns(n_states)
        for i in range(n_states):
            st.markdown(f"**From State {i}:**")
            row_cols = st.columns(n_states)
            for j in range(n_states):
                with row_cols[j]:
                    P[i, j] = st.number_input(
                        f"To State {j}",
                        0.0, 1.0, 1.0/n_states,
                        0.01,
                        key=f"P_{i}_{j}",
                        format="%.3f"
                    )
        
        # Load example matrices
        st.sidebar.subheader("Example Matrices")
        if st.sidebar.button("Weather Model (3 states)"):
            P = np.array([
                [0.7, 0.2, 0.1],  # Sunny -> Sunny, Cloudy, Rainy
                [0.3, 0.4, 0.3],  # Cloudy -> ...
                [0.2, 0.3, 0.5]   # Rainy -> ...
            ])
            st.info("Loaded 3-state weather model. Adjust matrix above if needed.")
        
        if st.sidebar.button("Machine States (2 states)"):
            P = np.array([
                [0.9, 0.1],  # Working -> Working, Broken
                [0.7, 0.3]   # Broken -> Working, Broken
            ])
            st.info("Loaded 2-state machine reliability model.")
        
        # Initial state
        st.sidebar.subheader("Initial Condition")
        init_choice = st.sidebar.radio(
            "Initial Distribution",
            ["Uniform", "Specific State"]
        )
        
        if init_choice == "Specific State":
            initial_state = st.sidebar.selectbox("Start State", list(range(n_states)))
        else:
            initial_state = None
        
        # Analyze button
        if st.button("Analyze Markov Chain"):
            # Validate matrix
            row_sums = P.sum(axis=1)
            if not np.allclose(row_sums, 1.0):
                st.error("Invalid transition matrix. Row sums: {row_sums}")
                st.warning("Each row must sum to 1.0")
                return
            
            with st.spinner("Analyzing Markov chain..."):
                # The cached function is called here
                result = analyze_markov_chain(P, n_steps, initial_state)
            
            if result is None:
                return
            
            # Store result in session state to persist it
            st.session_state.markov_result = result
        
        # Display results if they exist in session state
        if 'markov_result' in st.session_state:
            result = st.session_state.markov_result
            
            # Display results
            st.subheader("Results")
            
            # Transition matrix
            st.markdown("**Transition Matrix P:**")
            P_df = pd.DataFrame(P, 
                               columns=[f"State {j}" for j in range(n_states)],
                               index=[f"State {i}" for i in range(n_states)])
            st.dataframe(P_df.style.format("{:.3f}"))
            
            # Steady state
            st.markdown(r"""
            **Steady-State Distribution (π):**
            
            This is the long-run probability distribution. After many steps,
            the probability of being in any given state "settles" to these
            values, regardless of the starting state.
            
            It is the vector $\pi$ that solves the equation: $\pi = \pi \times P$.
            """)
            steady_df = pd.DataFrame({
                'State': [f"State {i}" for i in range(n_states)],
                'Probability': result['steady_state']
            })
            st.dataframe(steady_df.style.format({'Probability': '{:.4f}'}))
            
            # Visualization
            st.subheader("State Probability Evolution")
            st.markdown("""
            This chart shows the probability of being in each state at each
            time step, starting from your initial condition. You can see the
            probabilities converge towards the steady-state values.
            """)
            fig_evolution = plot_markov_evolution(result)
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            # State diagram
            st.subheader("Transition Diagram")
            st.markdown("**Transition Matrix Heatmap:**")
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=P,
                x=[f"State {j}" for j in range(n_states)],
                y=[f"State {i}" for i in range(n_states)],
                colorscale='Blues',
                text=P,
                texttemplate='%{text:.3f}',
                textfont={"size": 14},
                colorbar=dict(title="Probability")
            ))
            fig_heatmap.update_layout(
                title='Transition Probabilities',
                xaxis_title='To State',
                yaxis_title='From State',
                height=500
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Time evolution table
            st.subheader("Probability Distribution Over Time")
            distributions_df = pd.DataFrame(
                result['distributions'],
                columns=[f"State {i}" for i in range(n_states)]
            )
            distributions_df.insert(0, 'Step', range(n_steps + 1))
            st.dataframe(distributions_df.style.format({
                col: '{:.4f}' for col in distributions_df.columns if col != 'Step'
            }))
            
            st.download_button(
                "Download Evolution Data",
                data=distributions_df.to_csv(index=False),
                file_name="markov_evolution.csv",
                mime="text/csv"
            )
            
            # Analysis insights
            st.subheader("Analysis Insights")
            
            # Check if chain is absorbing
            has_absorbing = np.any(np.diag(P) == 1.0)
            if has_absorbing:
                absorbing_states = [i for i in range(n_states) if P[i, i] == 1.0]
                st.info(f"This chain has absorbing state(s): {absorbing_states}")
            
            # Check if chain is regular (eventually all entries positive)
            is_regular = np.all(result['distributions'][-1] > 0)
            if is_regular:
                st.success("This is a regular Markov chain (converges to steady state)")
            else:
                st.warning("This chain may not be regular (some states unreachable)")
            
            # Expected return time
            st.markdown(r"""
            **Expected Return Times:**
            
            This is the average number of steps it takes to return to a
            state after leaving it.
            
            **Calculation:** $1 / \pi_i$ (where $\pi_i$ is the steady-state
            probability for state `i`).
            """)
            return_times = 1 / result['steady_state']
            return_df = pd.DataFrame({
                'State': [f"State {i}" for i in range(n_states)],
                'Expected Return Time': return_times
            })
            st.dataframe(return_df.style.format({'Expected Return Time': '{:.2f}'}))


# ========================================
# ADDITIONAL PAGES: ECONOMIC ANALYSIS
# ========================================

def economic_analysis_page():
    """Cost-benefit analysis for queue designs"""
    st.header("Economic Analysis of Queue Systems")
    st.markdown("By **Leonardo H. Talero-Sarmiento**")
    
    st.markdown(r"""
    This tool helps you find the **optimal number of servers (c)** by
    balancing two competing costs:
    
    1.  **Cost of Service ($C_s$):** The cost of operating the servers
        (e.g., salaries, equipment). This cost *increases* as you add
        more servers.
    2.  **Cost of Waiting ($C_w$):** The cost of customers waiting in the
        system (e.g., lost sales, dissatisfaction, penalties). This cost
        *decreases* as you add more servers.
    
    **Goal:** Find the number of servers `c` that minimizes the
    **Total Cost**.
    
    $Total Cost = (\text{Server Cost}) + (\text{Waiting Cost})$
    """)
    
    st.sidebar.header("System Parameters")
    lambda_rate = st.sidebar.number_input("Arrival Rate (λ) [customers/hr]", 0.1, 100.0, 10.0, 0.5)
    mu_rate = st.sidebar.number_input("Service Rate (μ) [customers/hr]", 0.1, 100.0, 12.0, 0.5)
    
    st.sidebar.subheader("Cost Parameters")
    cost_server_hr = st.sidebar.number_input("Cost per Server per Hour ($)", 1.0, 1000.0, 50.0, 5.0)
    cost_wait_hr = st.sidebar.number_input("Customer Wait Cost per Hour ($)", 1.0, 1000.0, 30.0, 5.0)
    hours_per_day = st.sidebar.number_input("Operating Hours per Day", 1, 24, 8, 1)
    
    max_servers = st.sidebar.slider("Max Servers to Evaluate", 1, 20, 10, 1)
    
    # ===================================================================
    # START OF IMPROVED EXPLANATION
    # ===================================================================

    st.subheader("How the Calculation Works")
    
    st.markdown(f"""
    The tool finds the optimum by simulating the total cost for every
    server configuration, from `c = 1` to `c = {max_servers}`.
    
    For each number of servers `c`, it performs the following steps:
    """)
    
    st.markdown(r"""
    **1. Calculate System Performance (M/M/c Formulas)**
    - First, it calculates the **system utilization ($\rho$)** to see if the
      system is stable: $\rho = \lambda / (c \times \mu)$.
    - If $\rho \ge 1$, the system is unstable and costs are infinite.
    - If $\rho < 1$ (stable), it calculates the **average customer wait time ($W_q$)**
      using the M/M/c theoretical formulas.
    
    **2. Calculate the Two Opposing Costs (per Day)**
    
    - **A. Daily Server Cost (Rises with `c`)**
        - This is the direct cost of staffing. It's a simple linear
          increase.
        - $Server Cost = (\text{Cost per Server}) \times (\text{Hours per Day}) \times c$
    
    - **B. Daily Waiting Cost (Falls with `c`)**
        - This is the "hidden" cost of poor service. As `c` increases,
          $W_q$ (wait time) drops dramatically, and so does this cost.
        - $Waiting Cost = (\text{Total Daily Arrivals}) \times (\text{Avg. Wait Time } W_q) \times (\text{Cost per Wait Hour})$
        - Where: $Total Daily Arrivals = \lambda \times (\text{Hours per Day})$
    
    **3. Find the Minimum Total Cost**
    - $Total Cost = (\text{Daily Server Cost}) + (\text{Daily Waiting Cost})$
    - The tool calculates this total cost for `c=1`, `c=2`, `c=3`, etc.,
      and presents them in the table.
    - The **optimal configuration** is the value of `c` that has the
      **lowest Total Daily Cost**. This is the "sweet spot" where you are
      spending just enough on servers to minimize customer waiting costs.
    """)
    
    # ===================================================================
    # END OF IMPROVED EXPLANATION
    # ===================================================================

    if st.button("Calculate Optimal Configuration"):
        results = []
        
        for c in range(1, max_servers + 1):
            theoretical = mmc_theoretical(lambda_rate, mu_rate, c)
            
            if theoretical['W'] == np.inf:
                # System is unstable, skip
                results.append({
                    'servers': c,
                    'rho': theoretical['rho'],
                    'Wq': np.inf, 'W': np.inf, 'Lq': np.inf,
                    'server_cost_daily': cost_server_hr * hours_per_day * c,
                    'wait_cost_daily': np.inf,
                    'total_cost_daily': np.inf
                })
                continue
            
            # Daily costs
            server_cost_daily = cost_server_hr * hours_per_day * c
            
            # Expected customers per day
            customers_per_day = lambda_rate * hours_per_day
            
            # Total wait time per day
            total_wait_daily = customers_per_day * theoretical['Wq']
            wait_cost_daily = total_wait_daily * cost_wait_hr
            
            total_cost_daily = server_cost_daily + wait_cost_daily
            
            results.append({
                'servers': c,
                'rho': theoretical['rho'],
                'Wq': theoretical['Wq'],
                'W': theoretical['W'],
                'Lq': theoretical['Lq'],
                'server_cost_daily': server_cost_daily,
                'wait_cost_daily': wait_cost_daily,
                'total_cost_daily': total_cost_daily
            })
        
        if not results:
            st.error("No stable configurations found. Adjust parameters.")
            return
        
        results_df = pd.DataFrame(results)
        
        # Store in session state
        st.session_state.economic_results_df = results_df

    # Display results if they exist in session state
    if 'economic_results_df' in st.session_state:
        results_df = st.session_state.economic_results_df
        
        # Find optimal (ignoring unstable inf costs)
        stable_results = results_df[results_df['total_cost_daily'] != np.inf]
        
        if stable_results.empty:
            st.error("No stable configurations found in the evaluated range.")
            st.dataframe(results_df)
            return
        
        optimal_idx = stable_results['total_cost_daily'].idxmin()
        optimal = stable_results.loc[optimal_idx]
        
        st.subheader("Cost Analysis Results")
        st.dataframe(results_df.style.format({
            'rho': '{:.3f}',
            'Wq': '{:.3f}',
            'W': '{:.3f}',
            'Lq': '{:.2f}',
            'server_cost_daily': '${:,.2f}',
            'wait_cost_daily': '${:,.2f}',
            'total_cost_daily': '${:,.2f}'
        }).highlight_min(subset=['total_cost_daily'], color='lightgreen'))
        
        st.success(f"""
        **Optimal Configuration: {int(optimal['servers'])} servers**
        
        - Daily Server Cost: ${optimal['server_cost_daily']:,.2f}
        - Daily Waiting Cost: ${optimal['wait_cost_daily']:,.2f}
        - **Total Daily Cost: ${optimal['total_cost_daily']:,.2f}**
        - Utilization: {optimal['rho']:.1%}
        - Avg Wait Time: {optimal['Wq']:.3f} hours
        """)
        
        # Visualization
        st.subheader("Cost-Benefit Trade-off")
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cost Breakdown', 'Total Cost Curve')
        )
        
        fig.add_trace(go.Bar(
            x=results_df['servers'],
            y=results_df['server_cost_daily'],
            name='Server Cost',
            marker_color='blue'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=results_df['servers'],
            y=results_df['wait_cost_daily'],
            name='Waiting Cost',
            marker_color='red'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=results_df['servers'],
            y=results_df['total_cost_daily'],
            mode='lines+markers',
            name='Total Cost',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ), row=1, col=2)
        
        # Mark optimal
        fig.add_trace(go.Scatter(
            x=[optimal['servers']],
            y=[optimal['total_cost_daily']],
            mode='markers',
            marker=dict(size=15, color='gold', symbol='star'),
            name='Optimal',
            showlegend=False
        ), row=1, col=2)
        
        fig.update_xaxes(title_text="Number of Servers", row=1, col=1)
        fig.update_xaxes(title_text="Number of Servers", row=1, col=2)
        fig.update_yaxes(title_text="Daily Cost ($)", row=1, col=1)
        fig.update_yaxes(title_text="Total Daily Cost ($)", row=1, col=2)
        
        fig.update_layout(height=500, barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        
        st.download_button(
            "Download Economic Analysis",
            data=results_df.to_csv(index=False),
            file_name="queue_economic_analysis.csv",
            mime="text/csv"
        )


# ========================================
# INTEGRATION WITH MAIN APP
# ========================================

def add_to_navigation():
    """
    Add these pages to your existing PAGES dictionary in the main DOE app:
    
    PAGES = {
        "Guide & Glossary": guide_and_glossary_page,
        "Stochastic Processes & Queueing": stochastic_queueing_page,
        "Economic Analysis (Queues)": economic_analysis_page,
        ... other pages ...
    }
    """
    pass


# ========================================
# STANDALONE EXECUTION (for testing)
# ========================================

if __name__ == "__main__":
    st.set_page_config(page_title="Stochastic Processes", layout="wide")
    
    # Add the new guide page to the dictionary
    PAGES = {
        "Guide & Glossary": guide_and_glossary_page,
        "Queueing Analysis": stochastic_queueing_page,
        "Economic Analysis": economic_analysis_page,
    }
    
    st.sidebar.title('Navigation')
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    # Clear session state if the page choice changes
    if 'current_page' not in st.session_state:
        st.session_state.current_page = choice
    
    if st.session_state.current_page != choice:
        st.session_state.current_page = choice
        # Clear calculation results when switching pages
        if 'comparison_df' in st.session_state:
            del st.session_state.comparison_df
        if 'markov_result' in st.session_state:
            del st.session_state.markov_result
        if 'economic_results_df' in st.session_state:
            del st.session_state.economic_results_df
            
    PAGES[choice]()