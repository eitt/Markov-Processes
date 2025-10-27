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
# MAIN PAGE FUNCTION
# ========================================

def stochastic_queueing_page():
    st.title("Stochastic Processes & Queueing Theory Analysis")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    
    st.markdown("""
    This page provides tools for analyzing **queueing systems** and **Markov chains**:
    
    - **Upload & Analyze**: Upload real queueing data in long format
    - **Simulate Queues**: Compare M/M/1, M/M/c configurations
    - **Markov Chains**: Analyze discrete-time Markov processes
    """)
    
    # Sidebar mode selection
    st.sidebar.header("Analysis Mode")
    mode = st.sidebar.radio(
        "Select Mode",
        ["📤 Upload Queue Data", "🎲 Simulate Queue Systems", "🔗 Markov Chain Analysis"]
    )
    
    # ========================================
    # MODE 1: UPLOAD DATA
    # ========================================
    if mode == "📤 Upload Queue Data":
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
            "📥 Download Template CSV",
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
                
                st.subheader("📊 Descriptive Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Arrival Rate (λ)", f"{metrics['lambda']:.3f}/unit time")
                with col2:
                    st.metric("Service Rate (μ)", f"{metrics['mu']:.3f}/unit time")
                with col3:
                    st.metric("Utilization (ρ)", f"{metrics['rho']:.3f}")
                    if metrics['rho'] >= 1:
                        st.error("⚠️ System is UNSTABLE (ρ ≥ 1)")
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
                st.subheader("📐 M/M/1 Theoretical Comparison")
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
                st.subheader("📈 Visualizations")
                
                tab1, tab2, tab3 = st.tabs(["Timeline", "Distributions", "Data Table"])
                
                with tab1:
                    fig_timeline = plot_queue_timeline(df_analyzed)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                with tab2:
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
    elif mode == "🎲 Simulate Queue Systems":
        st.header("Queue System Simulation & Comparison")
        
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
            st.sidebar.error("⚠️ Single server is UNSTABLE")
        else:
            st.sidebar.success("✅ Single server is stable")
        
        # Run comparison
        if st.button("🚀 Run Simulation & Comparison"):
            with st.spinner("Simulating queue systems..."):
                comparison_df = compare_queue_designs(
                    lambda_rate, mu_rate, max_servers, n_customers, seed
                )
            
            st.subheader("📊 Performance Comparison")
            
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
            st.subheader("💡 Recommendation")
            stable_configs = comparison_df[comparison_df['stable']]
            
            if len(stable_configs) == 0:
                st.error("❌ No stable configuration found. Increase service rate or reduce arrival rate.")
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
                "📥 Download Comparison Data",
                data=comparison_df.to_csv(index=False),
                file_name="queue_comparison_results.csv",
                mime="text/csv"
            )
            
            # Detailed single simulation
            st.subheader("🔍 Detailed Simulation (Select Configuration)")
            selected_servers = st.selectbox(
                "Number of Servers",
                comparison_df['servers'].tolist(),
                index=0
            )
            
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
    elif mode == "🔗 Markov Chain Analysis":
        st.header("Discrete-Time Markov Chain Analysis")
        
        st.markdown("""
        Analyze the evolution of a discrete-time Markov chain given a transition matrix **P**.
        """)
        
        st.sidebar.subheader("Chain Configuration")
        n_states = st.sidebar.slider("Number of States", 2, 6, 3, 1)
        n_steps = st.sidebar.slider("Time Steps to Simulate", 5, 100, 20, 5)
        
        # Manual matrix input
        st.subheader("Transition Matrix P")
        st.markdown("Enter the transition probabilities (rows must sum to 1):")
        
        P = np.zeros((n_states, n_states))
        
        cols = st.columns(n_states)
        for i in range(n_states):
            st.markdown(f"**From State {i}:**")
            row_cols = st.columns(n_states)
            for j in range(n_states):
                with row_cols[j]:
                    P[i, j] = st.number_input(
                        f"→ State {j}",
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
        if st.button("🔍 Analyze Markov Chain"):
            # Validate matrix
            row_sums = P.sum(axis=1)
            if not np.allclose(row_sums, 1.0):
                st.error(f"⚠️ Invalid transition matrix. Row sums: {row_sums}")
                st.warning("Each row must sum to 1.0")
                return
            
            with st.spinner("Analyzing Markov chain..."):
                result = analyze_markov_chain(P, n_steps, initial_state)
            
            if result is None:
                return
            
            # Display results
            st.subheader("📊 Results")
            
            # Transition matrix
            st.markdown("**Transition Matrix P:**")
            P_df = pd.DataFrame(P, 
                               columns=[f"State {j}" for j in range(n_states)],
                               index=[f"State {i}" for i in range(n_states)])
            st.dataframe(P_df.style.format("{:.3f}"))
            
            # Steady state
            st.markdown("**Steady-State Distribution (π):**")
            steady_df = pd.DataFrame({
                'State': [f"State {i}" for i in range(n_states)],
                'Probability': result['steady_state']
            })
            st.dataframe(steady_df.style.format({'Probability': '{:.4f}'}))
            
            # Visualization
            st.subheader("📈 State Probability Evolution")
            fig_evolution = plot_markov_evolution(result)
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            # State diagram
            st.subheader("🔄 Transition Diagram")
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
            st.subheader("📋 Probability Distribution Over Time")
            distributions_df = pd.DataFrame(
                result['distributions'],
                columns=[f"State {i}" for i in range(n_states)]
            )
            distributions_df.insert(0, 'Step', range(n_steps + 1))
            st.dataframe(distributions_df.style.format({
                col: '{:.4f}' for col in distributions_df.columns if col != 'Step'
            }))
            
            st.download_button(
                "📥 Download Evolution Data",
                data=distributions_df.to_csv(index=False),
                file_name="markov_evolution.csv",
                mime="text/csv"
            )
            
            # Analysis insights
            st.subheader("💡 Analysis Insights")
            
            # Check if chain is absorbing
            has_absorbing = np.any(np.diag(P) == 1.0)
            if has_absorbing:
                absorbing_states = [i for i in range(n_states) if P[i, i] == 1.0]
                st.info(f"🔒 This chain has absorbing state(s): {absorbing_states}")
            
            # Check if chain is regular (eventually all entries positive)
            is_regular = np.all(result['distributions'][-1] > 0)
            if is_regular:
                st.success("✅ This is a regular Markov chain (converges to steady state)")
            else:
                st.warning("⚠️ This chain may not be regular (some states unreachable)")
            
            # Expected return time
            st.markdown("**Expected Return Times:**")
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
    st.header("💰 Economic Analysis of Queue Systems")
    st.markdown("By **Leonardo H. Talero-Sarmiento**")
    
    st.markdown("""
    Determine the optimal number of servers by balancing:
    - **Server costs** (salary, equipment, space)
    - **Customer waiting costs** (lost sales, dissatisfaction)
    """)
    
    st.sidebar.header("System Parameters")
    lambda_rate = st.sidebar.number_input("Arrival Rate (λ)", 0.1, 100.0, 10.0, 0.5)
    mu_rate = st.sidebar.number_input("Service Rate (μ)", 0.1, 100.0, 12.0, 0.5)
    
    st.sidebar.subheader("Cost Parameters")
    cost_server_hr = st.sidebar.number_input("Cost per Server per Hour ($)", 1.0, 1000.0, 50.0, 5.0)
    cost_wait_hr = st.sidebar.number_input("Customer Wait Cost per Hour ($)", 1.0, 1000.0, 30.0, 5.0)
    hours_per_day = st.sidebar.number_input("Operating Hours per Day", 1, 24, 8, 1)
    
    max_servers = st.sidebar.slider("Max Servers to Evaluate", 1, 20, 10, 1)
    
    if st.button("💵 Calculate Optimal Configuration"):
        results = []
        
        for c in range(1, max_servers + 1):
            theoretical = mmc_theoretical(lambda_rate, mu_rate, c)
            
            if theoretical['W'] == np.inf:
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
        
        # Find optimal
        optimal_idx = results_df['total_cost_daily'].idxmin()
        optimal = results_df.loc[optimal_idx]
        
        st.subheader("📊 Cost Analysis Results")
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
            "📥 Download Economic Analysis",
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
        ... existing pages ...
        "Stochastic Processes & Queueing": stochastic_queueing_page,
        "Economic Analysis (Queues)": economic_analysis_page,
    }
    """
    pass


# ========================================
# STANDALONE EXECUTION (for testing)
# ========================================

if __name__ == "__main__":
    st.set_page_config(page_title="Stochastic Processes", layout="wide")
    
    PAGES = {
        "Queueing Analysis": stochastic_queueing_page,
        "Economic Analysis": economic_analysis_page,
    }
    
    st.sidebar.title('Navigation')
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[choice]()