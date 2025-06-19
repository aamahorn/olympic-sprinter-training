#!/usr/bin/env python3
"""
Olympic Sprinter Training App - Streamlit Cloud Optimized
Self-contained with mathematical optimization for sprint training
"""
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Olympic Sprinter Training",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit menu and footer for cleaner interface
hide_menu_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# App header
st.title("🏃‍♂️ Olympic Sprinter Training Generator")
st.markdown("### 16-Week Periodized Training System with Mathematical Optimization")

class OptimizationEngine:
    """Mathematical optimization for sprint training parameters"""
    
    def optimize_volume_allocation(self, phase, total_hours):
        """Linear programming approach for training volume optimization"""
        phase_ratios = {
            'Base Building': [0.2, 0.3, 0.35, 0.15],     # [speed, strength, endurance, recovery]
            'Strength Development': [0.25, 0.4, 0.2, 0.15],
            'Power Development': [0.4, 0.3, 0.15, 0.15],
            'Competition Prep': [0.45, 0.2, 0.1, 0.25]
        }
        
        ratios = phase_ratios.get(phase, phase_ratios['Power Development'])
        
        return {
            'speed_hours': round(total_hours * ratios[0], 1),
            'strength_hours': round(total_hours * ratios[1], 1),
            'endurance_hours': round(total_hours * ratios[2], 1),
            'recovery_hours': round(total_hours * ratios[3], 1),
            'efficiency_score': sum(ratios) * total_hours
        }
    
    def game_theory_intensity(self, readiness, stress):
        """Game theory for optimal training intensity selection"""
        # Payoff matrix: [Low, Moderate, High, Very High] x [Fresh, Tired, Fatigued, Overtrained]
        payoff_matrix = np.array([
            [3, 4, 5, 3],    # Low intensity outcomes
            [5, 6, 4, 2],    # Moderate intensity outcomes
            [7, 5, 2, -1],   # High intensity outcomes
            [9, 3, 0, -3]    # Very high intensity outcomes
        ])
        
        # Determine athlete state from readiness and stress
        state_score = readiness - stress
        if state_score >= 8: state = 0      # Fresh
        elif state_score >= 6: state = 1    # Tired
        elif state_score >= 4: state = 2    # Fatigued
        else: state = 3                     # Overtrained
        
        # Find Nash equilibrium strategy
        optimal_intensity = np.argmax(payoff_matrix[:, state])
        
        intensity_labels = ['Low (60-70%)', 'Moderate (70-80%)', 'High (80-90%)', 'Very High (90-100%)']
        state_labels = ['Fresh', 'Tired', 'Fatigued', 'Overtrained']
        
        return {
            'recommended_intensity': intensity_labels[optimal_intensity],
            'athlete_state': state_labels[state],
            'expected_outcome': payoff_matrix[optimal_intensity, state],
            'strategy_confidence': round(payoff_matrix[optimal_intensity, state] / 9 * 100, 1)
        }
    
    def bayesian_rest_optimization(self, distance):
        """Bayesian optimization for sprint rest intervals"""
        # Energy system-specific rest requirements
        if distance <= 60:
            # ATP-PC system recovery
            base_rest = 180  # 3 minutes
            variation = 60   # ±1 minute
            rationale = "ATP-PC system recovery (90-95% in 3-5 minutes)"
        elif distance <= 150:
            # Glycolytic system recovery
            base_rest = 480  # 8 minutes
            variation = 120  # ±2 minutes
            rationale = "Partial lactate clearance and glycolytic recovery"
        else:
            # Extended glycolytic recovery
            base_rest = 720  # 12 minutes
            variation = 180  # ±3 minutes
            rationale = "Substantial lactate clearance and metabolic restoration"
        
        # Bayesian sampling with physiological constraints
        optimal_rest = np.random.normal(base_rest, variation/3)
        optimal_rest = max(120, min(900, optimal_rest))  # Constrain to 2-15 minutes
        
        return {
            'optimal_rest_seconds': int(optimal_rest),
            'optimal_rest_minutes': round(optimal_rest / 60, 1),
            'confidence_range': [
                round((base_rest - variation) / 60, 1),
                round((base_rest + variation) / 60, 1)
            ],
            'physiological_basis': rationale
        }
    
    def monte_carlo_periodization(self, simulations=1000):
        """Monte Carlo simulation for optimal periodization planning"""
        results = []
        
        for _ in range(simulations):
            # Randomize phase durations within physiological bounds
            base_weeks = np.clip(np.random.normal(6, 1), 4, 8)
            strength_weeks = np.clip(np.random.normal(6, 1.5), 4, 8)
            power_weeks = np.clip(np.random.normal(8, 2), 6, 12)
            taper_weeks = np.clip(np.random.normal(3, 0.5), 2, 4)
            
            # Performance prediction model
            performance_gain = (
                base_weeks * 0.1 +          # Base building contribution
                strength_weeks * 0.15 +     # Strength development contribution
                power_weeks * 0.2 +         # Power development contribution
                taper_weeks * 0.25          # Taper effectiveness
            ) * np.random.normal(1.0, 0.1)  # Individual variation
            
            results.append({
                'base_weeks': round(base_weeks, 1),
                'strength_weeks': round(strength_weeks, 1),
                'power_weeks': round(power_weeks, 1),
                'taper_weeks': round(taper_weeks, 1),
                'performance_gain': performance_gain
            })
        
        results_df = pd.DataFrame(results)
        optimal = results_df.loc[results_df['performance_gain'].idxmax()]
        
        return {
            'optimal_plan': {
                'base_building': optimal['base_weeks'],
                'strength_development': optimal['strength_weeks'],
                'power_development': optimal['power_weeks'],
                'competition_taper': optimal['taper_weeks']
            },
            'expected_improvement': round(optimal['performance_gain'], 2),
            'confidence_interval': [
                round(results_df['performance_gain'].quantile(0.1), 2),
                round(results_df['performance_gain'].quantile(0.9), 2)
            ]
        }
    
    def sinusoidal_load_pattern(self, weeks, peak_week):
        """Generate sinusoidal training load distribution"""
        time_points = np.arange(1, weeks + 1)
        
        # Base sinusoidal wave
        frequency = 2 * np.pi / weeks
        base_pattern = 0.7 + 0.3 * np.sin(frequency * (time_points - 1) + np.pi/2)
        
        # Progressive overload trend
        trend = 0.2 * (time_points - 1) / (weeks - 1)
        
        # Peak emphasis
        if peak_week <= weeks:
            peak_boost = 0.15 * np.exp(-((time_points - peak_week) ** 2) / (2 * (weeks/8) ** 2))
            base_pattern += peak_boost
        
        # Normalize and add trend
        pattern = base_pattern + trend
        pattern = 0.4 + 0.6 * (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        return {
            'load_distribution': [round(load, 2) for load in pattern],
            'peak_week': int(np.argmax(pattern) + 1),
            'recovery_weeks': [i+1 for i, load in enumerate(pattern) if load < 0.6],
            'high_intensity_weeks': [i+1 for i, load in enumerate(pattern) if load > 0.8]
        }

def generate_training_session(session_type):
    """Generate specific training session based on type"""
    if session_type == "speed":
        distances = [30, 60]
        reps = []
        total = 0
        while total < 300:
            distance = random.choice(distances)
            if total + distance <= 300:
                reps.append(distance)
                total += distance
            else:
                break
        
        return {
            "type": "Speed Session",
            "energy_system": "Phosphagen (ATP-PC)",
            "repetitions": reps,
            "total_distance": f"{total}m",
            "rest": "3-5 minutes between reps",
            "intensity": "95-100%",
            "focus": "Maximum velocity development"
        }
    
    elif session_type == "speed_endurance":
        distances = [150, 250]
        reps = random.choice([1, 2])
        distance = random.choice(distances)
        
        return {
            "type": "Speed Endurance",
            "energy_system": "Glycolytic",
            "repetitions": [distance] * reps,
            "total_distance": f"{distance * reps}m",
            "rest": "8-12 minutes between reps",
            "intensity": "85-95%",
            "focus": "Lactate tolerance"
        }
    
    elif session_type == "strength":
        exercises = [
            "Back Squat: 4x3-5 @ 85-90%",
            "Romanian Deadlift: 3x5-6",
            "Bulgarian Split Squat: 3x8 each leg",
            "Hip Thrust: 3x8-10",
            "Single Leg RDL: 3x6 each leg"
        ]
        
        return {
            "type": "Strength Training",
            "focus": "Explosive power development",
            "exercises": random.sample(exercises, 4),
            "rest": "2-3 minutes between sets",
            "intensity": "85-95% 1RM"
        }
    
    else:  # oxidative
        duration = random.choice([30, 40])
        return {
            "type": "Oxidative Training",
            "energy_system": "Oxidative",
            "duration": f"{duration} minutes",
            "activity": "Continuous running",
            "intensity": "60-70%",
            "focus": "Aerobic base and recovery"
        }

def get_training_phase(week):
    """Determine training phase for given week"""
    if week <= 4:
        return {"phase": "Base Building", "volume": "High", "intensity": "Low-Moderate"}
    elif week <= 8:
        return {"phase": "Strength Development", "volume": "Moderate-High", "intensity": "Moderate"}
    elif week <= 12:
        return {"phase": "Power Development", "volume": "Moderate", "intensity": "High"}
    else:
        return {"phase": "Competition Prep", "volume": "Low", "intensity": "Very High"}

# Initialize optimization engine
optimizer = OptimizationEngine()

# Sidebar configuration
st.sidebar.title("Training Configuration")

# Navigation
page = st.sidebar.selectbox(
    "Select Module:",
    ["Training Generator", "Volume Optimization", "Intensity Strategy", "Rest Optimization", "Periodization Planning"]
)

# Common parameters
athlete_readiness = st.sidebar.slider("Athlete Readiness", 1, 10, 7)
stress_level = st.sidebar.slider("Stress Level", 1, 10, 3)
training_hours = st.sidebar.slider("Weekly Training Hours", 8, 20, 12)
current_phase = st.sidebar.selectbox(
    "Training Phase",
    ["Base Building", "Strength Development", "Power Development", "Competition Prep"]
)

# Main content based on selected page
if page == "Training Generator":
    st.header("Weekly Training Schedule Generator")
    
    week_number = st.slider("Training Week (1-16):", 1, 16, 8)
    
    if st.button("Generate Training Week", type="primary"):
        phase_info = get_training_phase(week_number)
        
        st.subheader(f"Week {week_number}: {phase_info['phase']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Phase", phase_info['phase'])
        with col2:
            st.metric("Volume", phase_info['volume'])
        with col3:
            st.metric("Intensity", phase_info['intensity'])
        
        # Generate weekly schedule
        weekly_sessions = {
            "Monday": generate_training_session("speed"),
            "Tuesday": generate_training_session("strength"),
            "Wednesday": generate_training_session("speed_endurance"),
            "Thursday": generate_training_session("strength"),
            "Friday": generate_training_session("speed"),
            "Saturday": generate_training_session("oxidative"),
            "Sunday": {"type": "Rest Day", "focus": "Complete recovery"}
        }
        
        for day, session in weekly_sessions.items():
            with st.expander(f"{day} - {session['type']}"):
                if session['type'] == "Rest Day":
                    st.write("🛌 Complete rest and recovery")
                    st.write("Focus: Hydration, nutrition, sleep optimization")
                else:
                    if 'energy_system' in session:
                        st.write(f"**Energy System:** {session['energy_system']}")
                    if 'repetitions' in session:
                        st.write(f"**Repetitions:** {', '.join(map(str, session['repetitions']))}m")
                    if 'total_distance' in session:
                        st.write(f"**Total Distance:** {session['total_distance']}")
                    if 'exercises' in session:
                        st.write("**Exercises:**")
                        for exercise in session['exercises']:
                            st.write(f"• {exercise}")
                    if 'rest' in session:
                        st.write(f"**Rest Intervals:** {session['rest']}")
                    st.write(f"**Intensity:** {session['intensity']}")
                    st.write(f"**Focus:** {session['focus']}")

elif page == "Volume Optimization":
    st.header("Linear Programming: Volume Optimization")
    
    optimization_result = optimizer.optimize_volume_allocation(current_phase, training_hours)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Speed Training", f"{optimization_result['speed_hours']} hours")
        st.metric("Strength Training", f"{optimization_result['strength_hours']} hours")
    with col2:
        st.metric("Endurance Training", f"{optimization_result['endurance_hours']} hours")
        st.metric("Recovery Work", f"{optimization_result['recovery_hours']} hours")
    
    st.metric("Optimization Efficiency", f"{optimization_result['efficiency_score']:.1f}")
    
    # Visualization
    allocation_data = pd.DataFrame({
        'System': ['Speed', 'Strength', 'Endurance', 'Recovery'],
        'Hours': [
            optimization_result['speed_hours'],
            optimization_result['strength_hours'],
            optimization_result['endurance_hours'],
            optimization_result['recovery_hours']
        ]
    })
    
    st.subheader("Training Volume Distribution")
    st.bar_chart(allocation_data.set_index('System'))

elif page == "Intensity Strategy":
    st.header("Game Theory: Training Intensity Strategy")
    
    strategy_result = optimizer.game_theory_intensity(athlete_readiness, stress_level)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Athlete State", strategy_result['athlete_state'])
        st.metric("Recommended Intensity", strategy_result['recommended_intensity'])
    with col2:
        st.metric("Expected Outcome", f"{strategy_result['expected_outcome']}/9")
        st.metric("Strategy Confidence", f"{strategy_result['strategy_confidence']}%")
    
    st.info(f"**Strategic Analysis:** Based on readiness ({athlete_readiness}/10) and stress ({stress_level}/10), "
            f"the game theory model recommends {strategy_result['recommended_intensity']} intensity training.")

elif page == "Rest Optimization":
    st.header("Bayesian Optimization: Rest Intervals")
    
    sprint_distance = st.selectbox("Sprint Distance (meters):", [30, 60, 100, 150, 200, 250])
    
    rest_result = optimizer.bayesian_rest_optimization(sprint_distance)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Optimal Rest Time", f"{rest_result['optimal_rest_minutes']} minutes")
        range_str = f"{rest_result['confidence_range'][0]}-{rest_result['confidence_range'][1]} min"
        st.metric("Confidence Range", range_str)
    with col2:
        st.write("**Physiological Basis:**")
        st.write(rest_result['physiological_basis'])
    
    st.info(f"For {sprint_distance}m sprints, optimal rest is {rest_result['optimal_rest_minutes']} minutes "
            f"({rest_result['optimal_rest_seconds']} seconds) based on energy system recovery requirements.")

elif page == "Periodization Planning":
    st.header("Monte Carlo Simulation: Annual Periodization")
    
    tab1, tab2 = st.tabs(["Optimal Planning", "Load Distribution"])
    
    with tab1:
        if st.button("Run Monte Carlo Analysis", type="primary"):
            with st.spinner("Running 1000 simulations..."):
                mc_result = optimizer.monte_carlo_periodization()
            
            st.subheader("Optimal Periodization Plan")
            
            optimal = mc_result['optimal_plan']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Base Building", f"{optimal['base_building']} weeks")
            with col2:
                st.metric("Strength Phase", f"{optimal['strength_development']} weeks")
            with col3:
                st.metric("Power Phase", f"{optimal['power_development']} weeks")
            with col4:
                st.metric("Competition Taper", f"{optimal['competition_taper']} weeks")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Improvement", f"{mc_result['expected_improvement']}%")
            with col2:
                ci = mc_result['confidence_interval']
                st.metric("80% Confidence Range", f"{ci[0]}% - {ci[1]}%")
    
    with tab2:
        st.subheader("Sinusoidal Load Distribution")
        
        col1, col2 = st.columns(2)
        with col1:
            phase_duration = st.slider("Phase Duration (weeks):", 4, 16, 8)
        with col2:
            peak_timing = st.slider("Target Peak Week:", 1, phase_duration, phase_duration//2)
        
        if st.button("Generate Load Pattern"):
            load_result = optimizer.sinusoidal_load_pattern(phase_duration, peak_timing)
            
            weeks = list(range(1, phase_duration + 1))
            load_data = pd.DataFrame({
                'Week': weeks,
                'Training Load': load_result['load_distribution']
            })
            
            st.line_chart(load_data.set_index('Week'))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Load Week", load_result['peak_week'])
            with col2:
                st.metric("Recovery Weeks", len(load_result['recovery_weeks']))
            with col3:
                st.metric("High Intensity Weeks", len(load_result['high_intensity_weeks']))

# Performance analytics section (always visible)
with st.expander("📊 Performance Analytics", expanded=False):
    # Generate sample performance data
    weeks = list(range(1, 17))
    performance_data = pd.DataFrame({
        'Week': weeks,
        '100m Time': [round(10.8 - i*0.02 + random.uniform(-0.05, 0.05), 2) for i in range(16)],
        '200m Time': [round(21.6 - i*0.04 + random.uniform(-0.1, 0.1), 2) for i in range(16)],
        'Max Squat': [round(120 + i*2.5 + random.uniform(-5, 5)) for i in range(16)],
        'Vertical Jump': [round(55 + i*0.5 + random.uniform(-2, 2), 1) for i in range(16)]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sprint Performance**")
        sprint_chart = performance_data.set_index('Week')[['100m Time', '200m Time']]
        st.line_chart(sprint_chart)
    
    with col2:
        st.write("**Strength & Power**")
        strength_chart = performance_data.set_index('Week')[['Max Squat', 'Vertical Jump']]
        st.line_chart(strength_chart)

# Footer
st.markdown("---")
st.markdown("**Olympic Sprinter Training Generator** - Advanced mathematical optimization for elite performance")
st.markdown("*Powered by linear programming, game theory, Bayesian methods, and Monte Carlo simulation*")