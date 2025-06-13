#!/usr/bin/env python3
"""
Standalone Olympic Sprinter Training App
Simplified for cloud deployment with embedded fallback functionality
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import os
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Olympic Sprinter Training Generator",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🏃‍♂️ Olympic Sprinter Training Generator")
st.markdown("### 16-Week Periodized Training System for 100m/200m Sprinters")

# Training session generators
def generate_speed_session():
    """Generate speed session (phosphagen-dominant) with short sprints"""
    distances = [30, 60]
    total_distance = 0
    reps = []
    
    while total_distance < 300:
        distance = np.random.choice(distances)
        if total_distance + distance <= 300:
            reps.append(distance)
            total_distance += distance
        else:
            break
    
    return {
        "session_type": "Speed Session",
        "energy_system": "Phosphagen (ATP-PC)",
        "repetitions": reps,
        "total_distance": f"{total_distance}m",
        "rest_periods": "3-5 minutes between reps",
        "intensity": "95-100% effort",
        "focus": "Maximum speed development"
    }

def generate_speed_endurance_session():
    """Generate speed endurance session (glycolytic-dominant)"""
    distances = [150, 250]
    reps = np.random.choice([1, 2])
    distance = np.random.choice(distances)
    
    return {
        "session_type": "Speed Endurance",
        "energy_system": "Glycolytic",
        "repetitions": [distance] * reps,
        "total_distance": f"{distance * reps}m",
        "rest_periods": "8-12 minutes between reps",
        "intensity": "85-95% effort",
        "focus": "Lactate tolerance and speed maintenance"
    }

def generate_oxidative_session():
    """Generate oxidative energy session"""
    duration = np.random.choice([30, 40])
    
    return {
        "session_type": "Oxidative Training",
        "energy_system": "Oxidative",
        "duration": f"{duration} minutes",
        "intensity": "60-70% effort",
        "activity": "Continuous running or cycling",
        "focus": "Recovery and aerobic base"
    }

def generate_strength_session():
    """Generate strength training session"""
    exercises = [
        "Back Squat: 4x3-5 @ 85-90%",
        "Romanian Deadlift: 3x5-6",
        "Bulgarian Split Squat: 3x8 each leg",
        "Hip Thrust: 3x8-10",
        "Single Leg RDL: 3x6 each leg",
        "Calf Raises: 4x12-15"
    ]
    
    selected_exercises = np.random.choice(exercises, size=5, replace=False)
    
    return {
        "session_type": "Strength Training",
        "focus": "Explosive power and single leg strength",
        "exercises": list(selected_exercises),
        "emphasis": "Concentric/eccentric control",
        "rest_periods": "2-3 minutes between sets"
    }

def get_periodization_phase(week):
    """Determine training phase based on 16-week model"""
    if week <= 4:
        return {"phase": "Base Building", "volume": "High", "intensity": "Low-Moderate"}
    elif week <= 8:
        return {"phase": "Strength Development", "volume": "Moderate-High", "intensity": "Moderate"}
    elif week <= 12:
        return {"phase": "Power Development", "volume": "Moderate", "intensity": "High"}
    else:
        return {"phase": "Competition Prep", "volume": "Low", "intensity": "Very High"}

def create_weekly_schedule(week_number):
    """Create a complete weekly training schedule"""
    phase_info = get_periodization_phase(week_number)
    
    schedule = {
        "week": week_number,
        "phase": phase_info["phase"],
        "volume": phase_info["volume"],
        "intensity": phase_info["intensity"],
        "sessions": {}
    }
    
    # Generate sessions for each day
    sessions = {
        "Monday": generate_speed_session(),
        "Tuesday": generate_strength_session(),
        "Wednesday": generate_speed_endurance_session(),
        "Thursday": generate_strength_session(),
        "Friday": generate_speed_session(),
        "Saturday": generate_oxidative_session(),
        "Sunday": {"session_type": "Rest Day", "focus": "Recovery"}
    }
    
    schedule["sessions"] = sessions
    return schedule

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["Training Generator", "Periodization Overview", "Performance Analytics"]
)

if page == "Training Generator":
    st.header("Weekly Training Schedule Generator")
    
    # Week selection
    week_number = st.slider("Select Training Week (1-16):", 1, 16, 1)
    
    # Generate schedule
    if st.button("Generate Training Week", type="primary"):
        schedule = create_weekly_schedule(week_number)
        
        # Display phase information
        st.subheader(f"Week {week_number}: {schedule['phase']}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Phase", schedule['phase'])
        with col2:
            st.metric("Volume", schedule['volume'])
        with col3:
            st.metric("Intensity", schedule['intensity'])
        
        # Display daily sessions
        for day, session in schedule['sessions'].items():
            with st.expander(f"{day} - {session['session_type']}"):
                if session['session_type'] == "Rest Day":
                    st.write("🛌 **Complete rest and recovery**")
                    st.write("Focus on hydration, nutrition, and sleep")
                elif session['session_type'] == "Speed Session":
                    st.write(f"🏃‍♂️ **{session['session_type']}**")
                    st.write(f"**Energy System:** {session['energy_system']}")
                    st.write(f"**Repetitions:** {', '.join(map(str, session['repetitions']))}m")
                    st.write(f"**Total Distance:** {session['total_distance']}")
                    st.write(f"**Rest:** {session['rest_periods']}")
                    st.write(f"**Intensity:** {session['intensity']}")
                elif session['session_type'] == "Speed Endurance":
                    st.write(f"🔥 **{session['session_type']}**")
                    st.write(f"**Energy System:** {session['energy_system']}")
                    st.write(f"**Repetitions:** {', '.join(map(str, session['repetitions']))}m")
                    st.write(f"**Total Distance:** {session['total_distance']}")
                    st.write(f"**Rest:** {session['rest_periods']}")
                    st.write(f"**Intensity:** {session['intensity']}")
                elif session['session_type'] == "Oxidative Training":
                    st.write(f"🚴‍♂️ **{session['session_type']}**")
                    st.write(f"**Energy System:** {session['energy_system']}")
                    st.write(f"**Duration:** {session['duration']}")
                    st.write(f"**Intensity:** {session['intensity']}")
                    st.write(f"**Activity:** {session['activity']}")
                elif session['session_type'] == "Strength Training":
                    st.write(f"💪 **{session['session_type']}**")
                    st.write(f"**Focus:** {session['focus']}")
                    st.write("**Exercises:**")
                    for exercise in session['exercises']:
                        st.write(f"• {exercise}")
                    st.write(f"**Rest:** {session['rest_periods']}")

elif page == "Periodization Overview":
    st.header("16-Week Periodization Model")
    
    # Create periodization chart
    weeks = list(range(1, 17))
    phases = []
    volumes = []
    intensities = []
    
    for week in weeks:
        phase_info = get_periodization_phase(week)
        phases.append(phase_info['phase'])
        
        # Convert to numeric values for plotting
        volume_map = {"Low": 1, "Moderate": 2, "Moderate-High": 3, "High": 4}
        intensity_map = {"Low-Moderate": 1, "Moderate": 2, "High": 3, "Very High": 4}
        
        volumes.append(volume_map.get(phase_info['volume'], 2))
        intensities.append(intensity_map.get(phase_info['intensity'], 2))
    
    # Plot periodization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=weeks, y=volumes,
        mode='lines+markers',
        name='Training Volume',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=weeks, y=intensities,
        mode='lines+markers',
        name='Training Intensity',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title="Training Volume vs Intensity Across 16 Weeks",
        xaxis_title="Week",
        yaxis_title="Level (1=Low, 4=High)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Phase breakdown
    st.subheader("Training Phases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Phase 1: Base Building (Weeks 1-4)**")
        st.write("• High volume, low-moderate intensity")
        st.write("• Aerobic base development")
        st.write("• Movement preparation")
        
        st.write("**Phase 2: Strength Development (Weeks 5-8)**")
        st.write("• Moderate-high volume, moderate intensity")
        st.write("• Maximum strength focus")
        st.write("• Power development initiation")
    
    with col2:
        st.write("**Phase 3: Power Development (Weeks 9-12)**")
        st.write("• Moderate volume, high intensity")
        st.write("• Speed-strength emphasis")
        st.write("• Competition preparation")
        
        st.write("**Phase 4: Competition Prep (Weeks 13-16)**")
        st.write("• Low volume, very high intensity")
        st.write("• Peak performance focus")
        st.write("• Competition simulation")

elif page == "Performance Analytics":
    st.header("Performance Tracking & Analytics")
    
    # Sample performance data
    sample_data = {
        'Week': list(range(1, 17)),
        '100m_Time': [10.8 - (i * 0.02) + np.random.normal(0, 0.05) for i in range(16)],
        '200m_Time': [21.6 - (i * 0.04) + np.random.normal(0, 0.1) for i in range(16)],
        'Max_Squat': [120 + (i * 2.5) + np.random.normal(0, 5) for i in range(16)],
        'Vertical_Jump': [55 + (i * 0.5) + np.random.normal(0, 2) for i in range(16)]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(df, x='Week', y=['100m_Time', '200m_Time'], 
                      title='Sprint Times Progress',
                      labels={'value': 'Time (seconds)', 'variable': 'Distance'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.line(df, x='Week', y=['Max_Squat', 'Vertical_Jump'],
                      title='Strength & Power Progress',
                      labels={'value': 'Performance', 'variable': 'Metric'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Performance summary
    st.subheader("Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("100m Best", f"{min(df['100m_Time']):.2f}s", 
                 f"{min(df['100m_Time']) - max(df['100m_Time']):.2f}s")
    with col2:
        st.metric("200m Best", f"{min(df['200m_Time']):.2f}s",
                 f"{min(df['200m_Time']) - max(df['200m_Time']):.2f}s")
    with col3:
        st.metric("Max Squat", f"{max(df['Max_Squat']):.0f}kg",
                 f"+{max(df['Max_Squat']) - min(df['Max_Squat']):.0f}kg")
    with col4:
        st.metric("Vertical Jump", f"{max(df['Vertical_Jump']):.1f}cm",
                 f"+{max(df['Vertical_Jump']) - min(df['Vertical_Jump']):.1f}cm")

# Footer
st.markdown("---")
st.markdown("**Olympic Sprinter Training Generator** - Advanced periodization for 100m/200m sprint performance")