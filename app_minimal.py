#!/usr/bin/env python3
"""
Olympic Sprinter Training App - Minimal Dependencies Version
Uses only streamlit, pandas, and numpy for maximum compatibility
"""
import streamlit as st
import pandas as pd
import numpy as np
import random
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
        distance = random.choice(distances)
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
    reps = random.choice([1, 2])
    distance = random.choice(distances)
    
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
    duration = random.choice([30, 40])
    
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
    
    selected_exercises = random.sample(exercises, 5)
    
    return {
        "session_type": "Strength Training",
        "focus": "Explosive power and single leg strength",
        "exercises": selected_exercises,
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

def create_sample_data():
    """Create sample performance data"""
    weeks = list(range(1, 17))
    data = {
        'Week': weeks,
        '100m_Time': [round(10.8 - (i * 0.02) + random.uniform(-0.1, 0.1), 2) for i in range(16)],
        '200m_Time': [round(21.6 - (i * 0.04) + random.uniform(-0.2, 0.2), 2) for i in range(16)],
        'Max_Squat': [round(120 + (i * 2.5) + random.uniform(-10, 10)) for i in range(16)],
        'Vertical_Jump': [round(55 + (i * 0.5) + random.uniform(-4, 4), 1) for i in range(16)]
    }
    return pd.DataFrame(data)

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
                    st.write(f"**Focus:** {session['focus']}")
                elif session['session_type'] == "Speed Endurance":
                    st.write(f"🔥 **{session['session_type']}**")
                    st.write(f"**Energy System:** {session['energy_system']}")
                    st.write(f"**Repetitions:** {', '.join(map(str, session['repetitions']))}m")
                    st.write(f"**Total Distance:** {session['total_distance']}")
                    st.write(f"**Rest:** {session['rest_periods']}")
                    st.write(f"**Intensity:** {session['intensity']}")
                    st.write(f"**Focus:** {session['focus']}")
                elif session['session_type'] == "Oxidative Training":
                    st.write(f"🚴‍♂️ **{session['session_type']}**")
                    st.write(f"**Energy System:** {session['energy_system']}")
                    st.write(f"**Duration:** {session['duration']}")
                    st.write(f"**Intensity:** {session['intensity']}")
                    st.write(f"**Activity:** {session['activity']}")
                    st.write(f"**Focus:** {session['focus']}")
                elif session['session_type'] == "Strength Training":
                    st.write(f"💪 **{session['session_type']}**")
                    st.write(f"**Focus:** {session['focus']}")
                    st.write("**Exercises:**")
                    for exercise in session['exercises']:
                        st.write(f"• {exercise}")
                    st.write(f"**Emphasis:** {session['emphasis']}")
                    st.write(f"**Rest:** {session['rest_periods']}")

elif page == "Periodization Overview":
    st.header("16-Week Periodization Model")
    
    # Create periodization data
    weeks = list(range(1, 17))
    phases = []
    volumes = []
    intensities = []
    
    for week in weeks:
        phase_info = get_periodization_phase(week)
        phases.append(phase_info['phase'])
        
        # Convert to numeric values
        volume_map = {"Low": 1, "Moderate": 2, "Moderate-High": 3, "High": 4}
        intensity_map = {"Low-Moderate": 1, "Moderate": 2, "High": 3, "Very High": 4}
        
        volumes.append(volume_map.get(phase_info['volume'], 2))
        intensities.append(intensity_map.get(phase_info['intensity'], 2))
    
    # Display as table and line chart
    periodization_df = pd.DataFrame({
        'Week': weeks,
        'Phase': phases,
        'Volume': volumes,
        'Intensity': intensities
    })
    
    # Show line chart using Streamlit's built-in charting
    st.subheader("Training Load Progression")
    chart_data = periodization_df.set_index('Week')[['Volume', 'Intensity']]
    st.line_chart(chart_data)
    
    # Show detailed table
    st.subheader("Phase Details")
    st.dataframe(periodization_df, use_container_width=True)
    
    # Phase breakdown
    st.subheader("Training Phases Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Phase 1: Base Building (Weeks 1-4)**")
        st.write("• High volume, low-moderate intensity")
        st.write("• Aerobic base development")
        st.write("• Movement preparation")
        st.write("• General strength foundation")
        
        st.write("**Phase 2: Strength Development (Weeks 5-8)**")
        st.write("• Moderate-high volume, moderate intensity")
        st.write("• Maximum strength focus")
        st.write("• Power development initiation")
        st.write("• Sport-specific movement patterns")
    
    with col2:
        st.write("**Phase 3: Power Development (Weeks 9-12)**")
        st.write("• Moderate volume, high intensity")
        st.write("• Speed-strength emphasis")
        st.write("• Competition preparation")
        st.write("• Technical refinement")
        
        st.write("**Phase 4: Competition Prep (Weeks 13-16)**")
        st.write("• Low volume, very high intensity")
        st.write("• Peak performance focus")
        st.write("• Competition simulation")
        st.write("• Taper and recovery")

elif page == "Performance Analytics":
    st.header("Performance Tracking & Analytics")
    
    # Generate sample performance data
    df = create_sample_data()
    
    # Performance charts using Streamlit built-in charting
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sprint Times Progress")
        sprint_data = df.set_index('Week')[['100m_Time', '200m_Time']]
        st.line_chart(sprint_data)
    
    with col2:
        st.subheader("Strength & Power Progress")
        strength_data = df.set_index('Week')[['Max_Squat', 'Vertical_Jump']]
        st.line_chart(strength_data)
    
    # Performance summary
    st.subheader("Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_100m = min(df['100m_Time'])
        improvement_100m = max(df['100m_Time']) - best_100m
        st.metric("100m Best", f"{best_100m:.2f}s", f"-{improvement_100m:.2f}s")
    
    with col2:
        best_200m = min(df['200m_Time'])
        improvement_200m = max(df['200m_Time']) - best_200m
        st.metric("200m Best", f"{best_200m:.2f}s", f"-{improvement_200m:.2f}s")
    
    with col3:
        max_squat = max(df['Max_Squat'])
        squat_gain = max_squat - min(df['Max_Squat'])
        st.metric("Max Squat", f"{max_squat:.0f}kg", f"+{squat_gain:.0f}kg")
    
    with col4:
        max_jump = max(df['Vertical_Jump'])
        jump_gain = max_jump - min(df['Vertical_Jump'])
        st.metric("Vertical Jump", f"{max_jump:.1f}cm", f"+{jump_gain:.1f}cm")
    
    # Raw data table
    st.subheader("Detailed Performance Data")
    st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Olympic Sprinter Training Generator** - Advanced periodization for 100m/200m sprint performance")
st.markdown("*Using evidence-based training methods for elite sprinter development*")