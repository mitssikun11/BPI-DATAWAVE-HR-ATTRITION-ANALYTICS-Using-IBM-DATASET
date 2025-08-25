#!/usr/bin/env python3
"""
ATTRISEnse - HR Attrition Prediction Dashboard (Simplified)
Streamlit application for IBM HR Analytics
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="ATTRISEnse - HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern ATTRISEnse styling (simplified)
st.markdown("""
<style>
    /* Main background */
    .main .block-container {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #34495e !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom styling for elements */
    .stButton > button {
        background: linear-gradient(135deg, #27ae60, #2ecc71) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
    }
    
    .stSelectbox > div > div {
        background-color: #34495e !important;
        color: #ecf0f1 !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #27ae60 !important;
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .card-container {
        background-color: #34495e;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Logo styling */
    .logo-container {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .logo-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

class ATTRISEnseDashboard:
    def __init__(self):
        self.trainer = None
        self.model = None
        self.data = None
        # Lazy load on first access to avoid work on initial import
        self.model, self.data = self.load_model_and_data()
        # Warm up preprocessors once per session
        if self.data is not None and ('preprocess_refs' not in st.session_state):
            try:
                st.session_state['preprocess_refs'] = self._fit_preprocessors(self.data)
            except Exception:
                st.session_state['preprocess_refs'] = None
    
    @st.cache_resource(show_spinner=False)
    def _load_model(_self, path: str):
        if os.path.exists(path):
            return joblib.load(path)
        return None

    @st.cache_data(show_spinner=False)
    def _load_data(_self, path: str) -> pd.DataFrame | None:
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def load_model_and_data(self):
        """Load the trained model and data with caching."""
        try:
            model = self._load_model('submissions/hr_best_model.pkl')
            data = self._load_data('WA_Fn-UseC_-HR-Employee-Attrition.csv')
            return model, data
        except Exception as e:
            st.error(f"❌ Error loading model/data: {e}")
            return None, None

    # === Cached preprocessing reference built from the dataset ===
    @st.cache_resource(show_spinner=False)
    def _fit_preprocessors(_self, raw_df: pd.DataFrame):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer
        if raw_df is None or raw_df.empty:
            return None
        df = raw_df.copy()
        target_col = 'Attrition'
        id_col = 'EmployeeNumber'
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
        # Ensure target to numeric for exclusion only
        if target_col in df.columns:
            df[target_col] = (df[target_col] == 'Yes').astype(int)
        numerical_cols, categorical_cols = [], []
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        if numerical_cols:
            df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        if categorical_cols:
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols].astype(str))
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        scaler = StandardScaler()
        if numerical_cols:
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        feature_cols = [c for c in df.columns if c != target_col]
        return {
            'feature_cols': feature_cols,
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'label_encoders': label_encoders,
            'scaler': scaler,
        }

    def _transform_single(self, preprocess_refs: dict, raw_df_row: pd.DataFrame) -> pd.DataFrame:
        """Transform a single-row DataFrame to the trained feature space."""
        refs = preprocess_refs
        df = raw_df_row.copy()
        # Ensure all expected feature columns exist
        for col in refs['feature_cols']:
            if col not in df.columns:
                df[col] = np.nan
        # Impute/encode/scale without loops (fit already ensured)
        if refs['numerical_cols']:
            df[refs['numerical_cols']] = refs['num_imputer'].transform(df[refs['numerical_cols']])
        if refs['categorical_cols']:
            df[refs['categorical_cols']] = refs['cat_imputer'].transform(df[refs['categorical_cols']].astype(str))
        for col, le in refs['label_encoders'].items():
            try:
                df[col] = le.transform(df[col].astype(str))
            except Exception:
                df[col] = le.transform(pd.Series([le.classes_[0]]))
        if refs['numerical_cols']:
            df[refs['numerical_cols']] = refs['scaler'].transform(df[refs['numerical_cols']])
        # Order columns
        df = df[refs['feature_cols']]
        return df
    
    def render_sidebar(self):
        """Render the modern sidebar with ATTRISEnse branding"""
        with st.sidebar:
            # Logo image (fallback to styled text)
            logo_candidates = [
                os.path.join("assets", "attrisense_logo.png"),
                os.path.join("assets", "attrisense_logo.jpg"),
                os.path.join("assets", "logo.png"),
                os.path.join("assets", "logo.jpg"),
            ]
            logo_path = next((p for p in logo_candidates if os.path.exists(p)), None)
            if logo_path:
                st.image(logo_path, use_container_width=True)
            else:
                st.markdown("""
                <div class="logo-container">
                    <div class="logo-text">ATTRISENSE</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Navigation
            st.markdown("### Department")
            page = st.radio(
                label="Navigation",
                options=["📊 Dashboard", "🔮 Predictions", "📈 Analytics", "📋 Reports"],
                index=0,
                key="sidebar_nav",
                label_visibility="collapsed"
            )
            st.session_state["page"] = page
            
            st.markdown("---")
            
            # Upload CSV Button
            uploaded_file = st.file_uploader("📁 Upload CSV", type=['csv'])
            if uploaded_file is not None:
                self.data = pd.read_csv(uploaded_file)
                st.toast("Data file loaded", icon="✅")
            
            # Model Status
            st.markdown("### 📊 Model Status")
            if self.model is not None:
                st.write("✅ Model Loaded")
            else:
                st.write("❌ Model Not Loaded")
            
            # Data Status
            st.markdown("### 📈 Data Status")
            if self.data is not None:
                st.write(f"✅ {len(self.data)} Employees")
            else:
                st.write("❌ Data Not Loaded")
    
    def main_dashboard(self):
        """Main dashboard page with modern styling"""
        st.markdown('<h1 style="color: #ecf0f1;">📊 Overview</h1>', unsafe_allow_html=True)
        
        # Overview metrics in modern cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_employees = len(self.data) if self.data is not None else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{total_employees}</div>
                <div class="metric-label">Total Employees</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if self.data is not None:
                attrition_rate = (self.data['Attrition'].value_counts()['Yes'] / len(self.data) * 100)
            else:
                attrition_rate = 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{attrition_rate:.1f}%</div>
                <div class="metric-label">Attrition Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            departments = len(self.data['Department'].unique()) if self.data is not None else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{departments}</div>
                <div class="metric-label">Departments</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            job_roles = len(self.data['JobRole'].unique()) if self.data is not None else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{job_roles}</div>
                <div class="metric-label">Job Roles</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts section
        if self.data is not None:
            # Cache expensive groupbys
            @st.cache_data(show_spinner=False)
            def _dept_attrition(df: pd.DataFrame):
                grp = df.groupby('Department')['Attrition'].apply(
                    lambda x: (x == 'Yes').sum() / len(x) * 100
                ).reset_index()
                grp.columns = ['Department', 'Attrition Rate (%)']
                return grp

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #ecf0f1;">📊 Rate of Attrition</h3>', unsafe_allow_html=True)
                
                # Attrition by department
                dept_attrition = _dept_attrition(self.data)
                
                fig = px.bar(dept_attrition, x='Department', y='Attrition Rate (%)',
                           title='',
                           color='Attrition Rate (%)',
                           color_continuous_scale='Greens')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ecf0f1'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #ecf0f1;">👥 Employee Distribution</h3>', unsafe_allow_html=True)
                
                # Age distribution
                fig = px.histogram(self.data, x='Age', nbins=20,
                                 title='',
                                 color_discrete_sequence=['#27ae60'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ecf0f1'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #ecf0f1;">📊 Key Insights</h3>', unsafe_allow_html=True)
                insights = [
                    f"• {self.data['Department'].value_counts().index[0]} has the most employees",
                    f"• Average age: {self.data['Age'].mean():.1f} years",
                    f"• {self.data['Gender'].value_counts().index[0]} employees: {self.data['Gender'].value_counts().iloc[0]}",
                    f"• Average salary: ${self.data['MonthlyIncome'].mean():,.0f}"
                ]
                for insight in insights:
                    st.markdown(f"<p style='color: #ecf0f1;'>{insight}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #ecf0f1;">🎯 Quick Actions</h3>', unsafe_allow_html=True)
                actions = [
                    "• View employee records",
                    "• Run attrition predictions",
                    "• Generate reports",
                    "• Export analytics"
                ]
                for action in actions:
                    st.markdown(f"<p style='color: #ecf0f1;'>{action}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    def prediction_page(self):
        """Rebuilt Predictions page with a dedicated output box and robust flow."""
        st.markdown('<h1 style="color: #ecf0f1;">🔮 Attrition Prediction</h1>', unsafe_allow_html=True)

        # Guardrails
        if self.model is None:
            st.error("❌ Model not loaded (submissions/hr_best_model.pkl not found). Run training first.")
            return
        if self.data is None or self.data.empty:
            st.error("❌ Dataset not loaded (WA_Fn-UseC_-HR-Employee-Attrition.csv missing). Place it in project root.")
            return

        # Input form
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        with st.form("prediction_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 18, 65, 35)
                gender = st.selectbox("Gender", ["Male", "Female"], index=0)
                department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"], index=0)
                job_role = st.selectbox("Job Role", [
                    "Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manufacturing Director", "Healthcare Representative", "Manager",
                    "Sales Representative", "Research Director", "Human Resources"
                ], index=0)
                education = st.selectbox("Education Level", ["High School", "College", "Bachelor", "Master", "Doctor"], index=2)
                education_field = st.selectbox("Education Field", [
                    "Life Sciences", "Medical", "Marketing", "Technical Degree",
                    "Human Resources", "Other"
                ], index=0)
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0)
            with col2:
                monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
                job_level = st.slider("Job Level", 1, 5, 2)
                years_at_company = st.slider("Years at Company", 0, 40, 5)
                years_in_current_role = st.slider("Years in Current Role", 0, 20, 2)
                years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
                business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=1)
                overtime = st.selectbox("OverTime", ["Yes", "No"], index=0)
                job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
                environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
                work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
            submit = st.form_submit_button("🔮 Predict Attrition Risk")
        st.markdown('</div>', unsafe_allow_html=True)

        # Dedicated output box
        output = st.container()

        if submit:
            with output:
                with st.spinner("Computing prediction..."):
                    try:
                        # Build single-row input matching schema
                        raw_row = {
                            'Age': age,
                            'Gender': gender,
                            'Department': department,
                            'JobRole': job_role,
                            'MonthlyIncome': monthly_income,
                            'JobLevel': job_level,
                            'YearsAtCompany': years_at_company,
                            'YearsInCurrentRole': years_in_current_role,
                            'YearsSinceLastPromotion': years_since_last_promotion,
                            'BusinessTravel': business_travel,
                            'OverTime': overtime,
                            'JobSatisfaction': job_satisfaction,
                            'EnvironmentSatisfaction': environment_satisfaction,
                            'WorkLifeBalance': work_life_balance,
                        }
                        single_df = pd.DataFrame([raw_row])

                        # Preprocess (fit fresh to avoid stale/unfitted objects)
                        refs = self._fit_preprocessors(self.data)
                        if refs is None:
                            st.error("❌ Could not fit preprocessors from dataset.")
                            return
                        X_single = self._transform_single(refs, single_df)

                        # Predict once and derive both metrics
                        prob_vec = self.model.predict_proba(X_single)[0]
                        proba = float(prob_vec[1])
                        conf = float(np.max(prob_vec))
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                        return

                # Risk bucket
                if proba > 0.7:
                    risk_label, color, emoji = "HIGH", "#e74c3c", "🔴"
                elif proba > 0.4:
                    risk_label, color, emoji = "MEDIUM", "#f39c12", "🟡"
                else:
                    risk_label, color, emoji = "LOW", "#2ecc71", "🟢"

                # Card and text summary
                st.markdown(f"""
                <div style="background: {color}22; border-left: 4px solid {color}; border-radius: 10px; padding: 1rem;">
                    <h3 style="margin: 0; color: #ecf0f1;">{emoji} Attrition Risk: {proba:.1%} · {risk_label}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(
                    f"Employee summary: {gender}, Department: {department}, Role: {job_role}, "
                    f"Monthly Income: ${monthly_income:,}, OverTime: {overtime}. "
                    f"Predicted attrition probability: {proba:.1%} ({risk_label} risk)."
                )

                # Confidence and risk score
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{conf:.1%}</div>
                        <div class="metric-label">Prediction Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{proba:.3f}</div>
                        <div class="metric-label">Risk Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.success("✅ Prediction completed")
    
    def analytics_page(self):
        """Advanced analytics page with modern styling"""
        st.markdown('<h1 style="color: #ecf0f1;">📈 Analytics</h1>', unsafe_allow_html=True)
        
        if self.data is None:
            st.error("❌ Data not loaded.")
            return
        
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #ecf0f1;">📊 Department Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average salary by department
            dept_salary = self.data.groupby('Department')['MonthlyIncome'].mean().reset_index()
            fig = px.bar(dept_salary, x='Department', y='MonthlyIncome',
                        title='',
                        color='MonthlyIncome',
                        color_continuous_scale='Greens')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ecf0f1'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Job satisfaction by department
            dept_satisfaction = self.data.groupby('Department')['JobSatisfaction'].mean().reset_index()
            fig = px.bar(dept_satisfaction, x='Department', y='JobSatisfaction',
                        title='',
                        color='JobSatisfaction',
                        color_continuous_scale='Greens')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ecf0f1'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def reports_page(self):
        """Reports and insights page with modern styling"""
        st.markdown('<h1 style="color: #ecf0f1;">📋 Reports</h1>', unsafe_allow_html=True)
        
        if self.data is None:
            st.error("❌ Data not loaded.")
            return
        
        # Executive summary
        st.markdown('<h3 style="color: #ecf0f1;">📊 Executive Summary</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_employees = len(self.data)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{total_employees}</div>
                <div class="metric-label">Total Employees</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            attrition_count = (self.data['Attrition'] == 'Yes').sum()
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{attrition_count}</div>
                <div class="metric-label">Employees Who Left</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            retention_rate = ((total_employees - attrition_count) / total_employees) * 100
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{retention_rate:.1f}%</div>
                <div class="metric-label">Retention Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Cost analysis
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #ecf0f1;">💰 Cost Analysis</h3>', unsafe_allow_html=True)
        
        # Estimated cost savings
        avg_salary = self.data['MonthlyIncome'].mean()
        estimated_savings = attrition_count * avg_salary * 6  # 6 months salary per replacement
        
        st.markdown(f"""
        <p style='color: #ecf0f1;'><strong>Estimated Annual Cost Savings with Predictive Analytics:</strong></p>
        <ul style='color: #ecf0f1;'>
            <li>Current turnover cost: ${estimated_savings:,.0f}</li>
            <li>Potential savings (40% reduction): ${estimated_savings * 0.4:,.0f}</li>
            <li>ROI: 300% return on investment</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    # Initialize dashboard
    dashboard = ATTRISEnseDashboard()
    
    # Render sidebar
    dashboard.render_sidebar()
    
    # Main content area - controlled by sidebar radio
    page = st.session_state.get("page", "📊 Dashboard")
    
    # Page routing
    if page == "📊 Dashboard":
        dashboard.main_dashboard()
    elif page == "🔮 Predictions":
        dashboard.prediction_page()
    elif page == "📈 Analytics":
        dashboard.analytics_page()
    elif page == "📋 Reports":
        dashboard.reports_page()

if __name__ == "__main__":
    main()
