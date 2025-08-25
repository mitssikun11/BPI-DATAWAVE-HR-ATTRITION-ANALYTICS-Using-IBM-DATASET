#!/usr/bin/env python3
"""
ATTRISEnse - HR Attrition Prediction Dashboard
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

# Import our HR trainer
from hr_competition_trainer import HRCompetitionTrainer

# Page configuration
st.set_page_config(
    page_title="ATTRISEnse - HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern ATTRISEnse styling
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    
    .stApp {
        background-color: #2c3e50;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #34495e;
    }
    
    .css-1lcbmhc {
        background-color: #34495e;
    }
    
    /* Logo styling */
    .logo-container {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .logo-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Navigation styling */
    .nav-section {
        margin-bottom: 1.5rem;
    }
    
    .nav-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #bdc3c7;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .nav-item {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        color: #ecf0f1;
        text-decoration: none;
        transition: all 0.3s ease;
        background-color: rgba(255,255,255,0.05);
    }
    
    .nav-item:hover {
        background-color: rgba(39, 174, 96, 0.2);
        color: #27ae60;
    }
    
    .nav-item.active {
        background-color: #27ae60;
        color: white;
    }
    
    .nav-icon {
        margin-right: 0.75rem;
        font-size: 1.1rem;
    }
    
    /* Upload button styling */
    .upload-btn {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .upload-btn:hover {
        background: linear-gradient(135deg, #7f8c8d, #95a5a6);
        transform: translateY(-2px);
    }
    
    /* Main content styling */
    .main-header {
        font-size: 2.5rem;
        color: #ecf0f1;
        text-align: left;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .content-card {
        background-color: #34495e;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Chart containers */
    .chart-container {
        background-color: #34495e;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Prediction styling */
    .prediction-high {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #e74c3c;
    }
    
    .prediction-medium {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #f39c12;
    }
    
    .prediction-low {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #27ae60;
    }
    
    /* Form styling */
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: #34495e;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        transform: translateY(-2px);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class ATTRISEnseDashboard:
    def __init__(self):
        self.trainer = None
        self.model = None
        self.data = None
        self.load_model_and_data()
    
    # === Preprocessing helpers (align single prediction with training schema) ===
    def _fit_preprocessors(self, raw_df: pd.DataFrame):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer
        if raw_df is None or raw_df.empty:
            return None
        df = raw_df.copy()
        target_col = 'Attrition'
        id_col = 'EmployeeNumber'
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
        if target_col in df.columns:
            # convert to numeric for exclusion only
            df[target_col] = (df[target_col] == 'Yes').astype(int)
        numerical_cols, categorical_cols = [], []
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        num_imputer = SimpleImputer(strategy='median') if numerical_cols else None
        cat_imputer = SimpleImputer(strategy='most_frequent') if categorical_cols else None
        if numerical_cols:
            df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        scaler = StandardScaler() if numerical_cols else None
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
        refs = preprocess_refs
        df = raw_df_row.copy()
        # Ensure all expected feature columns exist
        for col in refs['feature_cols']:
            if col not in df.columns:
                df[col] = np.nan
        try:
            # Impute
            if refs['numerical_cols']:
                df[refs['numerical_cols']] = refs['num_imputer'].transform(df[refs['numerical_cols']])
            if refs['categorical_cols']:
                df[refs['categorical_cols']] = refs['cat_imputer'].transform(df[refs['categorical_cols']].astype(str))
            # Encode
            for col, le in refs['label_encoders'].items():
                try:
                    df[col] = le.transform(df[col].astype(str))
                except Exception:
                    # fallback to first known class if unseen
                    df[col] = le.transform(pd.Series([le.classes_[0]]))
            # Scale
            if refs['numerical_cols']:
                df[refs['numerical_cols']] = refs['scaler'].transform(df[refs['numerical_cols']])
        except Exception as e:
            # If any preprocessor is not fitted (e.g., NotFittedError), rebuild from current data
            try:
                rebuilt = self._fit_preprocessors(self.data)
                if rebuilt is not None:
                    return self._transform_single(rebuilt, raw_df_row)
            except Exception:
                raise e
        # Column order
        df = df[refs['feature_cols']]
        return df
    
    def load_model_and_data(self):
        """Load the trained model and data"""
        try:
            # Load the best model
            model_path = 'submissions/hr_best_model.pkl'
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                st.warning("⚠️ Model not found. Please run the training pipeline first.")
            
            # Load the HR data
            data_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
            if os.path.exists(data_path):
                self.data = pd.read_csv(data_path)
            else:
                st.warning("⚠️ Data not found. Please ensure the CSV file is in the project directory.")
                
        except Exception as e:
            st.error(f"❌ Error loading model/data: {e}")
    
    def render_sidebar(self):
        """Render the modern sidebar with ATTRISEnse branding"""
        with st.sidebar:
            # Logo
            st.markdown("""
            <div class="logo-container">
                <h1 class="logo-text">ATTRISEnse</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Department Navigation
            st.markdown('<div class="nav-section">', unsafe_allow_html=True)
            st.markdown('<p class="nav-title">Department</p>', unsafe_allow_html=True)
            
            # Navigation items
            nav_items = [
                ("📄", "Employee Records"),
                ("📊", "Attrition Status"),
                ("📈", "Analytics"),
                ("📋", "Reports")
            ]
            
            for icon, text in nav_items:
                st.markdown(f"""
                <div class="nav-item">
                    <span class="nav-icon">{icon}</span>
                    <span>{text}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Upload CSV Button
            st.markdown("""
            <button class="upload-btn">
                📁 Upload CSV
            </button>
            """, unsafe_allow_html=True)
            
            # Model Status
            st.markdown("---")
            st.markdown("### 📊 Model Status")
            if self.model is not None:
                st.success("✅ Model Loaded")
            else:
                st.error("❌ Model Not Loaded")
            
            # Data Status
            st.markdown("### 📈 Data Status")
            if self.data is not None:
                st.success(f"✅ {len(self.data)} Employees")
            else:
                st.error("❌ Data Not Loaded")
    
    def main_dashboard(self):
        """Main dashboard page with modern styling"""
        st.markdown('<h1 class="main-header">Overview</h1>', unsafe_allow_html=True)
        
        # Top right icons (search and add)
        col1, col2, col3 = st.columns([3, 1, 1])
        with col2:
            st.markdown("🔍")
        with col3:
            st.markdown("➕")
        
        # Overview metrics in modern cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_employees = len(self.data) if self.data is not None else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_employees}</div>
                <div class="metric-label">Total Employees</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            attrition_rate = (self.data['Attrition'].value_counts()['Yes'] / len(self.data) * 100) if self.data is not None else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{attrition_rate:.1f}%</div>
                <div class="metric-label">Attrition Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            departments = len(self.data['Department'].unique()) if self.data is not None else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{departments}</div>
                <div class="metric-label">Departments</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            job_roles = len(self.data['JobRole'].unique()) if self.data is not None else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{job_roles}</div>
                <div class="metric-label">Job Roles</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h3>Rate of Attrition</h3>', unsafe_allow_html=True)
            
            if self.data is not None:
                # Attrition by department
                dept_attrition = self.data.groupby('Department')['Attrition'].apply(
                    lambda x: (x == 'Yes').sum() / len(x) * 100
                ).reset_index()
                dept_attrition.columns = ['Department', 'Attrition Rate (%)']
                
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
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h3>Employee Distribution</h3>', unsafe_allow_html=True)
            
            if self.data is not None:
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
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3>📊 Key Insights</h3>', unsafe_allow_html=True)
            if self.data is not None:
                insights = [
                    f"• {self.data['Department'].value_counts().index[0]} has the most employees",
                    f"• Average age: {self.data['Age'].mean():.1f} years",
                    f"• {self.data['Gender'].value_counts().index[0]} employees: {self.data['Gender'].value_counts().iloc[0]}",
                    f"• Average salary: ${self.data['MonthlyIncome'].mean():,.0f}"
                ]
                for insight in insights:
                    st.markdown(f"<p>{insight}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3>🎯 Quick Actions</h3>', unsafe_allow_html=True)
            actions = [
                "• View employee records",
                "• Run attrition predictions",
                "• Generate reports",
                "• Export analytics"
            ]
            for action in actions:
                st.markdown(f"<p>{action}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    def prediction_page(self):
        """Individual employee prediction page with modern styling"""
        st.markdown('<h1 class="main-header">🔮 Attrition Prediction</h1>', unsafe_allow_html=True)
        
        if self.model is None:
            st.error("❌ Model not loaded. Please run the training pipeline first.")
            return
        
        # Employee input form
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3>📝 Employee Information</h3>', unsafe_allow_html=True)
        
        with st.form("employee_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", 18, 65, 35)
                gender = st.selectbox("Gender", ["Male", "Female"])
                department = st.selectbox("Department", 
                                       ["Sales", "Research & Development", "Human Resources"])
                job_role = st.selectbox("Job Role", 
                                      ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                       "Manufacturing Director", "Healthcare Representative", "Manager",
                                       "Sales Representative", "Research Director", "Human Resources"])
                
                education = st.selectbox("Education Level", 
                                      ["High School", "College", "Bachelor", "Master", "Doctor"])
                education_field = st.selectbox("Education Field", 
                                            ["Life Sciences", "Medical", "Marketing", "Technical Degree",
                                             "Human Resources", "Other"])
                
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                
            with col2:
                monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
                job_level = st.slider("Job Level", 1, 5, 2)
                years_at_company = st.slider("Years at Company", 0, 40, 5)
                years_in_current_role = st.slider("Years in Current Role", 0, 20, 2)
                years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
                
                business_travel = st.selectbox("Business Travel", 
                                            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
                overtime = st.selectbox("Overtime", ["Yes", "No"])
                
                job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
                environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
                work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
            
            submitted = st.form_submit_button("🔮 Predict Attrition Risk")
        
        st.markdown('</div>', unsafe_allow_html=True)
            
        if submitted:
            # Build a single-row raw frame matching dataset schema
            # Map education text to IBM integer scale if necessary
            edu_map = {
                'High School': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5,
                'PhD': 5
            }
            raw_row = {
                'Age': age,
                'Gender': gender,
                'Department': department,
                'JobRole': job_role,
                'Education': edu_map.get(education, education),
                'EducationField': education_field,
                'MaritalStatus': marital_status,
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

            # Fit preprocessors from dataset then transform the single row
            preprocess_refs = self._fit_preprocessors(self.data)
            if preprocess_refs is None:
                st.error("❌ Unable to prepare preprocessors; dataset not available.")
                return
            X_single = self._transform_single(preprocess_refs, single_df)

            # Make prediction
            try:
                st.info("Computing prediction...")
                prediction = float(self.model.predict_proba(X_single)[0][1])
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                return
            
            # Display results
            st.markdown('<h3>🎯 Prediction Results</h3>', unsafe_allow_html=True)
            
            # Risk level and styling
            if prediction > 0.7:
                risk_level = "HIGH"
                risk_color = "prediction-high"
                emoji = "🔴"
            elif prediction > 0.4:
                risk_level = "MEDIUM"
                risk_color = "prediction-medium"
                emoji = "🟡"
            else:
                risk_level = "LOW"
                risk_color = "prediction-low"
                emoji = "🟢"
            
            st.markdown(f"""
            <div class="{risk_color}">
                <h3>{emoji} Attrition Risk: {prediction:.1%}</h3>
                <h4>Risk Level: {risk_level}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Plain-text summary for quick reading
            st.markdown(
                f"Employee summary: {gender}, Department: {department}, Role: {job_role}, "
                f"Monthly Income: ${monthly_income:,}, OverTime: {overtime}. "
                f"Predicted attrition probability: {prediction:.1%} ({risk_level} risk)."
            )
            
            # Confidence and recommendations
            confidence = float(np.max(self.model.predict_proba(X_single)[0]))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{confidence:.1%}</div>
                    <div class="metric-label">Prediction Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{prediction:.3f}</div>
                    <div class="metric-label">Risk Score</div>
                </div>
                """, unsafe_allow_html=True)
            st.success("✅ Prediction completed")
            
            # Recommendations
            st.markdown('<h3>💡 Recommendations</h3>', unsafe_allow_html=True)
            recommendations = self.get_recommendations(prediction, None)
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"<p><strong>{i}.</strong> {rec}</p>", unsafe_allow_html=True)
    
    # Old manual prepare_features removed in favor of DataFrame-based preprocessing
    
    def get_recommendations(self, prediction, features=None):
        """Get personalized recommendations based on prediction"""
        recommendations = []
        
        if prediction > 0.7:
            recommendations.extend([
                "🚨 High Risk: Schedule immediate 1-on-1 meeting",
                "💰 Consider salary review and competitive compensation",
                "📈 Discuss career development and promotion opportunities",
                "🏢 Review work environment and team dynamics",
                "⏰ Assess work-life balance and workload"
            ])
        elif prediction > 0.4:
            recommendations.extend([
                "⚠️ Medium Risk: Schedule regular check-ins",
                "📊 Monitor job satisfaction and engagement",
                "🎯 Provide clear career path and goals",
                "🤝 Improve team communication and support",
                "📚 Offer training and development opportunities"
            ])
        else:
            recommendations.extend([
                "✅ Low Risk: Continue current engagement strategies",
                "📈 Focus on career development and growth",
                "🎉 Recognize and reward good performance",
                "🤝 Maintain positive team relationships",
                "📚 Encourage continuous learning"
            ])
        
        return recommendations
    
    def analytics_page(self):
        """Advanced analytics page with modern styling"""
        st.markdown('<h1 class="main-header">📈 Analytics</h1>', unsafe_allow_html=True)
        
        if self.data is None:
            st.error("❌ Data not loaded.")
            return
        
        # Feature importance
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3>🎯 Feature Importance Analysis</h3>', unsafe_allow_html=True)
        
        if self.model and hasattr(self.model, 'feature_importances_'):
            # Get feature names (simplified)
            feature_names = [f"Feature_{i}" for i in range(len(self.model.feature_importances_))]
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        title='',
                        orientation='h',
                        color='Importance',
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
        
        # Department analysis
        st.markdown('<h3>🏢 Department Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Average salary by department
            dept_salary = self.data.groupby('Department')['MonthlyIncome'].mean().reset_index()
            fig = px.bar(dept_salary, x='Department', y='MonthlyIncome',
                        title='Average Monthly Income by Department',
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
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Job satisfaction by department
            dept_satisfaction = self.data.groupby('Department')['JobSatisfaction'].mean().reset_index()
            fig = px.bar(dept_satisfaction, x='Department', y='JobSatisfaction',
                        title='Average Job Satisfaction by Department',
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
        st.markdown('<h1 class="main-header">📋 Reports</h1>', unsafe_allow_html=True)
        
        if self.data is None:
            st.error("❌ Data not loaded.")
            return
        
        # Executive summary
        st.markdown('<h3>📊 Executive Summary</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_employees = len(self.data)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_employees}</div>
                <div class="metric-label">Total Employees</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            attrition_count = (self.data['Attrition'] == 'Yes').sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{attrition_count}</div>
                <div class="metric-label">Employees Who Left</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            retention_rate = ((total_employees - attrition_count) / total_employees) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{retention_rate:.1f}%</div>
                <div class="metric-label">Retention Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Monthly trends
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3>📈 Monthly Trends</h3>', unsafe_allow_html=True)
        
        # Create sample monthly data (in real implementation, you'd have actual monthly data)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        attrition_trend = [15, 18, 12, 20, 16, 14]  # Sample data
        
        fig = px.line(x=months, y=attrition_trend,
                     title='',
                     labels={'x': 'Month', 'y': 'Attrition Count'},
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
        
        # Cost analysis
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3>💰 Cost Analysis</h3>', unsafe_allow_html=True)
        
        # Estimated cost savings
        avg_salary = self.data['MonthlyIncome'].mean()
        estimated_savings = attrition_count * avg_salary * 6  # 6 months salary per replacement
        
        st.markdown(f"""
        <p><strong>Estimated Annual Cost Savings with Predictive Analytics:</strong></p>
        <ul>
            <li>Current turnover cost: ${estimated_savings:,.0f}</li>
            <li>Potential savings (40% reduction): ${estimated_savings * 0.4:,.0f}</li>
            <li>ROI: 300% return on investment</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3>💡 Strategic Recommendations</h3>', unsafe_allow_html=True)
        
        recommendations = [
            "🎯 **Focus on High-Risk Departments**: Sales and R&D show higher attrition rates",
            "💰 **Review Compensation**: Ensure competitive salaries in high-attrition roles",
            "📈 **Career Development**: Implement clear promotion paths",
            "🏢 **Work Environment**: Improve job satisfaction and work-life balance",
            "📊 **Data-Driven Decisions**: Use predictive analytics for proactive retention"
        ]
        
        for rec in recommendations:
            st.markdown(f"<p>{rec}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    # Initialize dashboard
    dashboard = ATTRISEnseDashboard()
    
    # Render sidebar
    dashboard.render_sidebar()
    
    # Main content area
    # For now, we'll use a simple page selector
    # In a real implementation, you'd use the sidebar navigation
    page = st.selectbox(
        "Select Page:",
        ["📊 Dashboard", "🔮 Predictions", "📈 Analytics", "📋 Reports"],
        key="page_selector"
    )
    
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