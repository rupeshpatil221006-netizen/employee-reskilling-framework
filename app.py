# employee_reskilling_app.py

import os
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Reskilling Framework",
    page_icon="🎓",
    layout="wide"
)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

DB_PATH = "data/workforce.db"
MODEL_PATH = "models/readiness_model.pkl"
DATA_PATH = "data/employee_dataset.csv"

SKILLS = [
    "Python",
    "Machine_Learning",
    "Data_Analysis",
    "Communication",
    "Problem_Solving",
    "Cloud_Computing",
    "Cybersecurity",
    "Leadership",
    "Adaptability",
    "Digital_Literacy"
]

NUMERIC_FEATURES = [
    "Age",
    "Years_of_Experience",
    "Training_Hours",
    "Learning_Progress",
    "Previous_Performance",
    "Attendance",
    "Job_Satisfaction"
] + SKILLS

CATEGORICAL_FEATURES = [
    "Department",
    "Current_Role",
    "Education"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "Workforce_Readiness"

CLASSES = [
    "READY",
    "ALMOST READY",
    "NEEDS RESKILLING",
    "CAREER TRANSITION SUGGESTED"
]

# --------------------------------------------------
# JOB ROLES AND REQUIRED SKILLS
# --------------------------------------------------

JOB_ROLES = {
    "Data Scientist": {
        "Python": 8,
        "Machine_Learning": 9,
        "Data_Analysis": 9,
        "Communication": 6,
        "Problem_Solving": 8,
        "Cloud_Computing": 6,
        "Cybersecurity": 3,
        "Leadership": 5,
        "Adaptability": 7,
        "Digital_Literacy": 8
    },

    "AI Engineer": {
        "Python": 9,
        "Machine_Learning": 9,
        "Data_Analysis": 7,
        "Communication": 6,
        "Problem_Solving": 9,
        "Cloud_Computing": 8,
        "Cybersecurity": 5,
        "Leadership": 5,
        "Adaptability": 8,
        "Digital_Literacy": 9
    },

    "Cybersecurity Analyst": {
        "Python": 6,
        "Machine_Learning": 4,
        "Data_Analysis": 6,
        "Communication": 7,
        "Problem_Solving": 9,
        "Cloud_Computing": 7,
        "Cybersecurity": 10,
        "Leadership": 5,
        "Adaptability": 8,
        "Digital_Literacy": 9
    },

    "Cloud Engineer": {
        "Python": 7,
        "Machine_Learning": 4,
        "Data_Analysis": 5,
        "Communication": 6,
        "Problem_Solving": 8,
        "Cloud_Computing": 10,
        "Cybersecurity": 8,
        "Leadership": 5,
        "Adaptability": 8,
        "Digital_Literacy": 9
    },

    "Project Manager": {
        "Python": 3,
        "Machine_Learning": 3,
        "Data_Analysis": 6,
        "Communication": 9,
        "Problem_Solving": 8,
        "Cloud_Computing": 4,
        "Cybersecurity": 4,
        "Leadership": 10,
        "Adaptability": 9,
        "Digital_Literacy": 8
    }
}

# --------------------------------------------------
# COURSE RECOMMENDATIONS
# --------------------------------------------------

COURSES = {
    "Python": [
        "Python Programming for Beginners",
        "Advanced Python Development"
    ],
    "Machine_Learning": [
        "Machine Learning Fundamentals",
        "Advanced Machine Learning"
    ],
    "Data_Analysis": [
        "Data Analysis with Python",
        "Advanced Data Analytics"
    ],
    "Communication": [
        "Effective Workplace Communication",
        "Business Communication Skills"
    ],
    "Problem_Solving": [
        "Critical Thinking and Problem Solving",
        "Analytical Thinking"
    ],
    "Cloud_Computing": [
        "Cloud Computing Fundamentals",
        "AWS and Azure Training"
    ],
    "Cybersecurity": [
        "Cybersecurity Fundamentals",
        "Ethical Hacking and Security"
    ],
    "Leadership": [
        "Leadership Development",
        "Team Management Skills"
    ],
    "Adaptability": [
        "Adaptability in the Workplace",
        "Change Management"
    ],
    "Digital_Literacy": [
        "Digital Skills Training",
        "Advanced Digital Technologies"
    ]
}

# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def create_database():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            Employee_ID TEXT PRIMARY KEY,
            Employee_Name TEXT,
            Department TEXT,
            Current_Role TEXT,
            Age INTEGER,
            Years_of_Experience REAL,
            Education TEXT,
            Training_Hours REAL,
            Learning_Progress REAL,
            Previous_Performance REAL,
            Attendance REAL,
            Job_Satisfaction REAL,
            Python REAL,
            Machine_Learning REAL,
            Data_Analysis REAL,
            Communication REAL,
            Problem_Solving REAL,
            Cloud_Computing REAL,
            Cybersecurity REAL,
            Leadership REAL,
            Adaptability REAL,
            Digital_Literacy REAL
        )
    """)

    conn.commit()
    conn.close()


def get_employees():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM employees", conn)
    conn.close()
    return df


def save_employee(data):
    conn = sqlite3.connect(DB_PATH)

    columns = list(data.keys())
    placeholders = ",".join(["?"] * len(columns))

    query = f"""
        INSERT OR REPLACE INTO employees
        ({",".join(columns)})
        VALUES ({placeholders})
    """

    conn.execute(query, list(data.values()))
    conn.commit()
    conn.close()


def delete_employee(employee_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "DELETE FROM employees WHERE Employee_ID = ?",
        (employee_id,)
    )
    conn.commit()
    conn.close()


# --------------------------------------------------
# DATASET GENERATION
# --------------------------------------------------

def generate_dataset():
    np.random.seed(42)

    departments = [
        "IT",
        "HR",
        "Finance",
        "Marketing",
        "Operations",
        "Sales"
    ]

    roles = [
        "Software Developer",
        "HR Executive",
        "Data Analyst",
        "Business Analyst",
        "Marketing Executive",
        "System Administrator"
    ]

    education = [
        "Diploma",
        "Graduate",
        "Postgraduate",
        "Doctorate"
    ]

    records = []

    for i in range(2200):

        record = {
            "Employee_ID": f"EMP{i+1:04d}",
            "Employee_Name": f"Employee {i+1}",
            "Department": np.random.choice(departments),
            "Current_Role": np.random.choice(roles),
            "Age": np.random.randint(21, 61),
            "Years_of_Experience": round(
                np.random.uniform(0, 30), 1
            ),
            "Education": np.random.choice(education),
            "Training_Hours": round(
                np.random.uniform(0, 100), 1
            ),
            "Learning_Progress": round(
                np.random.uniform(0, 100), 1
            ),
            "Previous_Performance": round(
                np.random.uniform(30, 100), 1
            ),
            "Attendance": round(
                np.random.uniform(50, 100), 1
            ),
            "Job_Satisfaction": np.random.randint(1, 11)
        }

        for skill in SKILLS:
            record[skill] = np.random.randint(1, 11)

        skill_average = np.mean(
            [record[s] for s in SKILLS]
        )

        score = (
            skill_average * 0.45
            + record["Training_Hours"] * 0.08
            + record["Learning_Progress"] * 0.08
            + record["Previous_Performance"] * 0.15
            + record["Attendance"] * 0.10
            + record["Job_Satisfaction"] * 1.0
            + record["Years_of_Experience"] * 0.15
            + np.random.normal(0, 5)
        )

        if score >= 65:
            readiness = "READY"
        elif score >= 55:
            readiness = "ALMOST READY"
        elif score >= 45:
            readiness = "NEEDS RESKILLING"
        else:
            readiness = "CAREER TRANSITION SUGGESTED"

        record[TARGET] = readiness
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(DATA_PATH, index=False)

    return df


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

def train_model(df):

    X = df[FEATURES]
    y = df[TARGET]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=7
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = []
    best_score = -1
    best_model = None
    best_name = None

    for name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="f1_macro"
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results.append({
            "Model": name,
            "Accuracy": round(accuracy * 100, 2),
            "Cross Validation F1": round(
                cv_scores.mean(), 3
            )
        })

        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_model = pipeline
            best_name = name

    joblib.dump(best_model, MODEL_PATH)

    predictions = best_model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    return (
        pd.DataFrame(results),
        best_name,
        report
    )


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# --------------------------------------------------
# SKILL GAP ANALYSIS
# --------------------------------------------------

def calculate_skill_gap(employee, target_role):

    requirements = JOB_ROLES[target_role]

    gaps = []

    for skill, required in requirements.items():

        current = float(employee.get(skill, 0))
        gap = max(0, required - current)

        gaps.append({
            "Skill": skill.replace("_", " "),
            "Current Level": current,
            "Required Level": required,
            "Skill Gap": gap
        })

    return pd.DataFrame(gaps)


# --------------------------------------------------
# COURSE RECOMMENDATION
# --------------------------------------------------

def recommend_courses(gap_df):

    recommendations = []

    for _, row in gap_df.iterrows():

        skill = row["Skill"].replace(" ", "_")
        gap = row["Skill Gap"]

        if gap > 0 and skill in COURSES:

            for course in COURSES[skill]:

                recommendations.append({
                    "Missing Skill": row["Skill"],
                    "Skill Gap": gap,
                    "Recommended Course": course
                })

    return pd.DataFrame(recommendations)


# --------------------------------------------------
# CAREER COMPATIBILITY
# --------------------------------------------------

def career_recommendations(employee):

    results = []

    for role, requirements in JOB_ROLES.items():

        scores = []

        for skill, required in requirements.items():

            current = float(employee.get(skill, 0))
            scores.append(min(current / required, 1))

        compatibility = np.mean(scores) * 100

        results.append({
            "Job Role": role,
            "Compatibility": round(compatibility, 2)
        })

    return pd.DataFrame(results).sort_values(
        "Compatibility",
        ascending=False
    )


# --------------------------------------------------
# READINESS PREDICTION
# --------------------------------------------------

def predict_readiness(employee, model):

    input_df = pd.DataFrame([
        {
            feature: employee.get(feature)
            for feature in FEATURES
        }
    ])

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    probability_df = pd.DataFrame({
        "Readiness Category": model.classes_,
        "Probability": probabilities
    })

    return prediction, probability_df


# --------------------------------------------------
# K-MEANS EMPLOYEE SEGMENTATION
# --------------------------------------------------

def employee_segmentation(df):

    clustering_features = [
        "Training_Hours",
        "Learning_Progress",
        "Previous_Performance"
    ] + SKILLS

    data = df[clustering_features].fillna(0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    df = df.copy()
    df["Employee_Segment"] = kmeans.fit_predict(
        scaled_data
    )

    return df


# --------------------------------------------------
# STREAMLIT APPLICATION
# --------------------------------------------------

create_database()

if not os.path.exists(DATA_PATH):
    dataset = generate_dataset()
else:
    dataset = pd.read_csv(DATA_PATH)

model = load_model()

if model is None:
    with st.spinner("Training machine learning models..."):
        model_results, best_model_name, model_report = train_model(
            dataset
        )
    model = load_model()
else:
    model_results = None
    best_model_name = "Saved Model"
    model_report = None

st.sidebar.title("🎓 Employee Reskilling")
st.sidebar.write("Corporate Workforce Transition Framework")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Employee Prediction",
        "Skill Gap Analysis",
        "Course Recommendations",
        "Career Transition",
        "Employee Management",
        "Workforce Analytics",
        "Model Performance"
    ]
)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("🎓 Employee Reskilling Dashboard")

    employees = get_employees()

    total_employees = len(employees)

    if total_employees == 0:
        total_employees = len(dataset)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Employees", total_employees)

    col2.metric(
        "Departments",
        dataset["Department"].nunique()
    )

    col3.metric(
        "Average Training Hours",
        round(dataset["Training_Hours"].mean(), 2)
    )

    col4.metric(
        "Average Learning Progress",
        f"{dataset['Learning_Progress'].mean():.1f}%"
    )

    st.subheader("Workforce Readiness Distribution")

    st.bar_chart(
        dataset[TARGET].value_counts()
    )

    st.subheader("Department-wise Employee Distribution")

    st.bar_chart(
        dataset["Department"].value_counts()
    )

    st.subheader("Employee Dataset")

    st.dataframe(
        dataset.head(100),
        use_container_width=True
    )


# --------------------------------------------------
# EMPLOYEE PREDICTION
# --------------------------------------------------

elif page == "Employee Prediction":

    st.title("🔍 Workforce Readiness Prediction")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            employee_id = st.text_input(
                "Employee ID",
                "EMP_NEW"
            )

            employee_name = st.text_input(
                "Employee Name"
            )

            department = st.selectbox(
                "Department",
                [
                    "IT",
                    "HR",
                    "Finance",
                    "Marketing",
                    "Operations",
                    "Sales"
                ]
            )

            current_role = st.selectbox(
                "Current Role",
                [
                    "Software Developer",
                    "HR Executive",
                    "Data Analyst",
                    "Business Analyst",
                    "Marketing Executive",
                    "System Administrator"
                ]
            )

            age = st.number_input(
                "Age",
                18,
                65,
                25
            )

            experience = st.number_input(
                "Years of Experience",
                0.0,
                45.0,
                2.0
            )

            education = st.selectbox(
                "Education",
                [
                    "Diploma",
                    "Graduate",
                    "Postgraduate",
                    "Doctorate"
                ]
            )

        with col2:

            training_hours = st.slider(
                "Training Hours",
                0.0,
                200.0,
                30.0
            )

            learning_progress = st.slider(
                "Learning Progress (%)",
                0.0,
                100.0,
                50.0
            )

            previous_performance = st.slider(
                "Previous Performance",
                0.0,
                100.0,
                70.0
            )

            attendance = st.slider(
                "Attendance (%)",
                0.0,
                100.0,
                85.0
            )

            satisfaction = st.slider(
                "Job Satisfaction",
                1,
                10,
                7
            )

        st.subheader("Employee Skill Levels")

        skills_data = {}

        columns = st.columns(2)

        for index, skill in enumerate(SKILLS):

            with columns[index % 2]:

                skills_data[skill] = st.slider(
                    skill.replace("_", " "),
                    0,
                    10,
                    5,
                    key=f"prediction_{skill}"
                )

        submitted = st.form_submit_button(
            "Predict Workforce Readiness"
        )

    if submitted:

        employee = {
            "Age": age,
            "Years_of_Experience": experience,
            "Education": education,
            "Department": department,
            "Current_Role": current_role,
            "Training_Hours": training_hours,
            "Learning_Progress": learning_progress,
            "Previous_Performance": previous_performance,
            "Attendance": attendance,
            "Job_Satisfaction": satisfaction,
            **skills_data
        }

        prediction, probabilities = predict_readiness(
            employee,
            model
        )

        st.success(
            f"Predicted Workforce Readiness: {prediction}"
        )

        st.subheader("Prediction Probabilities")

        st.dataframe(
            probabilities,
            use_container_width=True
        )

        st.bar_chart(
            probabilities.set_index(
                "Readiness Category"
            )
        )

        employee_record = {
            "Employee_ID": employee_id,
            "Employee_Name": employee_name,
            **employee
        }

        if st.button("Save Employee"):

            save_employee(employee_record)

            st.success("Employee saved successfully!")


# --------------------------------------------------
# SKILL GAP ANALYSIS
# --------------------------------------------------

elif page == "Skill Gap Analysis":

    st.title("📊 Skill Gap Analysis")

    target_role = st.selectbox(
        "Select Target Job Role",
        list(JOB_ROLES.keys())
    )

    st.subheader("Enter Current Employee Skills")

    employee_skills = {}

    columns = st.columns(2)

    for index, skill in enumerate(SKILLS):

        with columns[index % 2]:

            employee_skills[skill] = st.slider(
                skill.replace("_", " "),
                0,
                10,
                5,
                key=f"gap_{skill}"
            )

    if st.button("Analyze Skill Gaps"):

        gap_df = calculate_skill_gap(
            employee_skills,
            target_role
        )

        st.subheader("Skill Gap Report")

        st.dataframe(
            gap_df,
            use_container_width=True
        )

        st.bar_chart(
            gap_df.set_index("Skill")[
                ["Current Level", "Required Level"]
            ]
        )

        missing_skills = gap_df[
            gap_df["Skill Gap"] > 0
        ]

        st.metric(
            "Skills Requiring Improvement",
            len(missing_skills)
        )


# --------------------------------------------------
# COURSE RECOMMENDATIONS
# --------------------------------------------------

elif page == "Course Recommendations":

    st.title("📚 Personalized Training Recommendations")

    target_role = st.selectbox(
        "Choose Target Role",
        list(JOB_ROLES.keys())
    )

    employee_skills = {}

    for skill in SKILLS:

        employee_skills[skill] = st.slider(
            skill.replace("_", " "),
            0,
            10,
            5,
            key=f"course_{skill}"
        )

    if st.button("Recommend Courses"):

        gap_df = calculate_skill_gap(
            employee_skills,
            target_role
        )

        recommendations = recommend_courses(
            gap_df
        )

        if recommendations.empty:

            st.success(
                "Employee has all required skills for this role."
            )

        else:

            st.subheader("Recommended Courses")

            st.dataframe(
                recommendations,
                use_container_width=True
            )


# --------------------------------------------------
# CAREER TRANSITION
# --------------------------------------------------

elif page == "Career Transition":

    st.title("🚀 Career Transition Recommendations")

    employee_skills = {}

    for skill in SKILLS:

        employee_skills[skill] = st.slider(
            skill.replace("_", " "),
            0,
            10,
            5,
            key=f"career_{skill}"
        )

    if st.button("Find Suitable Careers"):

        recommendations = career_recommendations(
            employee_skills
        )

        st.subheader("Career Compatibility")

        st.dataframe(
            recommendations,
            use_container_width=True
        )

        st.bar_chart(
            recommendations.set_index("Job Role")
        )

        best_role = recommendations.iloc[0]

        st.success(
            f"Best matching role: {best_role['Job Role']} "
            f"({best_role['Compatibility']}% compatibility)"
        )


# --------------------------------------------------
# EMPLOYEE MANAGEMENT
# --------------------------------------------------

elif page == "Employee Management":

    st.title("👥 Employee Management")

    employees = get_employees()

    st.subheader("Registered Employees")

    st.dataframe(
        employees,
        use_container_width=True
    )

    st.subheader("Delete Employee")

    if not employees.empty:

        selected_id = st.selectbox(
            "Select Employee ID",
            employees["Employee_ID"].tolist()
        )

        confirm = st.checkbox(
            "Confirm employee deletion"
        )

        if st.button("Delete Employee"):

            if confirm:
                delete_employee(selected_id)
                st.success("Employee deleted successfully!")
                st.rerun()
            else:
                st.warning(
                    "Please confirm before deleting."
                )

    st.subheader("Import Employee CSV")

    uploaded_file = st.file_uploader(
        "Upload employee CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_df = pd.read_csv(uploaded_file)

        st.dataframe(uploaded_df.head())

        if st.button("Import Employees"):

            imported = 0

            for _, row in uploaded_df.iterrows():

                employee = row.to_dict()

                required_columns = [
                    "Employee_ID",
                    "Employee_Name"
                ]

                if all(
                    column in employee
                    for column in required_columns
                ):

                    save_employee(employee)
                    imported += 1

            st.success(
                f"{imported} employees imported successfully."
            )


# --------------------------------------------------
# WORKFORCE ANALYTICS
# --------------------------------------------------

elif page == "Workforce Analytics":

    st.title("📈 Workforce Analytics")

    segmented_df = employee_segmentation(dataset)

    st.subheader("Employee Skill Distribution")

    skill_means = dataset[SKILLS].mean().sort_values(
        ascending=False
    )

    st.bar_chart(skill_means)

    st.subheader("Training Hours vs Learning Progress")

    st.scatter_chart(
        dataset,
        x="Training_Hours",
        y="Learning_Progress",
        color="Department"
    )

    st.subheader("Employee Clusters")

    st.bar_chart(
        segmented_df["Employee_Segment"].value_counts()
    )

    st.subheader("Department-wise Training")

    department_training = dataset.groupby(
        "Department"
    )["Training_Hours"].mean()

    st.bar_chart(department_training)

    st.subheader("Readiness by Department")

    readiness_department = pd.crosstab(
        dataset["Department"],
        dataset[TARGET]
    )

    st.dataframe(
        readiness_department,
        use_container_width=True
    )

    st.download_button(
        "Download Workforce Analytics CSV",
        segmented_df.to_csv(index=False),
        file_name="workforce_analytics.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    if model_results is not None:

        st.subheader("Model Comparison")

        st.dataframe(
            model_results,
            use_container_width=True
        )

        st.bar_chart(
            model_results.set_index("Model")[
                "Accuracy"
            ]
        )

        st.success(
            f"Selected Model: {best_model_name}"
        )

        st.subheader("Classification Report")

        st.json(model_report)

    else:

        st.info(
            "The saved machine learning model is loaded."
        )

        st.write(
            "To compare models again, delete "
            "models/readiness_model.pkl and restart the application."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(
    "AI-Based Employee Reskilling Framework"
)
st.sidebar.caption(
    "Machine Learning | Workforce Analytics | Career Transition"
)