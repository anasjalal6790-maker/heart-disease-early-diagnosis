# استيراد مكتبة النظام لإدارة المتغيرات والتحقق من المجلدات
import os
# استيراد مكتبة دوت إنف لقراءة المفاتيح السرية من ملف دوت إنف الآمن
from dotenv import load_dotenv
# استيراد حزمة جوجل جين إيه آي الرسمية للتعامل مع جيميناي
from google import genai

# تعريف فئة محرك التوصيات الطبي المربوط بـ Gemini API
class AdvisorEngine:
    """
    Class responsible for integrating with Google Gemini API
    to generate highly personalized Arabic clinical recommendations.
    Adheres to the Single Responsibility Principle (SRP).
    """
    # دالة البناء وتحميل المفتاح السري بالمسار المطلق وتهيئة العميل
    # دالة البناء وتحميل المفتاح السري بالمسار المطلق والبحث الهجين المتقدم
    def __init__(self):
        # حساب المسار الفيزيائي المطلق لملف .env في المجلد الرئيسي للمشروع محلياً
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        dotenv_path = os.path.join(project_root, '.env')
            
            # تحميل ملف البيئة من المسار المطلق بشكل آمن
        load_dotenv(dotenv_path=dotenv_path)
            
            # تهيئة متغير المفتاح السري فارغاً لبدء البحث الهجين المطور
        self.api_key = None
            
            # 1. محاولة جلب المفتاح السري من أسرار منصة Streamlit (وهي الطريقة المضمونة والآمنة سحابياً)
        try:
            import streamlit as st
            if "GEMINI_API_KEY" in st.secrets:
                self.api_key = st.secrets["GEMINI_API_KEY"]
                print("[*] Gemini API Key loaded from Streamlit Cloud Secrets.")
        except Exception:
                # تجاهل الخطأ في حال تشغيل الملف بشكل مستقل خارج بيئة Streamlit
            pass
            
            # 2. إذا لم نعثر عليه سحابياً، نقوم بقراءته من ملف البيئة المحلي (.env) أو الطرفية
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if self.api_key:
                print("[*] Gemini API Key loaded from Local .env / Environment.")
            
        self.client = None
            
            # التحقق النهائي من وجود المفتاح وتهيئة العميل
        if not self.api_key:
            print("[!] Warning: GEMINI_API_KEY not found in environment variables. Set it in .env file or Streamlit Secrets.")
        else:
                # تهيئة عميل جوجل جين إيه آي الرسمي بالكامل للاتصال بالنماذج
            self.client = genai.Client(api_key=self.api_key)
            print("[*] Gemini API Client initialized successfully.")

    # دالة توليد التوصيات الطبية الاستشارية باللغة العربية بناءً على معطيات المريض
    def generate_advice(self, patient_data, has_disease, risk_percentage):
        # محاولة تهيئة العميل في حال إرسال المفتاح لاحقاً بشكل ديناميكي
        if not self.client:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                return "خطأ: لم يتم إعداد مفتاح API الخاص بـ Gemini في ملف .env. يرجى توفيره لتوليد التوصيات."
            self.client = genai.Client(api_key=self.api_key)

        # تحويل القيم الرقمية إلى مسميات طبية واضحة بالإنجليزية لتسهيل فهم الموديل لها
        gender_str = "Male" if patient_data['sex'] == 1 else "Female"
        chest_pain_types = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
        cp_str = chest_pain_types.get(patient_data['cp'], "Unknown")
        
        # تحديد حالة السكر والذبحة الناتجة عن المجهود بصيغة طبية مفهومة لنموذج الذكاء
        fbs_str = "Fasting Blood Sugar > 120 mg/dl (Elevated)" if patient_data['fbs'] == 1 else "Normal Fasting Blood Sugar (< 120 mg/dl)"
        exang_str = "Experienced Exercise-Induced Angina" if patient_data['exang'] == 1 else "No Exercise-Induced Angina"
        
        # بناء ملف المريض السريري الكامل لإرساله مع المحث الاستشاري
        clinical_profile = f"""
        [Patient Clinical Summary]
        - Age: {patient_data['age']} years old
        - Gender: {gender_str}
        - Chest Pain Type: {cp_str} (Numeric Code: {patient_data['cp']})
        - Resting Blood Pressure (trestbps): {patient_data['trestbps']} mmHg
        - Serum Cholesterol (chol): {patient_data['chol']} mg/dl
        - Fasting Blood Sugar Status (fbs): {fbs_str}
        - Resting Electrocardiographic Results (restecg): {patient_data['restecg']}
        - Maximum Heart Rate Achieved (thalach): {patient_data['thalach']} bpm
        - Exercise-Induced Angina (exang): {exang_str}
        - ST Depression (oldpeak): {patient_data['oldpeak']}
        - Slope of Peak Exercise ST Segment (slope): {patient_data['slope']}
        - Number of Major Vessels Colored by Fluoroscopy (ca): {patient_data['ca']}
        - Thalassemia Type (thal): {patient_data['thal']}
        
        [AI ML Diagnostic Assessment]
        - Risk Classification: {"HIGH RISK OF HEART DISEASE" if has_disease else "LOW/NORMAL RISK OF HEART DISEASE"}
        - ML Model Prediction Confidence: {risk_percentage:.2f}%
        """

        # صياغة محث النظام الموجه جينياً لجعل Gemini يعمل كطبيب قلب استشاري باللغة العربية
        system_prompt = """
        You are an elite, compassionate clinical cardiologist and expert preventive medicine specialist.
        Your goal is to analyze the patient's diagnostic profile alongside the ML risk score and provide a highly personalized, empathetic, and comprehensive clinical consultation.
        
        Your output MUST be written completely in professional and clear Arabic (اللغة العربية الفصحى) with Markdown styling.
        
        You must structure your consultation into exactly these 4 key pillars:
        
        1) Customized Dietary Advice & Nutrition Plans (التغذية والنظام الغذائي المخصص)
           - Provide tailored nutrition guidelines based on their BP (trestbps), cholesterol (chol), sugar (fbs), and overall risk level.
           
        2) Lifestyle Modifications, Physical Activity, & Exercise (تعديل نمط الحياة والنشاط البدني)
           - Suggest safe, specific physical exercises or adjustments, considering their age, maximum heart rate (thalach), and exercise-induced angina status (exang).
           
        3) Medical Follow-ups, Urgency Level, & Next Steps (المتابعة الطبية، مستوى الاستعجال، والخطوات القادمة)
           - Explicitly define the urgency level (عالية - متوسطة - روتينية) based on the risk score and indicators.
           - Detail what diagnostic investigations (e.g., Echo, Stress ECG, Holter monitor) the patient should request during their next consultation.
           
        4) Long-term Preventive Care (الرعاية الوقائية على المدى الطويل)
           - Outline actionable strategies for chronic risk factor management, stress reduction, and vital signs monitoring.
        
        Strict Safety Disclaimer:
        You must include a clear, professional medical disclaimer in Arabic at the end. Clearly state that this is an AI-powered educational screening recommendation tool and does not substitute for an in-person, professional diagnosis and treatment plan from a licensed cardiologist.
        """

        # محاولة الاتصال بخوادم جوجل لإرجاع الاستشارة الطبية المنسقة بالكامل
        try:
            print("[*] Requesting medical consultation from Gemini API...")
            response = self.client.models.generate_content(
                model='gemini-3.5-flash', # تم الترقية للموديل الأحدث والأسرع المعتمد لعام 2026
                contents=f"{system_prompt}\n\nAnalyzing clinical profile:\n{clinical_profile}"
            )
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini API: {str(e)}"

# الشرط الرئيسي لتشغيل فحص الاتصال بـ Gemini بشكل مستقل ومباشر
if __name__ == '__main__':
    print("--- Starting Advisor Engine Validation ---")
    
    # تجميع معطيات مريض وهمي عالي الخطورة لإرسالها كفحص فني للاتصال
    mock_patient_data = {
        'age': 58,
        'sex': 1,
        'cp': 2,
        'trestbps': 140,
        'chol': 289,
        'fbs': 1,
        'restecg': 0,
        'thalach': 111,
        'exang': 1,
        'oldpeak': 2.8,
        'slope': 1,
        'ca': 2,
        'thal': 3
    }
    
    # تهيئة كائن المحرك وتشغيل توليد الاستشارة للاختبار الفني
    engine = AdvisorEngine()
    advice = engine.generate_advice(mock_patient_data, has_disease=True, risk_percentage=88.5)
    
    print("\n[+] Gemini Response Generated:\n")
    print(advice)
    print("\n--- Validation finished ---")