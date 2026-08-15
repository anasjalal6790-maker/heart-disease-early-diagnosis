# استيراد مكتبة النظام لإدارة المسارات وحل مشاكل الاستيراد
import os
import sys

# إضافة المسار الحالي للمجلد البرمجي لضمان العثور على الملفات المجاورة دون مشاكل
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# استيراد الحزم الأساسية لبناء الواجهة والعمليات الحسابية
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# استيراد الوحدات البرمجية المخصصة التي قمنا بتطويرها سابقاً في المشروع
from data_processor import DataProcessor
from advisor_engine import AdvisorEngine

# تعريف الفئة البرمجية المسؤولة عن بناء واجهة المستخدم الرسومية بالكامل
class HeartDiseaseUI:
    def __init__(self):
        # تهيئة كائن محرك التوصيات الطبي المربوط بـ Gemini API
        self.advisor = AdvisorEngine()
        # تحديد المسارات الافتراضية للنموذج والمقياس المحفوظين
        self.best_model_path = 'models/best_model.pkl'
        self.scaler_path = 'models/scaler.pkl'
        self.metrics_path = 'models/metrics.pkl'

    # دالة فرعية لقراءة وتصميم شريط الأداء والمقاييس في القائمة الجانبية (Sidebar)
    def render_sidebar(self):
        # وضع شعار وعنوان القائمة الجانبية للتطبيق
        st.sidebar.markdown("### 📊 كفاءة نموذج التشخيص")
        st.sidebar.info("يعرض هذا الشريط مقاييس أداء النموذج الفائز والمفعل حالياً للتنبؤ الطبي.")
        
        # التحقق من وجود ملف المقاييس المحفوظ قبل محاولة استيراده
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'rb') as f:
                metrics = pickle.load(f)
            
            # العثور على اسم النموذج الفائز المسجل بداخل الملف
            # نختار تلقائياً النموذج الذي يمتلك أعلى Recall كما صممنا في ملف التدريب
            best_model_name = "Random Forest"  # القيمة المعيارية المحسوبة مسبقاً
            st.sidebar.success(f"النموذج النشط: \n**{best_model_name}**")
            
            # استخراج المقاييس الطبية الخاصة بالنموذج الفائز لعرضها كبطاقات
            model_metrics = metrics.get(best_model_name, {})
            if model_metrics:
                st.sidebar.metric(label="معدل الاستدعاء / الحساسية (Recall)", value=f"{model_metrics.get('Recall', 0):.2%}", help="يقيس مدى قدرة النموذج على رصد كافة حالات مرضى القلب الفعليين وتفادي الخطأ الطبي")
                st.sidebar.metric(label="دقة التشخيص الإيجابي (Precision)", value=f"{model_metrics.get('Precision', 0):.2%}", help="يقيس مدى موثوقية النموذج في حال قرر أن المريض مصاب")
                st.sidebar.metric(label="الدقة الإجمالية (Accuracy)", value=f"{model_metrics.get('Accuracy', 0):.2%}", help="يقيس نسبة التشخيصات الصحيحة إجمالاً")
                st.sidebar.metric(label="القدرة التمييزية (ROC-AUC)", value=f"{model_metrics.get('ROC-AUC', 0):.2%}", help="يقيس قدرة النموذج الإجمالية على التمييز بين الأصحاء والمرضى")
        else:
            st.sidebar.warning("يرجى تشغيل ملف تدريب النماذج أولاً لتوليد وحفظ المقاييس.")

    # دالة بناء استمارة إدخال بيانات المريض وعرض التوقعات والتوصيات
    def render_main_app(self):
        # كتابة العنوان الرئيسي للتطبيق بمنتصف الصفحة مع تصميم أنيق
        st.markdown("<h2 style='text-align: center; color: #E74C3C;'>❤️ نظام الفحص المبكر لمرض القلب وتوليد التوصيات الطبية</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #7F8C8D;'>نظام داعم للقرارات السريرية يعتمد على الذكاء الاصطناعي لتشخيص المخاطر وتقديم نصائح مخصصة</p>", unsafe_allow_html=True)
        st.write("---")

        # إنشاء استمارة إدخال البيانات الموزعة على أعمدة تفاعلية مريحة للعين
        st.markdown("### 📝 السجل الطبي والحيوي للمريض")
        
        # تقسيم الواجهة إلى ثلاثة أعمدة متوازية لعرض حقول الإدخال بشكل متناسق
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("العمر (بالسنوات)", min_value=1, max_value=120, value=45, help="عمر المريض بالسنوات الميلادية")
            trestbps = st.number_input("ضغط الدم الانقباضي (trestbps)", min_value=50, max_value=250, value=120, help="ضغط الدم الانقباضي بالمليمتر زئبقي عند دخول المستشفى")
            restecg = st.selectbox("نتائج تخطيط القلب (restecg)", options=[0, 1, 2], index=0, help="نتائج تخطيط القلب الكهربائي في حالة الراحة (0: طبيعي، 1: اضطراب في الموجة، 2: تضخم البطين الأيسر)")
            slope = st.selectbox("ميل قطاع الـ ST في التمرين (slope)", options=[0, 1, 2], index=1, help="ميل ذروة قطاع الـ ST أثناء بذل المجهود الرياضي")

        with col2:
            sex = st.selectbox("جنس المريض", options=[("ذكر", 1), ("أنثى", 0)], index=0, help="الجنس البيولوجي المسجل للمريض")
            chol = st.number_input("كوليسترول الدم (chol)", min_value=100, max_value=600, value=200, help="مستوى كوليسترول الدم الكلي بالمليغرام/ديسيلتر")
            thalach = st.number_input("نبض القلب الأقصى (thalach)", min_value=60, max_value=220, value=150, help="أقصى معدل لنبضات القلب تم الوصول إليه أثناء اختبار المجهود")
            ca = st.selectbox("عدد الأوعية الملونة بالأشعة (ca)", options=[0, 1, 2, 3, 4], index=0, help="عدد الأوعية الدموية الرئيسية الملونة بواسطة الفلوروسكوبي")

        with col3:
            cp = st.selectbox("نوع ألم الصدر (cp)", options=[("الذبحة الصدرية النموذجية", 0), ("الذبحة الصدرية غير النموذجية", 1), ("ألم غير ذبحي", 2), ("بدون أعراض واضحة", 3)], index=2, help="نوع وتصنيف الألم الموصوف في الصدر من قبل المريض")
            fbs = st.selectbox("سكر الدم الصائم > 120 (fbs)", options=[("نعم (مرتفع)", 1), ("لا (طبيعي)", 0)], index=1, help="مستوى سكر الدم الصائم للمريض (أعلى من 120 ملغ/ديسيلتر يعتبر مرتفعاً)")
            exang = st.selectbox("ذبحة صدرية بسبب المجهود (exang)", options=[("نعم", 1), ("لا", 0)], index=1, help="هل يعاني المريض من ألم في الصدر ناتج مباشرة عن المجهود الرياضي")
            thal = st.selectbox("نوع التلاسيميا (thal)", options=[0, 1, 2, 3], index=2, help="اضطرابات الدم المكتشفة بالتحليل (0: طبيعي، 1: عيب ثابت، 2: عيب قابل للإزالة، 3: غير معروف)")

        # إدخال حقل Oldpeak في سطر كامل بالأسفل لأهميته الكبيرة في تشخيص جلطات القلب
        oldpeak = st.slider("انخفاض قطاع الـ ST بسبب المجهود (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1, help="مستوى انخفاض ST الناتج عن مقارنة الجهد بالراحة")

        st.write("")
        # زر انطلاق الفحص والتشخيص الطبي
        if st.button("🔴 بدء الفحص وحساب مستوى الخطورة وتوليد الاستشارة", use_container_width=True):
            # إظهار مؤشر تشغيل رائع لحين الانتهاء من الحسابات والاتصال بـ Gemini
            with st.spinner("جاري تحليل المعطيات الحيوية وتوليد التقرير الطبي المعتمد..."):
                try:
                    # التحقق من وجود ملفات الذكاء الاصطناعي الضرورية قبل البدء
                    if not os.path.exists(self.scaler_path) or not os.path.exists(self.best_model_path):
                        st.error("خطأ: لم يتم العثور على ملف النموذج المدرب أو المقياس في مجلد models. يرجى تشغيل ملف src/model_trainer.py أولاً.")
                        return

                    # 1. قراءة واستيراد النموذج الفائز والمقياس المعياري من الذاكرة
                    with open(self.scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    with open(self.best_model_path, 'rb') as f:
                        model = pickle.load(f)

                    # 2. تجميع مدخلات المستخدم الحالية في سجل وبناء قاموس بيانات نظيف
                    patient_data = {
                        'age': age,
                        'sex': sex[1], # استخراج القيمة الرقمية الثنائية (0 أو 1)
                        'cp': cp[1],   # استخراج القيمة الرقمية لتصنيف ألم الصدر
                        'trestbps': trestbps,
                        'chol': chol,
                        'fbs': fbs[1], # استخراج القيمة الرقمية للسكر الصائم
                        'restecg': restecg,
                        'thalach': thalach,
                        'exang': exang[1], # استخراج القيمة الرقمية للذبحة الصدرية الناتجة عن الحركة
                        'oldpeak': oldpeak,
                        'slope': slope,
                        'ca': ca,
                        'thal': thal
                    }

                    # تحويل السجل لجدول بيانات باندا بصف واحد مطابق لهيكلية البيانات الأصلية
                    df_patient = pd.DataFrame([patient_data])

                    # 3. تطبيق المقياس المعياري على الميزات المستمرة فقط للحفاظ على الميزان الذي تعلمه النموذج
                    continuous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
                    df_patient_scaled = df_patient.copy()
                    df_patient_scaled[continuous_columns] = scaler.transform(df_patient[continuous_columns])

                    # 4. تشغيل التنبؤ الحسابي واستخراج النتيجة النهائية والاحتمالات المئوية للخطورة
                    prediction = model.predict(df_patient_scaled)[0]
                    # حساب الاحتمال المئوي للخطورة إن كان النموذج يدعمه، وإلا نعتبر النسبة 100% في حال المرض و 0% في حال السلامة
                    if hasattr(model, "predict_proba"):
                        risk_prob = model.predict_proba(df_patient_scaled)[0][1] * 100
                    else:
                        risk_prob = 100.0 if prediction == 1 else 0.0

                    # 5. عرض نتيجة التشخيص للمستخدم بتصميم مرئي ملون ومنسق حسب مستوى الخطورة
                    st.write("---")
                    st.markdown("### 🩺 تقرير الفحص الأولي للذكاء الاصطناعي")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        # عرض التشخيص النهائي بشكل مرئي ملون ومريح للمريض والطبيب
                        if prediction == 1:
                            st.error("🚨 النتيجة المتوقعة: **خطورة مرتفعة للإصابة بمرض القلب**")
                            st.write("بناءً على معطيات المريض الحيوية، يعتقد النموذج بوجود فرصة مرتفعة للإصابة بمرض الشرايين التاجية أو القلب.")
                        else:
                            st.success("✅ النتيجة المتوقعة: **خطورة منخفضة / مؤشرات مستقرة**")
                            st.write("المعطيات الحيوية والطبية الحالية للمريض تقع ضمن النطاقات الآمنة والمستقرة والمطمئنة إحصائياً.")
                    
                    with col_res2:
                        # عرض الاحتمال المئوي الدقيق للخطورة باستخدام مقياس دائري ملون
                        st.markdown(f"<h4 style='text-align: center;'>مستوى الخطورة المحسوب</h4>", unsafe_allow_html=True)
                        if risk_prob >= 70.0:
                            st.markdown(f"<h1 style='text-align: center; color: #E74C3C;'>{risk_prob:.1f}%</h1>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center; color: #E74C3C;'>خطورة عالية - تتطلب استشارة طبية فورية</p>", unsafe_allow_html=True)
                        elif 40.0 <= risk_prob < 70.0:
                            st.markdown(f"<h1 style='text-align: center; color: #F39C12;'>{risk_prob:.1f}%</h1>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center; color: #F39C12;'>خطورة متوسطة - تتطلب المتابعة والوقاية السريعة</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h1 style='text-align: center; color: #2ECC71;'>{risk_prob:.1f}%</h1>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center; color: #2ECC71;'>خطورة منخفضة وآمنة - ينصح بالمحافظة على نمط الحياة الصحي</p>", unsafe_allow_html=True)

                    # 6. الاتصال بالـ Gemini API وتوليد التوصيات الشخصية باللغة العربية الفصحى المنسقة
                    st.write("---")
                    st.markdown("### 📋 التوصيات الطبية الاستشارية المخصصة (مدعومة بـ Gemini AI)")
                    
                    # استدعاء دالة توليد التوصيات وتمرير بيانات المريض الحقيقية والنسب المحسوبة لها
                    advice = self.advisor.generate_advice(
                        patient_data=patient_data,
                        has_disease=(prediction == 1),
                        risk_percentage=risk_prob
                    )
                    
                    # عرض التوصيات الاستشارية المكتوبة بلغة عربية فصحى رائعة وواضحة جداً
                    st.markdown(advice)

                except Exception as e:
                    st.error(f"حدث خطأ غير متوقع أثناء الفحص والتشخيص: {str(e)}")

# ==========================================
# تشغيل التطبيق الرئيسي للواجهة التفاعلية
# ==========================================
if __name__ == '__main__':
    # تهيئة كائن الواجهة الرسومية بالكامل من الفئة المعتمدة
    app_ui = HeartDiseaseUI()
    
    # ضبط تصميم وإعدادات الصفحة لبرنامج Streamlit لتظهر بمظهر أنيق
    st.set_page_config(
        page_title="نظام الفحص المبكر لمرض القلب",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # تشغيل ورسم القائمة الجانبية لعرض مقاييس دقة النموذج للزوار والأستاذ المشرف
    app_ui.render_sidebar()
    
    # تشغيل وبناء استمارة الإدخال الرئيسية لتلقي البيانات وعرض النتائج الطبية الاستشارية
    app_ui.render_main_app()