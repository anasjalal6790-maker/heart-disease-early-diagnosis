# استيراد مكتبة النظام لإدارة المجلدات والمسارات البرمجية
import os
# استيراد مكتبة بيكل لحفظ وتصدير كائنات بايثون بصيغة ثنائية
import pickle
# استيراد مكتبة نامباي للعمليات الحسابية وإدارة المصفوفات الرقمية
import numpy as np
# استيراد خوارزمية الانحدار اللوجستي للتصنيف الثنائي
from sklearn.linear_model import LogisticRegression
# استيراد خوارزمية الغابة العشوائية للتصنيف التجميعي
from sklearn.ensemble import RandomForestClassifier
# استيراد خوارزمية إكس جي بوست المتطورة لتعزيز التدرج
from xgboost import XGBClassifier
# استيراد الشبكة العصبية الاصطناعية متعددة الطبقات
from sklearn.neural_network import MLPClassifier
# استيراد دوال قياس الأداء الطبي المعتمدة من مكتبة المقاييس
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# تعريف فئة تدريب النماذج المسؤولة عن تدريب وتقييم وتصدير النماذج
class ModelTrainer:
    """
    Class responsible for training, evaluating, and exporting the machine learning models.
    Adheres to the Single Responsibility Principle (SRP).
    """
    # دالة البناء وتحديد مجلد حفظ النماذج الافتراضي باسم models
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        # إنشاء قاموس يحتوي على النماذج الأربعة وتعريف معاملاتها الابتدائية
        self.models = {
            # تهيئة نموذج الانحدار اللوجستي بـ 1000 جولة تدريب وبدء عشوائي ثابت
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            # تهيئة نموذج الغابة العشوائية بـ 100 شجرة قرار وبدء عشوائي ثابت
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            # تهيئة نموذج إكس جي بوست بـ 100 شجرة وتحديد دالة خسارة الاحتمالات
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            # تهيئة كائن الشبكة العصبية الاصطناعية متعددة الطبقات بالمعاملات المحددة
            'Artificial Neural Network (ANN)': MLPClassifier(
                # تحديد طبقتين مخفيتين تحتويان على 64 و 32 عصبوناً بالترتيب
                hidden_layer_sizes=(64, 32),
                # تحديد دالة التنشيط ريلو للطبقات المخفية لتسريع معالجة البيانات
                activation='relu',
                # تحديد المحسن آدم لتحديث أوزان الخلايا العصبية تلقائياً وبذكاء
                solver='adam',
                # تحديد الحد الأقصى لجولات تدريب الشبكة العصبية بـ 500 جولة
                max_iter=500,
                # تحديد قيمة البدء العشوائي الثابت لضمان استقرار وتطابق النتائج الحسابية
                random_state=42
            )
        }
        # إنشاء قاموس فارغ مخصص لتخزين النماذج بعد اكتمال تدريبها
        self.trained_models = {}
        # إنشاء قاموس فارغ لتخزين نتائج ومقاييس تقييم أداء كل نموذج
        self.evaluation_results = {}

    # تعريف دالة لتدريب كافة النماذج الأربعة معاً وتغذيتها ببيانات التدريب
    def train_all(self, X_train, y_train):
        # طباعة رسالة تفيد ببدء مرحلة التدريب الفعلي في الطرفية
        print("[*] Starting model training stage...")
        # المرور بحلقة تكرارية على أسماء النماذج وكائناتها بداخل القاموس المنسق
        for name, model in self.models.items():
            # طباعة اسم النموذج الجاري تدريبه حالياً في سطر الأوامر
            print(f"    -> Training {name}...")
            # استدعاء دالة الملاءمة والتدريب وتمرير مصفوفات ميزات المريض والتشخيص الحقيقي لها
            model.fit(X_train, y_train)
            # حفظ كائن النموذج المدرب بنجاح داخل قاموس النماذج المدربة
            self.trained_models[name] = model
        # طباعة رسالة في الطرفية تؤكد اكتمال تدريب كافة النماذج بنجاح
        print("[+] All models trained successfully.")

    # تعريف دالة لتقييم كافة النماذج المدربة بناءً على عينة الاختبار المستبعدة
    def evaluate_all(self, X_test, y_test):
        # طباعة رسالة تفيد ببدء عملية التقييم وحساب المقاييس الطبية
        print("[*] Evaluating models based on medical-grade metrics...")
        # المرور بحلقة تكرارية على النماذج التي تم تدريبها وحفظها بنجاح
        for name, model in self.trained_models.items():
            # التنبؤ بالتصنيفات الطبية النهائية (0 أو 1) لعينة الاختبار وحفظها في متغير مستقل
            y_pred = model.predict(X_test)
            
            # فحص دعم النموذج لدالة الاحتمالات وتعيين قيمة الاحتمال للفئة 1
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred
            
            # حساب مقياس الاستدعاء أو الحساسية لتقييم رصد الحالات المصابة
            recall = recall_score(y_test, y_pred)
            # حساب مقياس الدقة لتقييم مدى موثوقية التشخيصات الإيجابية الصادرة
            precision = precision_score(y_test, y_pred)
            # حساب درجة إف-1 لتمثيل المتوسط التوافقي المتوازن بين الاستدعاء والدقة
            f1 = f1_score(y_test, y_pred)
            # حساب نسبة الدقة الإجمالية لكافة التوقعات الصحيحة الصادرة من النموذج
            accuracy = accuracy_score(y_test, y_pred)
            # حساب المساحة تحت منحنى الخصائص لتقييم قدرة التمييز الإجمالية
            roc_auc = roc_auc_score(y_test, y_prob)
            
            # تخزين قيم المقاييس الخمسة المحسوبة بداخل قاموس النتائج الخاص بالنموذج الحالي
            self.evaluation_results[name] = {
                'Recall': recall,
                'Precision': precision,
                'F1-Score': f1,
                'Accuracy': accuracy,
                'ROC-AUC': roc_auc
            }
            
            # طباعة سطر يفصل ملخص الأداء الخاص بالنموذج الحالي في الطرفية
            print(f"\n--- {name} Performance Summary ---")
            # طباعة قيمة الاستدعاء مع الإشارة لكونها الأولوية الطبية
            print(f"    * Recall (Sensitivity): {recall:.4f} (Priority Metric)")
            # طباعة قيمة دقة التشخيص بترميز عشري من أربع خانات
            print(f"    * Precision:            {precision:.4f}")
            # طباعة قيمة درجة إف-1 بترميز عشري من أربع خانات
            print(f"    * F1-Score:             {f1:.4f}")
            # طباعة نسبة الدقة الإجمالية بترميز عشري من أربع خانات
            print(f"    * Accuracy:             {accuracy:.4f}")
            # طباعة قيمة المساحة تحت منحنى الخصائص بترميز عشري من أربع خانات
            print(f"    * ROC-AUC:              {roc_auc:.4f}")

    # تعريف دالة لاختيار وتصدير النموذج الأفضل طبياً بالاعتماد على مقياس الاستدعاء (Recall)
    def save_best_model(self):
        # تهيئة متغير فارغ للاحتفاظ باسم النموذج الفائز الحاصل على أعلى استدعاء
        best_model_name = None
        # تهيئة متغير قيمة الاستدعاء الأفضل بقيمة سالبة ابتدائية ليسهل تخطيها في مقارنات الدورة الأولى
        best_recall = -1.0
        
        # المرور بحلقة تكرارية على مقاييس أداء النماذج الأربعة المخزنة لدينا
        for name, metrics in self.evaluation_results.items():
            # التحقق مما إذا كان النموذج الحالي يمتلك معدل استدعاء أعلى من القيمة القصوى السابقة
            if metrics['Recall'] > best_recall:
                # تحديث قيمة الاستدعاء الأفضل بالقيمة الجديدة الأعلى للنموذج الحالي
                best_recall = metrics['Recall']
                # حفظ اسم النموذج الحالي كالنموذج الفائز الجديد بالتقييم الطبي الحمائي
                best_model_name = name
                
        # طباعة اسم النموذج الفائز مع معدل الاستدعاء الخاص به في الطرفية
        print(f"\n[*] Selected Best Model based on Recall: {best_model_name} (Recall: {best_recall:.4f})")
        
        # التأكد من إنشاء مجلد حفظ النماذج مع تجنب إطلاق خطأ إن كان المجلد منشأً مسبقاً
        os.makedirs(self.models_dir, exist_ok=True)
        # دمج مسار المجلد مع اسم الملف لتكوين مسار حفظ ملف النموذج الفائز
        best_model_path = os.path.join(self.models_dir, 'best_model.pkl')
        # فتح ملف حفظ كائن النموذج بمسار الكتابة الثنائية مع ضمان إغلاقه تلقائياً بعد الحفظ
        with open(best_model_path, 'wb') as f:
            # ترميز وحفظ كائن النموذج المدرب الفائز ثنائياً داخل الملف المفتوح
            pickle.dump(self.trained_models[best_model_name], f)
            
        # طباعة رسالة تفيد بنجاح تصدير وحفظ كائن النموذج الفائز في مساره المعتمد
        print(f"[+] Exported best model file to: {best_model_path}")
        
        # دمج مسار المجلد مع الاسم لتكوين مسار حفظ ملف المقاييس الكاملة للنماذج
        metrics_path = os.path.join(self.models_dir, 'metrics.pkl')
        # فتح ملف حفظ قاموس المقاييس بمسار الكتابة الثنائية لضمان إغلاقه تلقائياً بعد الكتابة
        with open(metrics_path, 'wb') as f:
            # ترميز وحفظ قاموس مقاييس أداء النماذج بالكامل لتوفيره لواجهة المستخدم لاحقاً
            pickle.dump(self.evaluation_results, f)
        # طباعة رسالة تفيد بنجاح تصدير وحفظ قاموس المقاييس بالكامل
        print(f"[+] Exported validation metrics summary to: {metrics_path}")

# الشرط الرئيسي لضمان تشغيل الأكواد التالية فقط عند استدعاء وتشغيل هذا الملف بشكل مباشر
if __name__ == '__main__':
    # استيراد كلاس معالجة البيانات من الملف المجاور لبدء الأنبوب البرمجي المشترك
    from data_processor import DataProcessor
    
    # طباعة رسالة ترحيبية تفيد بانطلاق أنبوب التدريب المتكامل
    print("--- Starting End-to-End ML Training Pipeline ---")
    # بدء كتلة المحاولة لرصد وتجنب انهيار التطبيق البرمي في حال حدوث أي خطأ مفاجئ أثناء التدريب
    try:
        # إنشاء كائن جديد من فئة معالجة البيانات بتهيئتها الافتراضية لقراءة الملف
        processor = DataProcessor()
        # تشغيل أنبوب المعالجة بالكامل لاستلام البيانات النظيفة والمقيسة والمجزأة لـ 80/20
        X_train, X_test, y_train, y_test = processor.run_pipeline()
        
        # إنشاء كائن جديد من فئة تدريب وتقييم النماذج الأربعة
        trainer = ModelTrainer()
        # استدعاء دالة تدريب النماذج وتمرير مصفوفات بيانات التدريب الحيوية لها
        trainer.train_all(X_train, y_train)
        # استدعاء دالة تقييم النماذج وحساب مقاييسها الطبية بناءً على عينة الاختبار المستبعدة
        trainer.evaluate_all(X_test, y_test)
        
        # استدعاء دالة مقارنة واختيار النموذج ذي الاستدعاء الأعلى وتصديره بالكامل على القرص الصلب
        trainer.save_best_model()
        # طباعة رسالة نجاح تؤكد اكتمال تنفيذ كامل الأنبوب دون أخطاء برمجية
        print("--- End-to-End Pipeline executed successfully with zero errors ---")
    # التقاط وتخزين أي خطأ برمي أو استثناء غير متوقع حدث بداخل كتلة المحاولة السابقة
    except Exception as e:
        # طباعة رسالة تفيد بانهيار الأنبوب وتوضيح نص الخطأ الملتقط للتحليل وحله
        print(f"[!] Pipeline process crashed with error: {e}")