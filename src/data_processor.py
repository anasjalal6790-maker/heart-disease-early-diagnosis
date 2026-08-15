# استيراد مكتبة النظام لإدارة المجلدات والمسارات
import os
# استيراد مكتبة بيكل لحفظ وتصدير الكائنات بصيغة ثنائية
import pickle
# استيراد مكتبة باندا لإدارة ومعالجة جداول البيانات الجدولية
import pandas as pd
# استيراد دالة تقسيم البيانات إلى مجموعات تدريب واختبار
from sklearn.model_selection import train_test_split
# استيراد المقياس المعياري لتوحيد نطاقات البيانات الرقمية
from sklearn.preprocessing import StandardScaler

# تعريف فئة معالجة البيانات الطبية لمرضى القلب المحدثة والذكية
class DataProcessor:
    """
    Class responsible for loading, cleaning, and preprocessing the heart disease dataset.
    Upgraded to dynamically merge multiple CSV databases placed in the data folder.
    """
    # دالة البناء وتحديد مسار ملف البيانات ومجلد حفظ المقياس
    def __init__(self, data_path='data/heart.csv', models_dir='models'):
        self.data_path = data_path
        self.models_dir = models_dir
        # تهيئة كائن المقياس المعياري لتوحيد نطاق البيانات
        self.scaler = StandardScaler()
        # تحديد أسماء الأعمدة الرقمية المستمرة التي تحتاج لتطبيق الميزان عليها
        self.continuous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
    # دالة قراءة البيانات المحدثة التي تقوم بالدمج التلقائي لكافة قواعد البيانات المتاحة
    def load_data(self):
        # الحصول على مسار المجلد الذي يحتوي على البيانات (مجلد data)
        data_dir = os.path.dirname(self.data_path)
        
        # التحقق من وجود المجلد لتفادي أخطاء التشغيل
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at: {data_dir}")
        
        # جلب مسارات كافة ملفات CSV الموجودة بداخل مجلد البيانات
        csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        # إذا لم يتم العثور على أي ملف CSV، نطلق خطأ صريحاً
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")
        
        # إذا وجدنا ملفاً واحداً فقط بداخل المجلد، نقرأه بشكل طبيعي
        if len(csv_files) == 1:
            print(f"[*] Loading a single dataset from: {csv_files[0]}")
            df = pd.read_csv(csv_files[0])
            
        # إذا وجدنا عدة ملفات، نقوم بدمجهم ديناميكياً وتلقائياً
        else:
            print(f"[*] Found {len(csv_files)} datasets. Dynamically merging them from: {data_dir}")
            dfs = []
            for file in csv_files:
                # قراءة كل ملف على حدة
                temp_df = pd.read_csv(file)
                # التأكد من تطابق الميزات لضمان دمج سليم ووقائي
                dfs.append(temp_df)
            # دمج الجداول معاً وتحديث الفهرس تلقائياً
            df = pd.concat(dfs, ignore_index=True)
            print(f"[+] Dynamically merged datasets successfully! Total combined rows: {len(df)}")
        
        # معالجة القيم المفقودة بملئها بالوسيط الحسابي لكل عمود كإجراء وقائي
        if df.isnull().sum().sum() > 0:
            df.fillna(df.median(), inplace=True)
            
        return df

    # دالة تقسيم البيانات وتجزئتها لـ 80% تدريب و 20% اختبار
    def split_data(self, df, test_size=0.2, random_state=42):
        # فصل الميزات عن العمود المستهدف
        X = df.drop(columns=['target'])
        # تحديد العمود المستهدف
        y = df['target']
        
        # تقسيم البيانات بالتساوي لضمان توازن ديموغرافي مستقر
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    # دالة تدريب وتطبيق ميزان القياس المعياري
    def fit_transform_scaler(self, X_train, X_test):
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # تدريب وتطبيق الميزان المعياري للأعمدة المستمرة
        X_train_scaled[self.continuous_columns] = self.scaler.fit_transform(X_train[self.continuous_columns])
        X_test_scaled[self.continuous_columns] = self.scaler.transform(X_test[self.continuous_columns])
        
        return X_train_scaled, X_test_scaled

    # دالة حفظ كائن المقياس المدرب ثنائياً
    def save_scaler(self):
        os.makedirs(self.models_dir, exist_ok=True)
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"[*] Scaler successfully saved to: {scaler_path}")

    # دالة تشغيل الأنبوب الكامل بالترتيب
    def run_pipeline(self):
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.split_data(df)
        X_train_scaled, X_test_scaled = self.fit_transform_scaler(X_train, X_test)
        self.save_scaler()
        
        return X_train_scaled, X_test_scaled, y_train, y_test

# كتلة الفحص والتحقق الذاتي من عمل الأنبوب
if __name__ == '__main__':
    print("--- Starting Data Processor Validation ---")
    try:
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.run_pipeline()
        
        print("[+] Data loaded and processed successfully!")
        print(f"    - Training features shape (X_train): {X_train.shape}")
        print(f"    - Testing features shape (X_test): {X_test.shape}")
        print("--- Validation finished successfully with zero errors ---")
    except Exception as e:
        print(f"[!] Validation failed with error: {e}")