<div align="center">
  <img src="https://img.icons8.com/color/100/000000/startup.png" alt="Startup Logo">
  <h1>🚀 مشروع التنبؤ بنجاح الشركات الناشئة (Startup Success Prediction)</h1>
  <p><strong>تطبيق متكامل يعتمد على تعلم الآلة لتحليل وتوقع نجاح الشركات الناشئة</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
</div>

---

## 📖 نبذة عن المشروع
تم تطوير هذا المشروع كجزء من مشروع التخرج لمسار **Cretiva NTI Track (Machine Learning for Data Science)**. 
يهدف المشروع إلى التنبؤ بما إذا كانت الشركة الناشئة ستحقق النجاح (مثل الاستحواذ أو الطرح العام) أو ستفشل، وذلك بالاعتماد على بيانات تاريخية تتضمن تفاصيل التمويل، فئات الصناعة، والموقع الجغرافي، وغيرها من العوامل المؤثرة.

## ✨ المميزات الرئيسية
- 📊 **تحليل استكشافي شامل (EDA):** معالجة القيم المفقودة، هندسة الميزات، واكتشاف الأنماط.
- 🧠 **نماذج تعلم آلة متقدمة:** مقارنة بين عدة خوارزميات (Random Forest, XGBoost, Logistic Regression) لاختيار النموذج الأفضل.
- 📈 **قابلية التفسير (Model Interpretability):** استخدام أدوات مثل SHAP لتفسير أهمية كل ميزة في اتخاذ القرار.
- 💻 **واجهة مستخدم تفاعلية (Web App):** تطبيق مبني باستخدام Streamlit يتيح للمستخدمين إدخال بيانات الشركة والحصول على التوقع فوراً (سواء لشركة واحدة أو لعدة شركات عبر ملف CSV).

## 🛠 التقنيات المستخدمة
- **البرمجة ومعالجة البيانات:** Python, Pandas, NumPy
- **تصوير البيانات:** Matplotlib, Seaborn
- **تعلم الآلة:** Scikit-Learn, XGBoost, SHAP
- **نشر التطبيق (Deployment):** Streamlit

## 📂 هيكلية المشروع

```text
├── .gitignore                <- الملفات المستثناة من تتبع Git
├── README.md                 <- ملف الوصف الخاص بالمشروع (هذا الملف)
├── requirements.txt          <- المكتبات والاعتماديات اللازمة لتشغيل المشروع
├── data/
│   ├── raw/                  <- البيانات الأصلية الخام
│   └── processed/            <- البيانات المعالجة والجاهزة للتدريب
├── notebooks/                <- ملفات Jupyter Notebooks
│   ├── 01_data_cleaning_and_feature_engineering.ipynb
│   └── 02_machine_learning_modeling.ipynb
├── models/                   <- النماذج المدربة (Models) والمقاييس (Scalers)
└── app/                      <- تطبيق الويب (Streamlit)
    └── app.py
```

## 👥 فريق العمل والأدوار (Team Members)

<table style="width:100%; text-align:center;">
  <tr>
    <th><strong>1. Ziad Bahaa</strong> 📊</th>
    <th><strong>2. Mohamed</strong> 🤖</th>
    <th><strong>3. Ahmed</strong> 💻</th>
  </tr>
  <tr>
    <td><b>تحليل ومعالجة البيانات</b><br>
        - التحليل الاستكشافي (EDA)<br>
        - معالجة القيم المفقودة والبيانات الشاذة (Outliers)<br>
        - هندسة الميزات (Feature Engineering)<br>
        - ترميز وتقييس البيانات (Encoding & Scaling)
    </td>
    <td><b>تعلم الآلة والنمذجة</b><br>
        - بناء واختبار 6-8 نماذج مختلفة<br>
        - ضبط المعلمات الفائقة (Hyperparameter Tuning)<br>
        - تقييم النماذج والمقارنة بينها<br>
        - تفسير مخرجات النماذج (SHAP)
    </td>
    <td><b>التطبيق والعرض التقديمي</b><br>
        - بناء واجهة المستخدم باستخدام Streamlit<br>
        - دمج نموذج الذكاء الاصطناعي مع الواجهة<br>
        - تمكين التنبؤ المتعدد (Bulk Prediction)<br>
        - إعداد العرض التقديمي النهائي (Presentation)
    </td>
  </tr>
</table>

## 🚀 كيفية التشغيل (How to Run)

للاستمتاع بتجربة التطبيق محلياً على جهازك، يرجى اتباع الخطوات التالية:

**1. قم بتثبيت المتطلبات والاعتماديات:**
تأكد من تواجدك في المجلد الرئيسي للمشروع، ثم نفذ الأمر التالي:
```bash
pip install -r requirements.txt
```

**2. تشغيل واجهة المستخدم (Streamlit App):**
```bash
streamlit run app/app.py
```
*سيتم فتح التطبيق تلقائياً في متصفحك الافتراضي على الرابط (غالباً: http://localhost:8501).*

---
<div align="center">
  <b>تم التصميم والتطوير بشغف من قِبل فريق العمل 💡</b>
</div>
