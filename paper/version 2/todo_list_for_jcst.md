# Actionable To-Do List เพื่อแก้ไขบทความให้ตอบโจทย์ JCST (และ SUJST) ครบทุกข้อ

เอกสารนี้แบ่งงานที่ต้องทำออกตาม **ตารางจัดลำดับความสำคัญ (Priority Matrix)** และรายละเอียดแยกตาม **ข้อกังวลหลัก 5 ข้อของ JCST** ร่วมกับ **ข้อที่ 6 (การแก้คำผิดและจัดหน้าในต้นฉบับ)**

---

## ตารางสรุปจัดลำดับความสำคัญ (Priority Matrix)

การจัดลำดับแบ่งเป็น 4 ระดับตาม **"ผลกระทบต่อการถูกปัดตก (Impact)"** เทียบกับ **"เวลาและทรัพยากรที่ใช้ (Effort)"**:
- **P0 (วิกฤต - ต้องแก้ทันที):** ข้อผิดพลาดร้ายแรงในไฟล์ Word ที่หากส่งไปจะถูก Desk Reject ทันที (ใช้เวลาแก้แค่ 15–20 นาที)
- **P1 (คุ้มค่าสูงสุด - Quick Wins):** ตอบโจทย์ JCST ข้อ 3, 4, 5 ได้ทันทีโดย **ไม่ต้องเทรนโมเดลใหม่ทั้งหมด** (รันสคริปต์ Python เสริมไม่เกิน 30 นาที)
- **P2 (สำคัญสูง - Core Re-training):** ตอบโจทย์ JCST ข้อ 1 และ 2 ซึ่งเป็นหัวใจของระเบียบวิธีวิจัย (ต้องรันเทรนโมเดลเพิ่ม)
- **P3 (งานเสริม - Nice-to-Have):** เพิ่มความแข็งแกร่งระดับ Q1/Q2 (ทำเมื่อมีเวลาเหลือ)

| ลำดับ | ระดับ | งานที่ต้องทำ (Action Item) | ตอบโจทย์ข้อไหน | ต้องเทรนใหม่ไหม? | เวลาที่ใช้ | ผลกระทบ (Impact) |
| :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | **P0** | **ล้างข้อความค้างจาก Template และย่อหน้าที่ก๊อปซ้ำในไฟล์ Word** (`State conclusion (do not`, หัวข้อซ้ำหน้า 10, `Running Head`, `(12 pt Bold)`, แก้ `9,600 km2` $\rightarrow$ `96,000 km²`, เพิ่มอ้างอิง `Zhang et al., 2023` และ `Figure 4`) | หมวด 6 (ป้องกันถูกปัดตกตั้งแต่ขั้นตรวจรูปแบบ) | ❌ ไม่ต้อง | 15–20 นาที | **สูงมาก** (ถ้าไม่แก้ เสี่ยงโดน Desk Reject ทันที) |
| **2** | **P1** | **คำนวณค่าสถิติ McNemar’s Test จริงใส่ในบทความ** และลบประโยคที่เขียนชี้เป้าว่า *"McNemar's test is still needed"* ออก (หน้า 6) | **JCST ข้อ 4** (Statistical reliability) | ❌ ไม่ต้อง (ใช้ผลทำนายเดิมได้) | 10 นาที | **สูงมาก** (ปิดจุดอ่อนที่เขียนเผาตัวเองไว้ในหน้า 6) |
| **3** | **P1** | **วัดค่า Accuracy, Macro-F1 และความเร็ว (Latency) ของไฟล์ `.tflite` (4.2 MB)** ด้วย `tf.lite.Interpreter` และส่งออกโมเดลตัวที่ดีที่สุดจริง (92.80%) แทนตัว 89.63% | **JCST ข้อ 5** (Unvalidated deployment) | ❌ ไม่ต้อง | 15 นาที | **สูงมาก** (เปลี่ยนจากการสารภาพว่าไม่ได้วัด เป็นมีตารางผลจริง) |
| **4** | **P1** | **รันสคริปต์คัดกรองภาพซ้ำ (MD5 + Perceptual Hash `pHash`)** บน 1,763 ภาพ + เปิดตรวจภาพที่ทายผิดข้ามอันดับ 8 ภาพด้วยสายตา และแก้ชื่อโฟลเดอร์ภาษาไทยของ *S. furcifera* | **JCST ข้อ 3** (Dataset verification & duplicates) | ❌ ไม่ต้อง (เว้นแต่พบภาพซ้ำข้าม Split จำนวนมาก) | 20–30 นาที | **สูง** (ยืนยันคุณภาพชุดข้อมูลระดับภาพได้อย่างเป็นระบบ) |
| **5** | **P2** | **เพิ่มการทดลอง Phase 2 แบบล็อกชั้น `BatchNormalization` (`Freeze BN`)** ของ EfficientNetB0 (และโมเดลอื่น) เทียบกับการปลดล็อกทุกชั้นแบบเดิม (`Unfreeze All`) | **JCST ข้อ 2** (Controlled experiment for central claim) | ✅ เทรนเฉพาะ Phase 2 เพิ่ม | 1–2 ชั่วโมง (บน GPU) | **สูงมาก** (พิสูจน์ข้อกล่าวอ้างหลักของชื่อเรื่องได้ 100%) |
| **6** | **P2** | **แบ่งชุดข้อมูลใหม่เป็น 3 ส่วน: Train (70%) / Validation (15%) / Independent Test Set (15%)** แบบ Stratified Split และวัดผลสุดท้ายบน Test Set | **JCST ข้อ 1** (Insufficient independent validation) | ✅ ต้องเทรนใหม่ | 2–4 ชั่วโมง (บน GPU) | **สูงมาก** (แก้ข้อติเรื่อง Validation Optimism ได้เด็ดขาด) |
| **7** | **P3** | **รันการทดลองซ้ำ 3 Random Seeds** (เช่น `seed = 42, 100, 2024`) เพื่อรายงานค่า $\text{Mean} \pm \text{SD}$ | **JCST ข้อ 4** (Reproducibility across runs) | ✅ ต้องเทรนใหม่ 3 รอบ | 6–8 ชั่วโมง (บน GPU) | **ปานกลาง–สูง** (จำเป็นสำหรับ Q1/Q2 หรือส่งกลับ JCST; สำหรับ Q4 อาจใช้ CI + McNemar แทนได้) |
| **8** | **P3** | **ทดสอบโมเดลกับภาพถ่ายแมลงจากแปลงนาจริงภายนอก GBIF** (External field test set ~30–50 ภาพ) หรือวัดความเร็วบนสมาร์ทโฟนจริง | **JCST ข้อ 5** (Field photo evaluation) | ❌ ไม่ต้อง | 1–2 ชั่วโมง (หาภาพและรันทดสอบ) | **ปานกลาง** (ช่วยให้บทความโดดเด่นมากในแง่การใช้งานจริง) |

---

## หมวดที่ 1: แก้ไขเรื่อง "ไม่มีชุดข้อมูลทดสอบแยกอิสระ (Insufficient Independent Validation)"
> **สิ่งที่ JCST ติ:** การแบ่งข้อมูลแบบ Train/Validation (80:20) เพียงชุดเดียวโดยใช้ชุด Validation ทั้งเลือกโมเดล (Early stopping) และรายงานผล ทำให้ค่าความแม่นยำ (92.80%) อาจดูสูงเกินจริง (Optimistic estimate)

### สิ่งที่ต้องทำ (To-Do):
- [ ] **1.1 แบ่งชุดข้อมูลออกเป็น 3 ส่วน (Train / Validation / Test) แบบ Stratified Split:**
  - เปลี่ยนจากการใช้ `ImageDataGenerator(validation_split=0.2)` เดิม มาเป็นการแบ่งไฟล์ระดับโฟลเดอร์หรือ DataFrame แบบ Stratified ในอัตราส่วน **70% Train / 15% Validation / 15% Held-out Test** (หรือ **80% Train+Val / 20% Held-out Test**)
  - *ข้อควรระวัง:* สำหรับคลาสที่มีภาพน้อยมาก เช่น *Dicladispa armigera* (13 ภาพ) และ *Orseolia oryzae* (22 ภาพ) ให้กำหนดเกณฑ์ขั้นต่ำว่าใน Validation และ Test ต้องมีอย่างน้อยคลาสละ 2–3 ภาพเสมอ
- [ ] **1.2 ใช้ Validation Set เฉพาะตอนปรับพารามิเตอร์/Early Stopping และเก็บ Test Set ไว้วัดผลครั้งเดียวตอนจบ:**
  - รายงานผลทั้ง **Validation Accuracy** (ระหว่างเทรน) และ **Independent Test Accuracy / Macro-F1 / MCC** (ผลลัพธ์สุดท้าย) ในตารางที่ 3 และตารางที่ 4
- [ ] **1.3 ลบประโยคที่เขียนยอมรับจุดอ่อนเดิมในบทความ:**
  - ลบประโยคในหน้า 10 (*"First, the reported accuracy was measured on a validation set that was also used for early stopping, which may lead to an optimistic estimate..."*) และหน้า 12 (*"its performance must be assessed on an independent test set"*) ออก แล้วเขียนแทนว่าได้ประเมินบน Held-out Independent Test Set เรียบร้อยแล้ว

---

## หมวดที่ 2: แก้ไขเรื่อง "ขาดการทดลองควบคุมเพื่อพิสูจน์ผลของ Fine-Tuning และ Batch Normalization"
> **สิ่งที่ JCST ติ:** บทความเคลมว่าผลของ Fine-tuning ขึ้นอยู่กับสถาปัตยกรรม (Architecture-dependent) แต่กลับทดสอบ Fine-tuning แค่คอนฟิกเดียว โดยไม่ได้แยกแยะว่าเป็นผลมาจากสถาปัตยกรรมเอง หรือมาจากชั้น `BatchNormalization` (BN) หรือ Learning Rate

### สิ่งที่ต้องทำ (To-Do):
- [ ] **2.1 เพิ่มการทดลองเปรียบเทียบ (Ablation Study) ใน Phase 2 (Fine-tuning) อย่างน้อย 2 รูปแบบ:**
  - **Condition A (Naive Unfreezing - แบบเดิม):** ปลดล็อกทุกเลเยอร์ใน Backbone รวมทั้งชั้น `BatchNormalization` (`base_model.trainable = True`) $\rightarrow$ เพื่อแสดงให้เห็นว่า EfficientNetB0 พังทลายลงเพราะสถิติ BN ถูกรบกวน
  - **Condition B (BN-Frozen Unfreezing - ตามแนวทางมาตรฐาน):** ปลดล็อกเลเยอร์ Convolution ใน Backbone แต่ **ล็อกชั้น `BatchNormalization` ทั้งหมดไว้ในโหมด Inference** (`if isinstance(layer, tf.keras.layers.BatchNormalization): layer.trainable = False`)
  - *(ทางเลือกเสริม Condition C):* ทดสอบปลดล็อกเฉพาะ 20–30% บนสุดของ Backbone (Top-block fine-tuning) หรือเปรียบเทียบ Learning Rate ($10^{-4}$ vs $10^{-5}$)
- [ ] **2.2 อัปเดตตารางที่ 3 และกราฟรูปที่ 2–3:**
  - แสดงผลเปรียบเทียบระหว่าง **Phase 1 (Frozen)** vs **Phase 2A (Unfreeze All incl. BN)** vs **Phase 2B (Unfreeze Conv, Freeze BN)** ทั้ง 3 สถาปัตยกรรม
- [ ] **2.3 ปรับเนื้อหาในหัวข้อ *Effects of fine-tuning* และ *Discussion*:**
  - ลบประโยค *"controlled experiments are needed..."* และ *"this mechanism was not tested directly in the present study"* ออก
  - เขียนสรุปผลจากการทดลองจริงเลยว่า การที่ EfficientNetB0 ประสิทธิภาพตกลงใน Phase 2A เกิดจากการอัปเดตสถิติของชั้น `BatchNormalization` บน Mini-batch ขนาดเล็ก ($B=16$) ร่วมกับโมดูล Squeeze-and-Excitation (SE) และเมื่อล็อกชั้น BN ใน Phase 2B แล้ว โมเดลสามารถรักษา/เพิ่มความแม่นยำได้อย่างไร

---

## หมวดที่ 3: แก้ไขเรื่อง "การตรวจสอบคุณภาพชุดข้อมูลและการคัดกรองภาพซ้ำ (Dataset Quality & De-duplication)"
> **สิ่งที่ JCST ติ:** อธิบายแค่การจับคู่ชื่อวิทยาศาสตร์ผ่าน GBIF แต่ไม่ได้แสดงการคัดกรองภาพซ้ำ/ภาพคล้ายกัน (Duplicate/Near-duplicate) อย่างเป็นระบบ และไม่ได้ยืนยันความถูกต้องของชนิดแมลงในระดับรายภาพ

### สิ่งที่ต้องทำ (To-Do):
- [ ] **3.1 รันสคริปต์คัดกรองภาพซ้ำ (Exact & Near-Duplicate Screening) บนโฟลเดอร์ `dataset/images`:**
  - ใช้ Python คำนวณ **Cryptographic Hash (MD5/SHA-256)** เพื่อเช็คไฟล์ภาพที่ซ้ำกัน 100%
  - ใช้ **Perceptual Hash (`imagehash` เช่น `phash` หรือ `dhash`)** หรือ Cosine Similarity ของ Feature Vector เพื่อตรวจหาภาพที่ถ่ายจากเหตุการณ์เดียวกันหรือภาพซ้ำเกือบทั้งหมด (Near-duplicates) ที่อาจรั่วไหลข้ามระหว่าง Train/Val/Test (Data leakage)
  - บันทึกจำนวนภาพที่ตรวจพบและตัดออก (หรือจับกลุ่มให้อยู่ใน Split เดียวกัน) แล้วรายงานตัวเลขนี้ลงในหัวข้อ *Dataset construction*
- [ ] **3.2 ตรวจสอบภาพที่ทายผิดข้ามอันดับ (Cross-order errors) และแก้ไขเรื่องชื่อโฟลเดอร์ภาษาไทย:**
  - เปิดตรวจสอบภาพในคลาสที่ทายผิดข้ามอันดับ (เช่น *Mythimna separata* ที่โมเดลทายว่าเป็น *Sogatella furcifera*) ว่าภาพต้นฉบับในชุดข้อมูลเป็นภาพที่ถูกต้องหรือเป็นภาพที่ติดป้ายกำกับผิดจาก GBIF หากเป็นภาพที่ผิดพลาด/ไม่ชัดเจน ให้คัดออกหรือระบุผลการตรวจสอบด้วยสายตา (Visual inspection)
  - แก้ไขชื่อโฟลเดอร์ของ *Sogatella furcifera* ที่มีตัวสะกดภาษาไทยให้ถูกต้องตามมาตรฐาน และลบย่อหน้าในหน้า 4 ที่เขียนบ่นเรื่องลำดับตัวอักษรสลับคลาสออก

---

## หมวดที่ 4: แก้ไขเรื่อง "ความสามารถในการทำซ้ำและความน่าเชื่อถือทางสถิติ (Reproducibility & Statistical Reliability)"
> **สิ่งที่ JCST ติ:** ต้องมีการฝึกสอนซ้ำหลายรอบ (Repeated runs) และต้องใช้สถิติเปรียบเทียบโมเดลแบบจับคู่ (Paired model comparisons) เพื่อยืนยันว่าความแตกต่างมีนัยสำคัญทางสถิติจริง

### สิ่งที่ต้องทำ (To-Do):
- [ ] **4.1 รันการทดลองซ้ำอย่างน้อย 3–5 Random Seeds (หรือ 5-Fold Stratified Cross-Validation):**
  - แทนที่จะรันด้วย `seed = 42` เพียงรอบเดียว ให้รันด้วย 3–5 Seeds (เช่น `42, 100, 2024`) หรือทำ 5-Fold Stratified CV เพื่อรายงานค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน ($\text{Mean} \pm \text{SD}$) ของ Accuracy, Macro-F1 และ MCC
- [ ] **4.2 คำนวณสถิติเปรียบเทียบโมเดลแบบจับคู่ (Exact McNemar’s Test & Cochran’s Q Test):**
  - เขียนโค้ด Python คำนวณค่าสถิติ **McNemar’s test** (และค่า $p$-value) เปรียบเทียบผลทำนายรายภาพบน Test Set ระหว่าง:
    1. EfficientNetB0 vs ResNet50V2
    2. EfficientNetB0 vs MobileNetV2
    3. Phase 1 (Frozen) vs Phase 2 (Fine-tuned) ของแต่ละโมเดล
  - นำค่า $p$-value จริงไปใส่ในหัวข้อ *Results* (หน้า 6) และ **ลบประโยคที่เขียนว่า *"such as McNemar’s test, is still needed"* ออกทันที**

---

## หมวดที่ 5: แก้ไขเรื่อง "การทดสอบการใช้งานจริงของโมเดล TFLite (Practical Deployment Validation)"
> **สิ่งที่ JCST ติ:** แปลงโมเดลเป็นไฟล์ TensorFlow Lite (4.2 MB) แล้ว แต่ไม่ได้วัดความแม่นยำหลังแปลงไฟล์ (Post-conversion accuracy) และไม่ได้วัดประสิทธิภาพเวลาประมวลผล (Inference latency) หรือทดสอบกับภาพถ่ายจากแปลงนาจริง

### สิ่งที่ต้องทำ (To-Do):
- [ ] **5.1 ส่งออก (Export) โมเดลตัวที่ได้ความแม่นยำสูงสุดจริงไปเป็น TFLite:**
  - แก้ไขโค้ดให้ส่งออกโมเดลที่ดีที่สุด (เช่น Phase-1 Frozen EfficientNetB0 ที่ได้ 92.80% หรือ Phase-2 BN-Frozen) แทนการส่งออกโมเดล Phase-2 ตัวที่ความแม่นยำตกเหลือ 89.63%
- [ ] **5.2 รันวัดผลโมเดล `.tflite` (Post-conversion Evaluation) ด้วย `tf.lite.Interpreter`:**
  - วัดค่า **Accuracy, Macro-F1 และ MCC หลังทำ Dynamic-range Quantization** เทียบกับโมเดล Float32 ก่อนแปลง (เพื่อพิสูจน์ว่าความแม่นยำลดลงน้อยมาก เช่น $< 0.2\%$)
  - วัดค่า **Inference Latency (มิลลิวินาทีต่อภาพ - ms/image)** และขนาดหน่วยความจำ (RAM/File size) ทั้งบน CPU/Mobile
- [ ] **5.3 (แนะนำเพิ่มเติม) ทดสอบกับชุดภาพถ่ายภายนอก/ภาพจากแปลงนาจริง (External Field Validation Set):**
  - หยิบภาพแมลงศัตรูข้าวจากแหล่งภาพถ่ายในแปลงนาจริง (เช่น ภาพจากคู่มือกรมส่งเสริมการเกษตร หรือภาพจากชุดข้อมูลภาคสนามที่ไม่ได้มาจาก GBIF จำนวน ~30–50 ภาพ) มารันทดสอบเพื่อรายงานผล Out-of-distribution / Field accuracy เบื้องต้น
- [ ] **5.4 ลบประโยคสารภาพในหน้า 10 ออก:**
  - ลบประโยค *"However, the accuracy of the quantized model and its on-device inference latency were not evaluated in this study."* ออก แล้วแทนที่ด้วยตารางหรือตัวเลขผลการวัดจริง

---

## หมวดที่ 6: แก้ไขข้อผิดพลาดด้านการจัดหน้าและคำผิดในต้นฉบับ (Manuscript Proofing Checklist)
- [ ] **6.1 ลบข้อความคำสั่งค้างในหน้า 11 (Conclusions):** ลบ `"The possibility that fine-tuning can degrade... State conclusion (do not "` ออก
- [ ] **6.2 ลบหัวข้อย่อยที่ก๊อปวางซ้ำในหน้า 10:** ลบหัวข้อ `"Per-class behavior of the exported model"` ที่โผล่ซ้ำและขาดครึ่งประโยคในหน้า 10 คอลัมน์ขวาออก
- [ ] **6.3 แก้ไขตัวเลขพื้นที่ปลูกข้าวของประเทศไทย (หน้า 2):** เปลี่ยนจาก `9,600 km2` เป็น `approximately 96,000 km² (9.6 million hectares)`
- [ ] **6.4 เพิ่มรายการอ้างอิงที่ตกหล่น:** เพิ่ม `Zhang et al. (2023)` ในส่วน *References* และใส่การอ้างอิง `Figure 4` ในเนื้อหาหลักก่อน `Figure 5`
- [ ] **6.5 แก้ไขข้อความจาก Template ที่ยังไม่ได้แก้:**
  - แก้ `Running Head` บนหัวกระดาษทุกหน้า
  - ลบคำว่า `(12 pt Bold)` ออกจากหัวข้อ `Declaration of Generative AI Use` (หน้า 12)
  - แก้คำผิดชื่อมหาวิทยาลัยในหน้า 12 จาก `Valaya Aongkorn` เป็น `Valaya Alongkorn`
  - แก้ข้อความใน `Data Availability Statement` (หน้า 12) ให้ตรงกับความจริงว่าใช้ข้อมูลจากฐานข้อมูลเปิด GBIF

