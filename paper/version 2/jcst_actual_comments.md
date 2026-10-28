# Official Editorial Decision & Comments from JCST (Journal of Current Science and Technology)

**Authors:** Daorathar Weerapan, Natradee Anupong, Chavalit Koweerawong  
**Original Manuscript Title (Version 1 submitted to JCST):** *"Architecture-Dependent Effects of Fine-Tuning in Transfer Learning for Rice Insect Pest Classification with a Small, Taxonomically Verified Dataset"*  
**Journal:** *Journal of Current Science and Technology (JCST)*  
**Editorial Decision:** **Decline Submission (Desk Reject / Preliminary Review)**

---

## 1. Original Editorial Decision Letter (English)

> **Daorathar Weerapan, Natradee Anupong, Chavalit Koweerawong:**
>
> Thank you very much for submitting your manuscript to *Journal of Current Science and Technology (JCST)*.  
> The editorial team have reached a decision regarding your submission to *Journal of Current Science and Technology*, *"Architecture-Dependent Effects of Fine-Tuning in Transfer Learning for Rice Insect Pest Classification with a Small, Taxonomically Verified Dataset"*.
>
> I have carefully preliminary reviewed your manuscript, and, unfortunately, must decline it from further consideration. It is JCST’s policy to consider and process only the manuscripts, with the corrected JCST format and the decisions based on level of innovation, the degree of advancement in concepts, the importance of the topic, and the breadth of potential interest to the research community.
>
> The editorial team has an initial assessment and decides that the manuscript cannot be considered for publication.  
> **Our decision is to: Decline Submission (Please read the reasons carefully):**
>
> The principal concerns are as follows:
>
> 1. **Insufficient independent validation.** The study relies on a single training-validation split without an independent test set. The reported validation accuracy may therefore provide an optimistic estimate of model performance.
> 2. **Limited experimental support for the central claim.** The reported architecture-dependent effects are based on one Fine-Tuning configuration. No controlled experiments were conducted to distinguish architectural effects from the influence of Batch Normalization, learning-rate settings, or other optimization factors.
> 3. **Insufficient verification of dataset quality.** The manuscript describes taxonomic name matching through GBIF but does not adequately demonstrate image-level species verification or systematic screening for duplicate and near-duplicate images.
> 4. **Inadequate assessment of reproducibility and statistical reliability.** Repeated training experiments and appropriate paired model comparisons are needed to determine whether the reported differences are consistent and statistically supported.
> 5. **Unvalidated practical deployment.** Although the model was converted to TensorFlow Lite, neither its post-conversion accuracy nor its performance on mobile devices or actual rice-field photographs was evaluated.
>
> Addressing these concerns would require substantial additional experiments and validation beyond a conventional manuscript revision.
>
> This decision is provided to you as quickly as possible so that you may send your manuscript elsewhere without further delay.
>
> We are truly grateful to have your trust and support for the journal, and we hope this outcome will not discourage you from submitting your research to JCST in the future.

---

## 2. คำแปลความเห็นจากกองบรรณาธิการ JCST (ฉบับภาษาไทย)

**ผลการพิจารณาเบื้องต้น:** **ปฏิเสธการรับพิจารณา (Decline Submission)**

**ข้อกังวลหลัก 5 ประการจากกองบรรณาธิการ JCST:**

1. **การตรวจสอบความถูกต้องด้วยชุดข้อมูลอิสระยังไม่เพียงพอ (Insufficient independent validation):**
   การศึกษานี้พึ่งพาการแบ่งข้อมูลเป็นชุดฝึกสอนและชุดตรวจสอบ (Single training–validation split) เพียงครั้งเดียวโดยไม่มีชุดข้อมูลทดสอบที่แยกเป็นอิสระ (Independent test set) ดังนั้น ค่าความแม่นยำบนชุดตรวจสอบ (Validation accuracy) ที่รายงานจึงอาจเป็นการประเมินประสิทธิภาพของโมเดลในแง่ดีเกินจริง (Optimistic estimate) เนื่องจากใช้ชุดข้อมูลเดียวกันในการเลือกโมเดลที่ดีที่สุด (Early stopping)

2. **ผลการทดลองที่รองรับข้อกล่าวอ้างหลักยังมีจำกัด (Limited experimental support for the central claim):**
   ผลกระทบของ Fine-tuning ที่แตกต่างกันตามสถาปัตยกรรม (Architecture-dependent effects) ซึ่งนำเสนอในบทความอ้างอิงจากการตั้งค่า Fine-tuning เพียงรูปแบบเดียว โดยไม่มีการทดลองแบบควบคุม (Controlled experiments) เพื่อแยกแยะว่าผลที่เกิดขึ้นมาจากตัวสถาปัตยกรรมเอง หรือมาจากอิทธิพลของชั้น Batch Normalization, การตั้งค่าอัตราการเรียนรู้ (Learning rate) หรือปัจจัยด้านการปรับค่าเหมาะที่สุด (Optimization factors) อื่นๆ

3. **การตรวจสอบคุณภาพของชุดข้อมูลยังไม่รัดกุมเพียงพอ (Insufficient verification of dataset quality):**
   ต้นฉบับอธิบายการจับคู่ชื่อทางอนุกรมวิธานผ่าน GBIF แต่ไม่ได้แสดงให้เห็นถึงการตรวจสอบความถูกต้องของชนิดแมลงในระดับรายภาพ (Image-level species verification) หรือการคัดกรองภาพซ้ำและภาพที่คล้ายกันมาก (Duplicate and near-duplicate images) อย่างเป็นระบบ

4. **การประเมินความสามารถในการทำซ้ำและความน่าเชื่อถือทางสถิติยังไม่เพียงพอ (Inadequate assessment of reproducibility and statistical reliability):**
   จำเป็นต้องมีการฝึกสอนซ้ำหลายรอบ (Repeated training runs / Cross-validation) และใช้การเปรียบเทียบโมเดลแบบจับคู่ทางสถิติที่เหมาะสม (Paired model comparisons เช่น McNemar's test) เพื่อยืนยันว่าความแตกต่างที่รายงานมีความสม่ำเสมอและมีนัยสำคัญทางสถิติรองรับจริง

5. **การนำไปใช้งานจริงยังไม่ได้รับการทดสอบยืนยัน (Unvalidated practical deployment):**
   แม้ว่าจะมีการแปลงโมเดลเป็นไฟล์ TensorFlow Lite (TFLite) แล้ว แต่ยังไม่ได้ประเมินความแม่นยำหลังการแปลงไฟล์ (Post-conversion accuracy) รวมถึงยังไม่ได้ทดสอบประสิทธิภาพการทำงานบนอุปกรณ์พกพาจริง หรือกับภาพถ่ายจากแปลงนาข้าวจริง

---

## 3. วิเคราะห์เปรียบเทียบ: สิ่งที่ปรับแก้แล้วใน Version 2 (SUJST) vs. จุดที่ยังเสี่ยงโดนโจมตีซ้ำ

จากการเปรียบเทียบต้นฉบับ **Version 2 (สำหรับส่ง SUJST)** กับข้อวิจารณ์ทั้ง 5 ข้อของ **JCST** พบว่าใน Version 2 ผู้วิจัยได้ใช้วิธี **"ลดระดับการเคลม (Tone down claims) และเขียนยอมรับเป็นข้อจำกัด (Limitations)"** ในเนื้อหาแทนการรันการทดลองใหม่ อย่างไรก็ตาม มีบางประเด็นที่สามารถทำเพิ่มได้ง่ายๆ ในโค้ด (`insect.ipynb`) เพื่อปิดจุดอ่อนก่อนส่ง **SUJST** ดังนี้:

| ข้อกังวลของ JCST | สิ่งที่ทำไปแล้วใน Version 2 (SUJST) | ข้อเสนอแนะเพิ่มเติมที่ทำได้ทันทีเพื่อป้องกัน Reviewer ของ SUJST ติซ้ำ |
| :--- | :--- | :--- |
| **1. ไม่มี Independent Test Set (ใช้แค่ Train/Val 80:20)** | เขียนยอมรับไว้ในข้อจำกัดหน้า 10 และหน้า 12 ว่าใช้ Validation set เดียวกับ Early stopping | หากไม่ต้องการเทรนใหม่แบบ Train/Val/Test (เช่น 70:15:15) ควรระบุให้ชัดเจนว่าเป็น Pilot benchmarking study และระวังอย่าเคลมความแม่นยำ 92.80% แรงเกินไปในตารางเปรียบเทียบ |
| **2. ทดสอบ Fine-tuning แค่คอนฟิกเดียว (ไม่ได้แยกผลของ BatchNorm)** | ปรับภาษาในหน้า 5 และหน้า 7 ให้ซอฟต์ลงว่าเป็นเพียง "สมมติฐานที่เป็นไปได้ (Plausible explanations)" | หากเทรน Phase 2 ของ EfficientNetB0 เพิ่มอีกแค่ 1 รัน โดยสั่งล็อก `BatchNormalization` (`layer.trainable = False` สำหรับชั้น BN) จะตอบข้อ 2 ได้อย่างสมบูรณ์แบบและทำให้บทความแข็งแรงขึ้นมาก |
| **3. การคัดกรองภาพซ้ำ (Duplicate/Near-duplicate) และตรวจสอบระดับภาพ** | ระบุในหน้า 4 ว่าเช็คความสมบูรณ์ของไฟล์และตั้งชื่อตาม Occurrence key เพื่อกันภาพซ้ำจากการดาวน์โหลดซ้ำ | สามารถรันสคริปต์เช็คค่า Hash (เช่น MD5/SHA256 หรือ Perceptual Hash `pHash`) บนโฟลเดอร์ภาพทั้ง 1,763 ภาพใน Python แล้วนำตัวเลขมารายงานในหัวข้อ *Dataset construction* ได้ภายในไม่กี่นาที |
| **4. ไม่มีสถิติเปรียบเทียบโมเดลแบบจับคู่ (Paired statistical test)** | ในหน้า 6 เขียนทิ้งท้ายไว้ว่า *"a paired statistical test... such as McNemar's test, is still needed"* | **จุดนี้โดนติแน่นอนหากปล่อยไว้:** เพราะผู้วิจัยเขียนชี้เป้าเองในหน้า 6 ว่าต้องทำ McNemar's test แต่กลับไม่ยอมคำนวณใส่ลงไป ทั้งที่มีผลทำนาย 347 ภาพอยู่แล้ว แนะนำให้รัน McNemar's test ใน Python แล้วใส่ค่า $p$-value จริงลงในบทความทันที |
| **5. ไม่ได้วัด Accuracy และ Latency ของไฟล์ TFLite (4.2 MB)** | ในหน้า 10 เขียนยอมรับตรงๆ ว่า *"However, the accuracy of the quantized model and its on-device inference latency were not evaluated"* | **แก้ไขง่ายมาก:** ใช้ `tf.lite.Interpreter` ใน Python รันวัด Accuracy, Macro-F1 และเวลาเฉลี่ยต่อภาพ (ms/image) ของไฟล์ `.tflite` บนชุดข้อมูล Validation 347 ภาพ แล้วนำตัวเลขจริงมาใส่แทนการบอกว่าไม่ได้วัด |

