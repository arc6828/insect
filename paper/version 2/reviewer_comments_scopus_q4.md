# Peer Review Report (Scopus Q4 Journal Standard)
**Target Journal:** *Suranaree Journal of Science and Technology (SUJST)* (Scopus Q4)  
**Manuscript Title:** ARCHITECTURE-DEPENDENT EFFECTS OF FINE-TUNING ON RICE INSECT PEST CLASSIFICATION: A COMPARISON OF MOBILENETV2, EFFICIENTNETB0, AND RESNET50V2  
**Recommendation:** **Major Revision** (Accept after addressing critical editorial defects and minor methodological clarifications)

---

## Part 1: English Version (Official Peer Review Report)

### 1. General Assessment & Summary
This manuscript presents an empirical comparison of three widely used convolutional neural network (CNN) architectures—MobileNetV2, EfficientNetB0, and ResNet50V2—on a 17-species dataset of rice insect pests in Thailand (1,763 images retrieved and taxonomically verified via the GBIF API). All models were trained under a controlled two-phase transfer learning protocol (Phase 1: frozen backbone; Phase 2: full-backbone fine-tuning). Furthermore, the authors propose a simple yet insightful taxonomic error analysis based on a uniform-error null model to verify whether misclassifications follow biological/morphological relationships rather than background shortcuts.

**Strengths of the manuscript:**
1. **Practical & MethodologicalHonesty:** Unlike many applied agricultural deep-learning papers that only report a single "best" accuracy figure, this work transparently reports both training phases, Wilson 95% confidence intervals, chance-corrected metrics (MCC and Cohen's $\kappa$), and the negative effects of naive full-backbone fine-tuning on EfficientNetB0.
2. **Taxonomic Error Analysis (Null Model):** Using an exact binomial test against a random taxonomic error baseline (Equation 7; 77.8% vs. 37.5% within-order errors) is a clever, lightweight diagnostic check that adds clear scientific value suitable for a Scopus-indexed journal.
3. **Taxonomic Hygiene in Dataset Construction:** Resolving synonyms and taxon identifiers via the GBIF API rather than scraping web search engines by keyword is commendable and well-motivated.

However, **the manuscript currently cannot be accepted in its present form** due to several **critical editing/proofing errors** (unfinished template instructions, duplicated paragraphs, a missing reference, and a factual error in Thailand's rice cultivation area) as well as a few **methodological questions** that can be easily resolved by the authors. I recommend **Major Revision** and encourage the authors to address the points below thoroughly.

---

### 2. Major Comments (Critical Issues to Address)

#### Major Comment 1: Unfinished Template Text, Duplicated Sections, and Editing Artifacts
Before this paper can go through production or external indexing, the following severe proofing errors **must** be fixed:
1. **Leftover Instruction in Conclusions (Page 11, Column 2):** The first paragraph of the *Conclusions* section begins with:
   > *"The possibility that fine-tuning can degrade a pretrained model is not new and has been noted in framework documentation (Chollet, 2020). State conclusion (do not This study evaluated three CNN architectures..."*
   This contains a copy-pasted sentence from the *Discussion* followed by a raw editing prompt `"State conclusion (do not "`. Please remove the prompt and rewrite the opening sentence cleanly.
2. **Duplicated and Truncated Subsection on Page 10 (Column 2):** Immediately preceding the *Discussion* section on Page 10, the subsection *"Per-class behavior of the exported model"* (which already appears on Page 8) is pasted again. Worse, its opening paragraph is repeated twice in a row, and the second copy ends mid-sentence:
   > *"...which is the model that was exported, rather than on the 92.80%"*
   Please delete this entire duplicated subsection on Page 10.
3. **Unedited Running Head & Heading Instructions (Pages 2–13):**
   - The page header across the entire manuscript still reads: *"Running Head: Provide a shortened version of the title, not exceeding 60 characters, including spaces. (9 pt)"*. Please provide an actual running head (e.g., `FINE-TUNING EFFECTS ON RICE PEST CLASSIFICATION`).
   - On Page 12, the heading reads `"Declaration of Generative AI Use (12 pt Bold)"`. Please remove `"(12 pt Bold)"`.
4. **Contradictory Data Availability Statement (Page 12):** The statement reads: *"No new data were generated or analyzed in support of this research."* This contradicts the entire manuscript, which constructs and analyzes a 1,763-image dataset downloaded from GBIF. Please revise this statement to explain how readers can access the dataset or the GBIF occurrence keys/DOI (note that GBIF requires formal DOI citation for occurrence downloads).

#### Major Comment 2: Methodological Questions & Experimental Completeness
1. **Why Export and Evaluate the Degraded Phase-2 EfficientNetB0 (89.63%) Instead of Phase-1 (92.80%)?**
   In Section *Results*, EfficientNetB0 achieved **92.80%** validation accuracy in Phase 1 (frozen) and dropped to **89.63%** in Phase 2 (fine-tuned), losing 11 validation images. Even ResNet50V2 achieved **90.20%** in Phase 2. Yet, all detailed per-class analyses (Table 4, Figures 4–6) and the TFLite export were performed on the degraded 89.63% Phase-2 EfficientNetB0 checkpoint, while Table 5 compares the 92.80% Phase-1 model against the literature.
   - *Action required:* Please clearly justify why the Phase-2 checkpoint (89.63%) was chosen for export and confusion-matrix analysis rather than the best overall checkpoint (Phase-1 EfficientNetB0 at 92.80%), or alternatively, update the exported model/evaluation to use the 92.80% checkpoint (or report both).
2. **Perform the McNemar's Test Mentioned on Page 6:**
   On Page 6, the authors state: *"Nevertheless, a paired statistical test on the per-image predictions, such as McNemar’s test, is still needed to determine whether this difference is significant."* Since you already have the per-image validation predictions for all three models on the 347 validation images, there is no reason to leave this as an undone hypothetical test. Computing exact McNemar's test $p$-values between the three architectures (and between Phase 1 and Phase 2 of each architecture) takes only a few lines of code and will substantially strengthen Table 3.
3. **Verify TFLite Quantized Model Accuracy:**
   On Page 10, the authors note that the model was converted to a 4.2 MB TFLite file using dynamic-range quantization, *"However, the accuracy of the quantized model and its on-device inference latency were not evaluated in this study."* Evaluating the `.tflite` file on the 347 validation images using `tf.lite.Interpreter` in Python takes less than a minute on a standard CPU. Please report the validation accuracy and Macro-F1 of the 4.2 MB `.tflite` model to confirm that quantization did not degrade classification performance.
4. **Clarify the Role of `BatchNormalization` Layers During Fine-Tuning:**
   The manuscript rightly notes that unfreezing all layers in EfficientNetB0 caused both training and validation accuracy to drop simultaneously, hypothesizing that mini-batch normalization statistics were disrupted. In `tf.keras`, setting `base_model.trainable = True` unfreezes `BatchNormalization` (BN) layers, causing them to update their moving mean and variance using small mini-batches ($B=16$). As documented by Chollet (2020), BN layers in Keras should normally be kept in inference mode (`training=False`) during fine-tuning. Please explicitly state in the *Methods* and *Discussion* that `BatchNormalization` layers were unfrozen in Phase 2 as part of testing "naive full-backbone unfreezing," and recommend freezing BN layers during Phase 2 as a concrete takeaway for practitioners.
5. **Clarify the *Sogatella furcifera* Directory Naming Issue (Page 4):**
   At the end of *Dataset construction* (Page 4), the authors write:
   > *"One additional issue concerns the class directory for Sogatella furcifera which contains a Thai spelling variant that affects the alphabetical ordering of class indices. Correcting the directory name may alter the assigned indices for two classes..."*
   This reads like an internal code note and raises concern for the reviewer: **Are the class labels in Table 4 and Figure 4 correctly mapped, or was there a label-index mismatch during your experiment?** If the labels were properly aligned during training and evaluation, please remove or rephrase this internal note into a general reproducibility note about preserving class-index mappings.

---

### 3. Minor Comments (Data, Citations, Cross-References, and Formatting)

1. **Factual Error in Thailand's Rice Cultivation Area (Page 2, Column 1):**
   The first sentence states: *"In Thailand, rice cultivation covers approximately 9,600 km2..."* This is off by a factor of 10. Thailand's rice cultivation area is approximately **60–65 million rai**, which equals **~9.6 to 10.4 million hectares** or **~96,000 to 104,000 $\text{km}^2$** (not $9,600\text{ km}^2$). It appears $9.6\text{ million ha}$ was misconverted to $9,600\text{ km}^2$ ($1\text{ ha} = 0.01\text{ km}^2$, so $9,600,000\text{ ha} = 96,000\text{ km}^2$). Also, format `km2` with a superscript (`km²`).
2. **Missing Reference in the Reference List:**
   On Page 4 (Column 2), the text cites `"(Wu et al., 2019 ; Zhang et al., 2023)"`, but **Zhang et al. (2023)** is missing from the *References* section. Please add the full bibliographic entry or remove the citation.
3. **Broken Section Cross-References:**
   The manuscript uses unnumbered section headings, but the text contains leftover numbered references:
   - Page 5, Column 1: *"necessary for the confusion matrix in **Section 4.3** to be valid"* -> change to *"in the per-class evaluation section"* (or number all sections consistently according to SUJST style).
   - Page 9, Column 2: *"under the uniform-error null model described in **Section 3.4**"* -> change to *"described in the Evaluation section"*.
4. **Missing In-Text Citation and Out-of-Order Numbering for Figure 4 and Figure 5:**
   - On Page 8 (Column 2), `Figure 5` is cited (`"The per-class results (Figure 5, Table 4)"`) **before** `Figure 4` is ever mentioned in the text.
   - In fact, **`Figure 4` (Confusion matrix) is never cited in the body text at all!** Please cite Figure 4 and Figure 5 in sequential order in the *Per-class behavior of the exported model* and *Taxonomic patterns in classification errors* subsections.
5. **Misspelling of University Name (Page 12):**
   In both the *Funding* and *Acknowledgements* sections on Page 12, the university name is spelled `"Valaya Aongkorn Rajabhat University"` (missing the `"l"` in `Alongkorn`). Please correct it to `"Valaya Alongkorn Rajabhat University"`.
6. **Future Dates in References (Pages 2, 12, 13):**
   - `Chollet, F. (2020)` lists `"Retrieved September 14, 2026"`.
   - `Office of Agricultural Economics (2026)` lists publication year `2026` for *Agricultural Statistics of Thailand 2025*. Please verify that these dates match the actual publication/retrieval years required by the journal.

---

### 4. Line-by-Line Language & Typographical Corrections
- **Page 1 (Abstract):** *"...indicate that fine-tuning does not necessarily improve performance, that its effects differ across architectures."* $\rightarrow$ Change to *"...improve performance, **and** that its effects differ across architectures."*
- **Page 2 (Introduction, Col 1):** *"While confusing a leaffolder caterpillar with a stem borer leads to treatments applied to the wrong part of the plant at the wrong time."* $\rightarrow$ Sentence fragment. Change `"While confusing..."` to `"Meanwhile, confusing..."` or join it with the preceding sentence.
- **Page 2 (Introduction, Col 1):** *"Those outbreaks attributed to insecticide misuse, accurate identification that supports targeted rather than prophylactic spraying has direct economic value."* $\rightarrow$ Fix grammar: *"Because those outbreaks were attributed to insecticide misuse, accurate identification..."*
- **Page 2 (Introduction, Col 2):** *"...encountered in applied research, a pretrained backbone is adapted..."* $\rightarrow$ Comma splice. Change to *"...encountered in applied research, **where** a pretrained backbone is adapted..."*
- **Page 2 (Introduction, Col 2):** `"(Tan & Le, 2019; Hu et al, 2018)"` $\rightarrow$ Missing period in `et al.`
- **Page 3 (Col 1):** *"Work restricted to rice pests report considerably higher accuracy"* $\rightarrow$ Subject-verb agreement: `"Work ... reports"` or `"Studies ... report"`.
- **Page 3 (Col 2, Heading):** `"Transfer learning and the fine-tuning"` $\rightarrow$ Remove `"the"` (`"Transfer learning and fine-tuning"`).
- **Page 4 (Col 1):** *"This setting commonly used in applied work."* $\rightarrow$ Missing verb: *"This setting **is** commonly used in applied work."*
- **Page 4 (Col 1):** *"with a maximum of 150 images per species per species"* $\rightarrow$ Remove duplicate `"per species"`.
- **Page 4 (Col 1):** *"helped revealed several naming issues"* $\rightarrow$ Change to *"helped **reveal** several naming issues"`.
- **Page 4 (Col 2):** *"a long-tailed distribution similar from IP102 (Wu et al., 2019 ; Zhang et al., 2023)"* $\rightarrow$ Change `"similar from"` to `"similar to"`, and remove the space before the semicolon.
- **Page 6 (Col 2):** *"Therefore, although EfficientNetB0 with frozen features was the leading model, but the available results do not establish..."* $\rightarrow$ Remove `"but"` (do not use `although` and `but` in the same clause).
- **Page 8 (Col 1–2):** *"with two and four validation images respectively, These small sample sizes limit..."* $\rightarrow$ Replace the comma after `respectively` with a period `.`.
- **Page 8 (Col 2):** *"validation images belonging of this species"* $\rightarrow$ Change to *"belonging **to** this species"`.
- **Page 9 (Col 1–2):** *"A further 18 errors (50.0%) A further 18 errors (50.0%) involved species..."* $\rightarrow$ Remove the duplicated phrase `"A further 18 errors (50.0%)"`.
- **Page 10 (Col 1):** *"One notable pattern involved Mythimna separata, a caterpillar, was twice predicted as Sogatella furcifera..."* $\rightarrow$ Change to *"...a caterpillar, **which** was twice predicted as..."*
- **Page 10 (Col 1):** *"However, the present the results do not establish..."* $\rightarrow$ Remove the extra `"the"` (*"the present results"*).

---
---

## ส่วนที่ 2: ฉบับภาษาไทย (ความเห็นผู้ทรงคุณวุฒิสำหรับวารสารระดับ Scopus Q4)

**วารสารเป้าหมาย:** *Suranaree Journal of Science and Technology (SUJST)* (Scopus Q4)  
**ผลการพิจารณาเบื้องต้น:** **ปรับปรุงแก้ไขหลัก (Major Revision)** — *งานวิจัยมีโครงสร้างและเนื้อหาที่ดีมาก เหมาะสมกับการตีพิมพ์ในวารสาร Scopus Q4 แต่มีจุดบกพร่องด้านการตรวจทานต้นฉบับ (Proofreading/Template leftovers) ที่ต้องแก้ไขอย่างเร่งด่วนก่อนส่งตีพิมพ์*

---

### 1. ภาพรวมและจุดเด่นของบทความ (General Assessment)
บทความวิจัยนี้ศึกษาเปรียบเทียบสถาปัตยกรรม CNN ที่นิยมใช้งาน 3 ตัว ได้แก่ MobileNetV2, EfficientNetB0 และ ResNet50V2 บนชุดข้อมูลภาพแมลงศัตรูข้าวในประเทศไทยจำนวน 17 ชนิด (รวม 1,763 ภาพ ที่ผ่านการตรวจสอบอนุกรมวิธานด้วย GBIF API) โดยควบคุมสภาวะการฝึกสอนแบบ 2 เฟส (Phase 1: แช่แข็ง Backbone และ Phase 2: ปลดล็อกทั้ง Backbone เพื่อทำ Fine-tuning) รวมถึงนำเสนอการวิเคราะห์ความผิดพลาดเชิงอนุกรมวิธาน (Taxonomic Error Analysis) เทียบกับ Null Model

**จุดแข็งที่ผู้ทรงคุณวุฒิชื่นชม:**
1. **ความโปร่งใสและความซื่อสัตย์ทางวิชาการ:** งานวิจัยส่วนใหญ่ในกลุ่มประยุกต์มักรายงานเฉพาะตัวเลขความแม่นยำที่สูงที่สุดเพียงค่าเดียว แต่บทความนี้รายงานผลทั้ง 2 เฟส รายงานช่วงความเชื่อมั่น 95% (Wilson Score Interval), ค่า MCC, Cohen’s $\kappa$ และชี้ให้เห็นว่า Fine-tuning แบบปลดล็อกทุกชั้นทำให้ EfficientNetB0 ประสิทธิภาพตกลงอย่างมีนัยสำคัญ ซึ่งมีประโยชน์ต่อผู้ปฏิบัติงานจริง
2. **การวิเคราะห์ความผิดพลาดเชิงอนุกรมวิธาน (Taxonomic Error Analysis):** การสร้างสมการที่ 7 (Null Model) และใช้การทดสอบ Exact Binomial Test เพื่อพิสูจน์ว่าความผิดพลาดของโมเดล 77.8% เกิดขึ้นภายในอันดับ (Order) เดียวกัน (สูงกว่าโอกาสเกิดแบบสุ่มที่ 37.5% อย่างมีนัยสำคัญทางสถิติ $p = 9.9 \times 10^{-7}$) เป็นวิธีตรวจสอบที่ฉลาด กระทัดรัด และเพิ่มคุณค่าทางวิชาการให้บทความโดดเด่นกว่างานเปรียบเทียบ CNN ทั่วไป
3. **การจัดการชุดข้อมูล (Taxonomic Verification):** การตรวจสอบชื่อวิทยาศาสตร์และชื่อพ้อง (Synonyms) ผ่าน GBIF API ก่อนดึงภาพ ช่วยลดปัญหา Label Noise ได้อย่างดี

อย่างไรก็ตาม **ต้นฉบับปัจจุบันยังไม่สามารถตอบรับการตีพิมพ์ได้ทันที** เนื่องจากพบ **ข้อผิดพลาดร้ายแรงจากการจัดหน้าและข้อความค้างจาก Template** รวมถึงประเด็นทางระเบียบวิธีวิจัยบางจุดที่ผู้วิจัยสามารถเพิ่มเติมได้ไม่ยาก จึงขอแนะนำให้ **Major Revision** ตามรายละเอียดด้านล่างนี้

---

### 2. ข้อคิดเห็นหลักที่ต้องแก้ไขอย่างเร่งด่วน (Major Comments)

#### ข้อที่ 1: ข้อความค้างจาก Template และการคัดลอกข้อความซ้ำ (สำคัญมาก เสี่ยงถูก Desk Reject หากไม่แก้)
1. **ข้อความคำสั่งค้างในหัวข้อ Conclusions (หน้า 11 คอลัมน์ขวา):**
   ย่อหน้าแรกของ *Conclusions* มีประโยคที่คัดลอกมาจาก *Discussion* และมีคำสั่งเตือนตัวเอง/คำสั่งจาก Template ค้างอยู่กลางประโยค ดังนี้:
   > *"The possibility that fine-tuning can degrade a pretrained model is not new and has been noted in framework documentation (Chollet, 2020). **State conclusion (do not** This study evaluated three CNN architectures..."*
   **แนวทางแก้ไข:** ลบประโยคแรกและคำว่า `"State conclusion (do not "` ออก แล้วขึ้นต้นบทสรุปด้วย `"This study evaluated three CNN architectures..."` ได้เลย
2. **หัวข้อย่อยซ้ำและประโยคขาดครึ่งในหน้า 10 (คอลัมน์ขวา):**
   ในหน้า 10 ก่อนขึ้นหัวข้อ *Discussion* มีหัวข้อย่อย `"Per-class behavior of the exported model"` โผล่ซ้ำขึ้นมาอีกครั้ง (ทั้งที่อยู่ในหน้า 8 แล้ว) แถมย่อหน้าแรกยังถูกวางซ้ำ 2 รอบติดกัน และรอบที่สองข้อความขาดหายไปกลางประโยค:
   > *"...which is the model that was exported, rather than on the 92.80%"*
   **แนวทางแก้ไข:** ลบหัวข้อย่อย `"Per-class behavior of the exported model"` และเนื้อหาทั้งสองย่อหน้าในหน้า 10 (คอลัมน์ขวา) ออกทั้งหมด
3. **ข้อความ Template ค้างที่หัวกระดาษ (Running Head) และหัวข้อ AI (หน้า 2–13):**
   - หัวกระดาษทุกหน้า (หน้า 2 ถึง 13) ยังเป็นข้อความคำแนะนำของวารสาร: *"Running Head: Provide a shortened version of the title, not exceeding 60 characters, including spaces. (9 pt)"* กรุณาเปลี่ยนเป็นชื่อเรื่องย่อจริง เช่น `FINE-TUNING EFFECTS ON RICE PEST CLASSIFICATION`
   - หน้า 12 หัวข้อ `"Declaration of Generative AI Use (12 pt Bold)"` มีคำว่า `(12 pt Bold)` ติดมาด้วย กรุณาลบออก
4. **Data Availability Statement ขัดแย้งกับเนื้อหาวิจัย (หน้า 12):**
   ในหน้า 12 ระบุว่า *"No new data were generated or analyzed in support of this research."* ซึ่งขัดแย้งกับตัวบทความที่รวบรวมและวิเคราะห์ภาพแมลง 1,763 ภาพจาก GBIF
   **แนวทางแก้ไข:** ปรับข้อความให้สอดคล้องกับความจริง เช่น ระบุว่าข้อมูลภาพดึงมาจากฐานข้อมูลสาธารณะ GBIF (พร้อมระบุวิธีเข้าถึงหรือลิงก์/DOI ของชุดข้อมูลตามเกณฑ์ของ GBIF)

#### ข้อที่ 2: ประเด็นด้านระเบียบวิธีวิจัยและการวิเคราะห์ผล (Methodological Comments)
1. **เหตุใดจึงเลือก Export และวิเคราะห์โมเดล EfficientNetB0 จาก Phase 2 (89.63%) แทนที่จะเป็น Phase 1 (92.80%)?**
   ในตารางที่ 3 โมเดลที่ได้ความแม่นยำสูงที่สุดคือ **EfficientNetB0 (Phase 1 - Frozen)** ที่ **92.80%** (และในตารางที่ 5 ผู้วิจัยก็นำตัวเลข 92.80% นี้ไปเปรียบเทียบกับงานวิจัยอื่น) แต่เมื่อนำโมเดลไปวัดผลรายคลาส (ตารางที่ 4, รูปที่ 4–6) และแปลงเป็นไฟล์ TFLite กลับนำโมเดล **Phase 2 (89.63%)** ที่ประสิทธิภาพตกลงไป 3.17% (ผิดเพิ่ม 11 ภาพ) มาใช้งาน
   **แนวทางแก้ไข:** กรุณาอธิบายเหตุผลให้ชัดเจนว่าทำไมจึงเลือก Checkpoint ของ Phase 2 มาวิเคราะห์รายคลาสและแปลงเป็น TFLite (เช่น เพื่อสะท้อนผลลัพธ์ของ Pipeline มาตรฐานที่จบด้วย Phase 2) หรือหากสามารถรายงานผลของ Checkpoint ที่ดีที่สุด (92.80%) ควบคู่กันได้ จะทำให้บทความสมบูรณ์และไม่มีข้อกังขาจากผู้อ่าน
2. **ควรคำนวณผล McNemar’s Test จริงแทนการกล่าวถึงเฉยๆ (หน้า 6):**
   ในหน้า 6 ผู้วิจัยระบุว่า *"Nevertheless, a paired statistical test on the per-image predictions, such as McNemar’s test, is still needed to determine whether this difference is significant."* เนื่องจากผู้วิจัยมีผลการทำนายรายภาพ (347 ภาพ) ของทุกโมเดลอยู่แล้ว การรัน McNemar's test ใน Python ใช้โค้ดเพียงไม่กี่บรรทัด ผู้ทรงคุณวุฒิแนะนำให้คำนวณและใส่ค่า $p$-value ของ McNemar's test ลงในเนื้อหาเลย จะช่วยยกระดับความน่าเชื่อถือของบทความขึ้นมาก
3. **การวัดความแม่นยำหลังแปลงเป็นไฟล์ TFLite (Quantized Model):**
   ในหน้า 10 ผู้วิจัยระบุว่าแปลงโมเดลเป็นไฟล์ TFLite ขนาด 4.2 MB ด้วย Dynamic-range quantization แต่ยังไม่ได้ทดสอบความแม่นยำของไฟล์ `.tflite` ดังกล่าว เนื่องจากผู้วิจัยสามารถใช้ `tf.lite.Interpreter` รันทดสอบกับภาพ Validation 347 ภาพบนคอมพิวเตอร์ได้ทันที (ใช้เวลาไม่ถึง 1 นาที) จึงแนะนำให้เพิ่มตัวเลข Accuracy / Macro-F1 ของไฟล์ `.tflite` เพื่อยืนยันว่าการทำ Quantization ไม่ทำให้ความแม่นยำตกลง
4. **การอธิบายกลไกของชั้น `BatchNormalization` ใน Keras ขณะทำ Fine-tuning:**
   ผู้วิจัยตั้งข้อสังเกตได้ดีมากว่าการปลดล็อกทุกชั้นของ EfficientNetB0 ใน Phase 2 ทำให้ทั้ง Training Accuracy และ Validation Accuracy ตกลงพร้อมกันเพราะสถิติของ Batch Normalization (BN) ถูกรบกวน ในทางเทคนิคของ `tf.keras` เมื่อสั่ง `base_model.trainable = True` ชั้น BN จะเปลี่ยนเข้าสู่โหมดปรับค่าเฉลี่ยและส่วนเบี่ยงเบนเคลื่อนที่ (Moving Mean/Variance) ตาม Batch ขนาดเล็ก (16 ภาพ) ซึ่งเอกสารของ Keras (Chollet, 2020) แนะนำให้ล็อกเฉพาะชั้น BN (`training=False`) แม้ในขณะทำ Fine-tuning ผู้วิจัยควรเสริมประเด็นนี้ในส่วน *Discussion* เพื่อเป็นข้อเสนอแนะเชิงเทคนิคที่ชัดเจนแก่ผู้อ่าน
5. **ความชัดเจนเรื่องชื่อโฟลเดอร์ภาษาไทยของ *Sogatella furcifera* (หน้า 4):**
   ในหน้า 4 ย่อหน้าสุดท้ายของหัวข้อ *Dataset construction* มีการกล่าวถึงปัญหาการสะกดชื่อโฟลเดอร์ภาษาไทยของ *Sogatella furcifera* ที่อาจทำให้ลำดับ Index ของคลาสสลับกัน ข้อความส่วนนี้เขียนเหมือนบันทึกช่วยจำในการแก้โค้ด (Debugging note) ซึ่งอาจทำให้ผู้อ่านกังวลว่า **ผลลัพธ์ในตารางที่ 4 และ Confusion Matrix ในรูปที่ 4 มีการสลับคลาสหรือไม่?**
   **แนวทางแก้ไข:** หากผู้วิจัยตรวจสอบและจัดการ Mapping ของคลาสถูกต้องแล้ว แนะนำให้ตัดหรือปรับข้อความส่วนนี้ให้เป็นการอธิบายหลักการจัดการ Class Index Mapping แบบวิชาการแทน เพื่อไม่ให้เกิดความเข้าใจผิด

---

### 3. ข้อคิดเห็นย่อยและการตรวจสอบข้อมูล/การอ้างอิง (Minor Comments)

1. **ตัวเลขพื้นที่ปลูกข้าวของประเทศไทยคลาดเคลื่อนไป 10 เท่า (หน้า 2 คอลัมน์ซ้าย):**
   ประโยคแรกของบทนำระบุว่า *"In Thailand, rice cultivation covers approximately 9,600 km2"* ตัวเลขนี้คลาดเคลื่อนไป 10 เท่าครับ พื้นที่ปลูกข้าวของไทยอยู่ที่ประมาณ **60–65 ล้านไร่** หรือคิดเป็นประมาณ **9.6 ล้านเฮกตาร์ (ha)** ซึ่งเมื่อแปลงเป็นตารางกิโลเมตรจะได้ประมาณ **96,000 $\text{km}^2$** (เนื่องจาก $1\text{ ha} = 0.01\text{ km}^2$) คาดว่าผู้วิจัยแปลงหน่วยจาก 9.6 ล้านเฮกตาร์เป็น $\text{km}^2$ ผิดหลักทศนิยม และอย่าลืมปรับ `km2` ให้เลข 2 เป็นตัวยก (`km²`)
2. **รายการอ้างอิงตกหล่น (Missing Reference):**
   ในหน้า 4 (คอลัมน์ขวา) มีการอ้างอิง `"(Wu et al., 2019 ; Zhang et al., 2023)"` แต่ในส่วน *References* ท้ายเล่ม **ไม่มีรายการของ Zhang et al. (2023)** กรุณาเพิ่มรายการอ้างอิงให้ครบถ้วน
3. **การอ้างอิงเลขหัวข้อ (Section Number) ที่ไม่มีอยู่จริงในบทความ:**
   ในต้นฉบับไม่ได้ใส่เลขกำกับหัวข้อ (Unnumbered headings) แต่ในเนื้อหากลับมีการอ้างถึงเลขหัวข้อเดิม:
   - หน้า 5 คอลัมน์ซ้าย: อ้างถึง `"Section 4.3"`
   - หน้า 9 คอลัมน์ขวา: อ้างถึง `"Section 3.4"`
   **แนวทางแก้ไข:** เปลี่ยนไปอ้างชื่อหัวข้อแทน หรือใส่เลขหัวข้อให้ตรงตามรูปแบบของวารสาร SUJST
4. **การอ้างอิงรูปภาพข้ามลำดับ และไม่มีการอ้างอิง Figure 4 ในเนื้อหา:**
   - ในหน้า 8 (คอลัมน์ขวา) มีการอ้างถึง `Figure 5` ก่อนที่จะมีการอ้างถึง `Figure 4`
   - ที่สำคัญคือ **ไม่มีการกล่าวถึง `Figure 4` (Confusion Matrix) ในเนื้อหาหลักเลย** กรุณาเพิ่มการอ้างอิง `Figure 4` ในหัวข้อ *Per-class behavior of the exported model* ก่อน `Figure 5`
5. **สะกดชื่อมหาวิทยาลัยผิดในหน้า 12:**
   ในหัวข้อ *Funding* และ *Acknowledgements* หน้า 12 สะกดคำว่า `"Valaya Aongkorn Rajabhat University"` (ตกตัว `l` ในคำว่า `Alongkorn` ไป 1 ตัว ในขณะที่หน้า 1 สะกดถูกต้องแล้ว) กรุณาแก้ไขให้ตรงกัน
6. **ตรวจสอบปี พ.ศ./ค.ศ. ในรายการอ้างอิง (หน้า 2, 12, 13):**
   - `Office of Agricultural Economics. (2026). Agricultural Statistics of Thailand 2025.`
   - `Chollet, F. (2020)... Retrieved September 14, 2026`
   กรุณาตรวจสอบความถูกต้องของปีที่พิมพ์และวันที่สืบค้นให้เป็นไปตามเกณฑ์ของวารสาร

---

### 4. จุดแก้ไขไวยากรณ์และคำผิดรายบรรทัด (Line-by-Line Corrections)
- **หน้า 1 (Abstract):** `"...improve performance, that its effects differ across architectures."` $\rightarrow$ เติมคำว่า **`and`** เป็น `"...improve performance, and that its effects differ across architectures."`
- **หน้า 2 (Introduction, คอลัมน์ซ้าย):** `"While confusing a leaffolder caterpillar with a stem borer leads to treatments applied to the wrong part of the plant at the wrong time."` $\rightarrow$ เป็นประโยคไม่สมบูรณ์ (Fragment) แนะนำให้เปลี่ยน `"While confusing..."` เป็น `"Meanwhile, confusing..."` หรือเชื่อมกับประโยคก่อนหน้า
- **หน้า 2 (Introduction, คอลัมน์ซ้าย):** `"Those outbreaks attributed to insecticide misuse, accurate identification..."` $\rightarrow$ ปรับไวยากรณ์เป็น `"Because those outbreaks were attributed to insecticide misuse, accurate identification..."`
- **หน้า 2 (Introduction, คอลัมน์ขวา):** `"(Tan & Le, 2019; Hu et al, 2018)"` $\rightarrow$ เติมจุดหลัง `et al.`
- **หน้า 3 (คอลัมน์ซ้าย):** `"Work restricted to rice pests report considerably higher accuracy"` $\rightarrow$ แก้เป็น `"Work ... reports"` หรือ `"Studies ... report"`
- **หน้า 3 (คอลัมน์ขวา, ชื่อหัวข้อ):** `"Transfer learning and the fine-tuning"` $\rightarrow$ ตัด `"the"` ออก เป็น `"Transfer learning and fine-tuning"`
- **หน้า 4 (คอลัมน์ซ้าย):** `"This setting commonly used in applied work."` $\rightarrow$ ตกคำกริยา `is` แก้เป็น `"This setting is commonly used in applied work."`
- **หน้า 4 (คอลัมน์ซ้าย):** `"with a maximum of 150 images per species per species"` $\rightarrow$ ลบคำว่า `"per species"` ที่พิมพ์ซ้ำออก 1 คำ
- **หน้า 4 (คอลัมน์ซ้าย):** `"helped revealed several naming issues"` $\rightarrow$ แก้เป็น `"helped reveal several naming issues"`
- **หน้า 4 (คอลัมน์ขวา):** `"similar from IP102 (Wu et al., 2019 ; Zhang et al., 2023)"` $\rightarrow$ แก้ `"similar from"` เป็น `"similar to"` และลบช่องว่างหน้าเครื่องหมาย `;`
- **หน้า 6 (คอลัมน์ขวา):** `"Therefore, although EfficientNetB0 with frozen features was the leading model, but the available results..."` $\rightarrow$ มีทั้ง `although` และ `but` ซ้อนกัน ให้ตัด `but` ออก
- **หน้า 8 (คอลัมน์ซ้าย–ขวา):** `"with two and four validation images respectively, These small sample sizes limit..."` $\rightarrow$ เปลี่ยนเครื่องหมายจุลภาค `,` หลัง `respectively` เป็นมหัพภาค `.`
- **หน้า 8 (คอลัมน์ขวา):** `"belonging of this species"` $\rightarrow$ แก้เป็น `"belonging to this species"`
- **หน้า 9 (คอลัมน์ซ้าย–ขวา):** `"A further 18 errors (50.0%) A further 18 errors (50.0%) involved species..."` $\rightarrow$ ลบข้อความ `"A further 18 errors (50.0%)"` ที่พิมพ์ซ้ำออก 1 ชุด
- **หน้า 10 (คอลัมน์ซ้าย):** `"One notable pattern involved Mythimna separata, a caterpillar, was twice predicted as Sogatella furcifera..."` $\rightarrow$ เติม `which` หน้า `was` เป็น `"...a caterpillar, which was twice predicted as..."`
- **หน้า 10 (คอลัมน์ซ้าย):** `"However, the present the results do not establish..."` $\rightarrow$ ลบ `"the"` ที่เกินมา 1 คำ เป็น `"However, the present results do not establish..."`

