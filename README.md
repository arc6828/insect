# 🌾 แพลตฟอร์มปัญญาประดิษฐ์บนอุปกรณ์พกพาสำหรับจำแนกแมลงศัตรูข้าวและระบบคำแนะนำเชิงปฏิบัติเรียลไทม์

โครงการวิจัยนี้มีวัตถุประสงค์เพื่อพัฒนาแบบจำลองการเรียนรู้เชิงลึก (Deep Learning) สำหรับจำแนกประเภทแมลงศัตรูข้าวของไทย เพื่อแปลงสภาพเป็นแบบจำลองประสิทธิภาพสูงขนาดเบา (TensorFlow Lite) สำหรับทำงานบนสมาร์ตโฟนของเกษตรกร พร้อมนำเสนอคำแนะนำป้องกันกำจัดการทำลายข้าวในแปลงแบบเรียลไทม์ โดยใช้วิธีดำเนินการวิจัยบน Google Colab

---

## 📊 รายละเอียดคลังภาพแมลงและการจับคู่อนุกรมวิธาน (Dataset Summary)

แบบจำลองถูกฝึกสอนจริงบน Google Drive ด้วยชุดข้อมูลที่ดาวน์โหลดเพิ่มเติมจาก **GBIF Occurrence API** ซึ่งขยายขนาดภาพขึ้นมาได้ทั้งหมด **1,763 ภาพ** ใน **17 คลาสสายพันธุ์** ส่วนอีก 5 คลาส (ด้วงงวงกินรากข้าว, ด้วงดำ, หนอนกอแถบลายสีม่วง, เพลี้ยจั๊กจั่นปีกลายหยัก, แมลงวันเจาะยอดข้าว) ไม่มีภาพถ่ายบน GBIF จึงถูกคัดออกจากกระบวนการฝึกสอนโมเดลเนื่องจากมีภาพไม่เพียงพอ ดังตารางต่อไปนี้:

| ลำดับ (No.) | ชื่อคลาสภาษาไทย | ชื่อวิทยาศาสตร์ 2 (ใหม่) | จำนวนภาพบน Google Drive (Drive Images) | GBIF Taxon Key | จำนวนภาพอ้างอิงบน GBIF | อันดับ & วงศ์ (Order / Family) |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | ด้วงงวงกินรากข้าว | *Hydronomidius molitor* Faust | 0 (ไม่มีในคลัง) | 1247989 | 0 | Coleoptera / Brachyceridae |
| 2 | ด้วงดำ | *Heteronychus lioderes* Redtenbacher | 0 (ไม่มีในคลัง) | 4995399 | 0 | Coleoptera / Scarabaeidae |
| 3 | มวนง่าม | *Tetroda denticulifera* (Berg) | 46 | 4783006 | 46 | Hemiptera / Pentatomidae |
| 4 | หนอนกระทู้กล้า | *Spodoptera mauritia* (Boisduval) | 150 | 5109848 | 2,488 | Lepidoptera / Noctuidae |
| 5 | หนอนกระทู้คอรวง | *Mythimna separata* (Walker) | 148 | 5802396 | 860 | Lepidoptera / Noctuidae |
| 6 | หนอนกอข้าวสีครีม | *Scirpophaga incertulas* (Walker) | 148 | 1881293 | 438 | Lepidoptera / Crambidae |
| 7 | หนอนกอสีชมพู | *Sesamia inferens* (Walker) | 65 | 1762353 | 96 | Lepidoptera / Noctuidae |
| 8 | หนอนกอแถบลายสีม่วง | *Chilo polychrysus* (Meyrick) | 0 (ไม่มีในคลัง) | 1883232 | 0 | Lepidoptera / Crambidae |
| 9 | หนอนกอแถบลายเล็ก | *Chilo suppressalis* (Walker) | 56 | 1883226 | 56 | Lepidoptera / Crambidae |
| 10 | หนอนปลอกข้าว | *Nymphula depunctalis* Guenée | 150 | 1884090 | 7,081 | Lepidoptera / Crambidae |
| 11 | หนอนห่อใบข้าว | *Cnaphalocrocis medinalis* (Guenée) | 146 | 1890320 | 2,241 | Lepidoptera / Crambidae |
| 12 | เพลี่ยกระโดดหลังขาว | *Sogatella furcifera* (Horvath) | 150 | 4777382 | 413 | Hemiptera / Delphacidae |
| 13 | เพลี้ยกระโดดสีน้ำตาล | *Nilaparvata lugens* (Stål) | 149 | 2056628 | 337 | Hemiptera / Delphacidae |
| 14 | เพลี้ยจักจั่นสีเขียว | *Nephotettix virescens* (Distant) | 75 | 4775727 | 76 | Hemiptera / Cicadellidae |
| 15 | เพลี้ยจั๊กจั่นปีกลายหยัก | *Recilia dorsalis* (Motsuchulsky) | 0 (ไม่มีในคลัง) | 2038818 | 0 | Hemiptera / Cicadellidae |
| 16 | เพลี้ยแป้ง | *Pseudococcus saccharicola* / *Trionymus* sp. | 54 | 2095307 | 54 | Hemiptera / Pseudococcidae |
| 17 | เพลี้ยไฟ | *Stenchaetothrips biformis* (Bagnall) | 144 | 4423994 | 277 | Thysanoptera / Thripidae |
| 18 | แมลงดำหนาม | *Dicladispa armigera* (Olivier) | 13 | 5876256 | 15 | Coleoptera / Chrysomelidae |
| 19 | แมลงบั่ว | *Orseolia oryzae* (Wood-Mason) | 22 | 1594735 | 22 | Diptera / Cecidomyiidae |
| 20 | แมลงวันเจาะยอดข้าว | *Hydrellia philippina* Ferino | 0 (ไม่มีในคลัง) | 1617662 | 0 | Diptera / Ephydridae |
| 21 | แมลงสิง | *Leptocorisa oratorius* (Fabricius) | 150 | 6543653 | 92 | Hemiptera / Alydidae |
| 22 | แมลงหล่า | *Scotinophara coarctata* (Fabricius) | 97 | 2079209 | 104 | Hemiptera / Pentatomidae |
| - | **รวม (Total)** | **17 คลาส (ดาวน์โหลดสำเร็จ)** | **1,763 ภาพ** | **-** | **14,577 ภาพ** | **-** |

---

## ⚙️ พารามิเตอร์การทดลองและอินพุตเชิงเทคนิค (Experimental Inputs & Settings)

ข้อมูลพารามิเตอร์เชิงเทคนิคทั้งหมดที่เกี่ยวข้องกับกระบวนการเตรียมรูปภาพนำเข้า (Data Preprocessing), การขยายข้อมูล (Data Augmentation) และค่าตัวแปรระดับสูง (Hyperparameters) ที่ใช้ในการฝึกสอนโมเดลถูกแยกบันทึกไว้ในเอกสารเฉพาะทาง:
👉 **[README_INPUTS.md](./README_INPUTS.md)**

---

## 📂 โครงสร้างโฟลเดอร์ของโครงการ (Repository Layout)

```text
e:/chavalit/colab/insect/
 ├── data/
 │   ├── recommendations.json      # ฐานข้อมูลคู่มือผู้เชี่ยวชาญการกำจัดแมลง 22 ชนิดภาษาไทย
 │   └── gbif_summary.json         # สถิติตัวเลข Taxonomic และจำนวนรูป StillImage บนคลังโลก
 ├── dataset/
 │   └── images/                   # คลังภาพตัวอย่างแมลงจำแนกแยกตามโฟลเดอร์ชื่อคลาสภาษาไทย (217 ภาพ)
 │       ├── ด้วงงวงกินรากข้าว/
 │       ├── ด้วงดำ/
 │       └── ... (ครบทั้งหมด 22 ชนิด)
 ├── insect.ipynb                  # สมุดบันทึกบน Google Colab สำหรับกระบวนการทดลองและแปลงโมเดล
 ├── README.md                     # [เอกสารนี้] สรุปจำนวนภาพและโครงสร้างคลังภาพของโครงการ
 ├── README_INPUTS.md              # พารามิเตอร์นำเข้า เทคนิค Preprocessing & Hyperparameters
 ├── README_ANALYSIS.md            # รายงานเปรียบเทียบผลลัพธ์การฝึกสอนโมเดลและการเปรียบเทียบเชิงลึกพร้อมกราฟ
 ├── output.png                    # ภาพกราฟแสดงการเปลี่ยนแปลง Accuracy & Loss ในรอบการฝึกสอน
 ├── RESEARCH_PAPER_DRAFT.md       # โครงร่างบทความวิจัยฉบับวิชาการภาษาไทยที่ปรับปรุงผลใหม่เรียบร้อย
 └── RESEARCH_PAPER_DRAFT_EN.md    # โครงร่างบทความวิจัยฉบับวิชาการภาษาอังกฤษ (English Version)
```

---

## 💡 ลิงก์อ่านรายละเอียดและผลลัพธ์การทดลองเพิ่มเติม
* ข้อกำหนดตัวแปรนำเข้าและ Preprocessing: 👉 **[README_INPUTS.md](./README_INPUTS.md)**
* การเปรียบเทียบค่าความแม่นยำและวิเคราะห์ Early Stopping: 👉 **[README_ANALYSIS.md](./README_ANALYSIS.md)**
* ร่างบทความวิจัยวิชาการ (ภาษาไทย): 👉 **[RESEARCH_PAPER_DRAFT.md](./RESEARCH_PAPER_DRAFT.md)**
* ร่างบทความวิจัยวิชาการ (ภาษาอังกฤษ): 👉 **[RESEARCH_PAPER_DRAFT_EN.md](./RESEARCH_PAPER_DRAFT_EN.md)**
